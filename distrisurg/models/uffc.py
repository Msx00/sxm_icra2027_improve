"""Compact uncertainty-conditioned Fourier U-Net.

This is a clean research implementation inspired by the global/local Fourier
design of LaMa and the motivation of UFFC.  It does not copy pretrained LaMa
weights; all weights are trained within the DistriSurg experiment.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _groups(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class FourierUnit(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.frequency_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, 1),
            nn.GroupNorm(_groups(channels * 2), channels * 2),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        original_dtype = value.dtype
        height, width = value.shape[-2:]
        spectrum = torch.fft.rfft2(value.float(), norm="ortho")
        spectrum = torch.view_as_real(spectrum)
        spectrum = spectrum.permute(0, 1, 4, 2, 3).contiguous()
        spectrum = spectrum.flatten(1, 2)
        spectrum = self.frequency_conv(spectrum)
        batch, twice_channels, freq_h, freq_w = spectrum.shape
        spectrum = spectrum.reshape(batch, twice_channels // 2, 2, freq_h, freq_w)
        spectrum = spectrum.permute(0, 1, 3, 4, 2).contiguous()
        spectrum = torch.view_as_complex(spectrum)
        output = torch.fft.irfft2(spectrum, s=(height, width), norm="ortho")
        return output.to(dtype=original_dtype)


class ConditionAffine(nn.Module):
    def __init__(self, condition_channels: int, channels: int) -> None:
        super().__init__()
        self.affine = nn.Sequential(
            nn.Conv2d(condition_channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels * 2, 1),
        )
        nn.init.zeros_(self.affine[-1].weight)
        nn.init.zeros_(self.affine[-1].bias)

    def forward(
        self, value: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        condition = F.interpolate(
            condition, size=value.shape[-2:], mode="bilinear", align_corners=False
        )
        gamma, beta = self.affine(condition).chunk(2, dim=1)
        return value * (1.0 + 0.25 * torch.tanh(gamma)) + beta


class UFFCBlock(nn.Module):
    def __init__(self, channels: int, condition_channels: int) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
        )
        self.local = nn.Conv2d(channels, channels, 3, padding=1)
        self.global_fourier = FourierUnit(channels)
        self.mix = nn.Conv2d(channels * 2, channels, 1)
        self.condition = ConditionAffine(condition_channels, channels)
        self.output = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(
        self, value: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.pre(value)
        mixed = self.mix(
            torch.cat((self.local(normalized), self.global_fourier(normalized)), dim=1)
        )
        mixed = self.condition(mixed, condition)
        return value + self.output(mixed)


class DecoderStage(nn.Module):
    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        output_channels: int,
        condition_channels: int,
    ) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            input_channels + skip_channels, output_channels, 3, padding=1
        )
        self.block = UFFCBlock(output_channels, condition_channels)

    def forward(
        self,
        value: torch.Tensor,
        skip: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        value = F.interpolate(
            value, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.block(self.projection(torch.cat((value, skip), dim=1)), condition)


class UncertaintyConditionedUFFC(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        base_channels: int = 32,
        condition_channels: int = 6,
    ) -> None:
        super().__init__()
        base = base_channels
        self.stem = nn.Conv2d(input_channels, base, 5, padding=2)
        self.enc0 = UFFCBlock(base, condition_channels)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.enc1 = UFFCBlock(base * 2, condition_channels)
        self.down2 = nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1)
        self.enc2 = UFFCBlock(base * 4, condition_channels)
        self.down3 = nn.Conv2d(base * 4, base * 8, 3, stride=2, padding=1)
        self.bottleneck = nn.ModuleList(
            [UFFCBlock(base * 8, condition_channels) for _ in range(2)]
        )
        self.dec2 = DecoderStage(base * 8, base * 4, base * 4, condition_channels)
        self.dec1 = DecoderStage(base * 4, base * 2, base * 2, condition_channels)
        self.dec0 = DecoderStage(base * 2, base, base, condition_channels)
        self.output = nn.Conv2d(base, output_channels, 3, padding=1)

    def forward(
        self, value: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        skip0 = self.enc0(self.stem(value), condition)
        skip1 = self.enc1(self.down1(skip0), condition)
        skip2 = self.enc2(self.down2(skip1), condition)
        value = self.down3(skip2)
        for block in self.bottleneck:
            value = block(value, condition)
        value = self.dec2(value, skip2, condition)
        value = self.dec1(value, skip1, condition)
        value = self.dec0(value, skip0, condition)
        return self.output(value)
