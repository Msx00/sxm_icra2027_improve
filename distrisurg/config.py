"""Typed experiment configuration with strict validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    train_root: str = "/home/data/mashixing/dataset_8tb/iMed/datasets/task2-nvs"
    train_scenes_file: str = (
        "/home/data/mashixing/dataset_8tb/iMed/comparison/"
        "task2-icra/train_scenes.txt"
    )
    eval_root: str = (
        "/home/data/mashixing/dataset_18tb/"
        "icra2027-diffusion-zero-test-dataset/endovis/dataset89"
    )
    height: int = 512
    width: int = 640
    min_overlap: float = 0.5
    num_workers: int = 4


@dataclass
class DSSConfig:
    sigma_samples: tuple[float, ...] = (-1.0, 0.0, 1.0)
    sigma_weights: tuple[float, ...] = (0.2741, 0.4518, 0.2741)
    min_depth_mm: float = 1.0
    max_depth_mm: float = 500.0
    min_sigma_mm: float = 0.25
    max_sigma_mm: float = 20.0
    base_footprint_px: float = 0.75
    normal_footprint_px: float = 0.75
    max_footprint_px: float = 2.5
    radius: int = 1
    visibility_temperature_mm: float = 2.0
    min_support: float = 0.03


@dataclass
class ModelConfig:
    base_channels: int = 32
    synthesis_bottleneck_blocks: int = 6
    source_feature_channels: int = 16
    reliability_channels: int = 24
    max_depth_residual_ratio: float = 0.10
    invalid_depth_confidence_scale: float = 0.35
    trusted_support: float = 0.35
    trusted_variance_mm2: float = 9.0
    trusted_entropy: float = 1.0
    router_residual_scale: float = 0.25
    # Remove sub-footprint salt-and-pepper synthesis islands before hard
    # composition. One disables cleanup; otherwise the value must be odd.
    trust_cleanup_kernel: int = 7


@dataclass
class LossConfig:
    hole: float = 3.0
    visible: float = 1.0
    seam: float = 2.0
    gradient: float = 0.2
    low_frequency: float = 0.0
    perceptual: float = 0.0
    perceptual_weights_path: str = ""
    depth_nll: float = 0.2
    cycle: float = 0.2
    router: float = 0.1
    uncertainty: float = 0.1
    seam_width: int = 7


@dataclass
class TrainConfig:
    seed: int = 6666
    steps: int = 10000
    batch_size: int = 1
    accumulation_steps: int = 4
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-4
    save_every: int = 500
    log_every: int = 20
    mixed_precision: bool = True
    amp_dtype: str = "bfloat16"
    depth_dropout: float = 0.15
    grad_clip: float = 1.0
    forbid_eval_scene_training: bool = True


@dataclass
class AblationConfig:
    depth_completion: bool = True
    distributional_depth: bool = True
    anisotropic_footprint: bool = True
    physics_router: bool = True
    hard_composition: bool = True
    rgbd_cycle: bool = True


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    dss: DSSConfig = field(default_factory=DSSConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)

    def validate(self) -> "ExperimentConfig":
        if min(self.data.height, self.data.width) <= 0:
            raise ValueError("image dimensions must be positive")
        if self.data.height % 8 or self.data.width % 8:
            raise ValueError("height and width must be multiples of 8")
        if not 0.0 <= self.data.min_overlap < 1.0:
            raise ValueError("min_overlap must be in [0, 1)")
        if len(self.dss.sigma_samples) != len(self.dss.sigma_weights):
            raise ValueError("sigma_samples and sigma_weights must have equal length")
        if not self.dss.sigma_samples:
            raise ValueError("at least one sigma sample is required")
        if abs(sum(self.dss.sigma_weights) - 1.0) > 1.0e-3:
            raise ValueError("sigma_weights must sum to one")
        if self.dss.radius < 1 or self.dss.radius > 4:
            raise ValueError("DSS radius must be between 1 and 4")
        if self.dss.visibility_temperature_mm <= 0:
            raise ValueError("visibility temperature must be positive")
        if self.model.base_channels < 8:
            raise ValueError("base_channels is too small")
        if not 1 <= self.model.synthesis_bottleneck_blocks <= 18:
            raise ValueError("synthesis_bottleneck_blocks must be between 1 and 18")
        if (
            self.model.trust_cleanup_kernel < 1
            or self.model.trust_cleanup_kernel > 15
            or self.model.trust_cleanup_kernel % 2 == 0
        ):
            raise ValueError("trust_cleanup_kernel must be an odd integer in [1, 15]")
        if not 0.0 <= self.train.depth_dropout < 1.0:
            raise ValueError("depth_dropout must be in [0, 1)")
        if self.train.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("amp_dtype must be 'float16' or 'bfloat16'")
        for name in (
            "hole", "visible", "seam", "gradient", "low_frequency",
            "perceptual", "depth_nll", "cycle", "router", "uncertainty",
        ):
            if getattr(self.loss, name) < 0:
                raise ValueError(f"loss.{name} must be non-negative")
        if self.loss.perceptual > 0 and not self.loss.perceptual_weights_path:
            raise ValueError(
                "loss.perceptual_weights_path must name a local VGG16 checkpoint "
                "when loss.perceptual > 0"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    known = set(instance.__dataclass_fields__)
    unknown = set(values).difference(known)
    if unknown:
        raise KeyError(
            f"unknown keys for {type(instance).__name__}: {sorted(unknown)}"
        )
    for key, value in values.items():
        current = getattr(instance, key)
        if hasattr(current, "__dataclass_fields__"):
            if not isinstance(value, dict):
                raise TypeError(f"{key} must be a mapping")
            _merge_dataclass(current, value)
        else:
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(instance, key, value)
    return instance


def _coerce_override(value: str) -> Any:
    return yaml.safe_load(value)


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
    extra_paths: list[str | Path] | None = None,
) -> ExperimentConfig:
    config = ExperimentConfig()
    paths = ([path] if path else []) + list(extra_paths or [])
    for config_path in paths:
        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise TypeError("configuration root must be a mapping")
        _merge_dataclass(config, payload)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must have KEY=VALUE form: {item!r}")
        dotted, raw = item.split("=", 1)
        parts = dotted.split(".")
        target: Any = config
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise KeyError(f"unknown override: {dotted}")
            target = getattr(target, part)
        if not hasattr(target, parts[-1]):
            raise KeyError(f"unknown override: {dotted}")
        value = _coerce_override(raw)
        current = getattr(target, parts[-1])
        if isinstance(current, tuple) and isinstance(value, list):
            value = tuple(value)
        setattr(target, parts[-1], value)
    return config.validate()
