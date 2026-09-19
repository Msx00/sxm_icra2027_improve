from __future__ import annotations

import unittest
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from distrisurg.config import ExperimentConfig, load_config
from distrisurg.data import StereoEndoscopyDataset, read_scene_list
from distrisurg.geometry import DistributionalSurfaceSplat, invert_rigid_transform
from distrisurg.geometry import projection_defect_taxonomy
from distrisurg.losses import (
    DistriSurgLoss,
    MaskedPerceptualLoss,
    multi_scale_low_frequency_loss,
)
from distrisurg.models import DistriSurg
from distrisurg.models.distrisurg import spatially_consistent_trusted_mask
from distrisurg.models.uffc import UncertaintyConditionedUFFC
from distrisurg.postprocess import repair_small_synthesis_regions


PROJECT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = Path("/home/data/mashixing/dataset_8tb/iMed/datasets/task2-nvs")
EVAL_ROOT = Path(
    "/home/data/mashixing/dataset_18tb/"
    "icra2027-diffusion-zero-test-dataset/endovis/dataset89"
)


def tiny_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.data.height = 32
    config.data.width = 40
    config.model.base_channels = 8
    config.model.synthesis_bottleneck_blocks = 1
    config.model.source_feature_channels = 4
    config.model.reliability_channels = 8
    config.dss.radius = 1
    config.dss.sigma_samples = (0.0,)
    config.dss.sigma_weights = (1.0,)
    config.model.trusted_support = 0.05
    return config.validate()


class ConfigAndDataTests(unittest.TestCase):
    def test_config_loads(self) -> None:
        value = load_config(PROJECT / "configs/train.yaml")
        self.assertEqual(value.data.height, 512)
        self.assertEqual(len(value.dss.sigma_samples), 3)
        self.assertEqual(value.train.amp_dtype, "bfloat16")
        self.assertEqual(value.model.synthesis_bottleneck_blocks, 6)
        self.assertEqual(value.loss.low_frequency, 0.2)
        self.assertEqual(value.loss.perceptual, 0.1)

    def test_new_options_load_from_override(self) -> None:
        value = load_config(
            overrides=[
                "model.synthesis_bottleneck_blocks=3",
                "loss.low_frequency=0.4",
                "loss.perceptual=0",
            ]
        )
        self.assertEqual(value.model.synthesis_bottleneck_blocks, 3)
        self.assertEqual(value.loss.low_frequency, 0.4)
        self.assertEqual(value.loss.perceptual, 0.0)

    def test_invalid_bottleneck_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_config(overrides=["model.synthesis_bottleneck_blocks=0"])

    def test_main_train_config_uses_soft_composition(self) -> None:
        value = load_config(PROJECT / "configs/train.yaml")
        self.assertFalse(value.ablation.hard_composition)

    @unittest.skipUnless(TRAIN_ROOT.is_dir(), "local iMED training data unavailable")
    def test_training_split_and_fixed_pose(self) -> None:
        names = read_scene_list(PROJECT / "configs/train_scenes.txt")
        dataset = StereoEndoscopyDataset(
            TRAIN_ROOT, height=32, width=40, scene_names=names[:1]
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["source_rgb"].shape), (3, 32, 40))
        self.assertEqual(tuple(sample["source_to_target"].shape), (4, 4))

    @unittest.skipUnless(EVAL_ROOT.is_dir(), "local dataset89 unavailable")
    def test_dataset89_per_frame_pose(self) -> None:
        dataset = StereoEndoscopyDataset(
            EVAL_ROOT,
            height=32,
            width=40,
            scene_names=["endovis_dataset_8_keyframe_1"],
        )
        sample = dataset[0]
        self.assertEqual(sample["scene"], "endovis_dataset_8_keyframe_1")
        self.assertTrue(sample["source_depth_valid"].any())


class GeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = tiny_config()
        self.renderer = DistributionalSurfaceSplat(self.config.dss)
        self.height, self.width = 24, 32
        yy, xx = torch.meshgrid(
            torch.arange(self.height), torch.arange(self.width), indexing="ij"
        )
        self.rgb = torch.stack(
            (
                xx.float() / self.width,
                yy.float() / self.height,
                torch.full_like(xx, 0.5).float(),
            )
        ).unsqueeze(0)
        self.depth = torch.full((1, 1, self.height, self.width), 100.0)
        self.valid = torch.ones_like(self.depth, dtype=torch.bool)
        self.sigma = torch.full_like(self.depth, 0.25)
        self.confidence = torch.ones_like(self.depth)
        self.intrinsics = torch.tensor(
            [
                [80.0, 0.0, self.width / 2],
                [0.0, 80.0, self.height / 2],
                [0.0, 0.0, 1.0],
            ]
        ).unsqueeze(0)
        self.transform = torch.eye(4).unsqueeze(0)

    def render(self, transform: torch.Tensor | None = None):
        return self.renderer(
            self.rgb,
            self.depth,
            self.sigma,
            self.valid,
            self.confidence,
            self.intrinsics,
            self.intrinsics,
            self.transform if transform is None else transform,
        )

    def test_identity_coverage_and_color(self) -> None:
        output = self.render()
        self.assertGreater(output["valid_mask"].float().mean().item(), 0.95)
        mask = output["valid_mask"].expand_as(self.rgb)
        error = (output["features"] - self.rgb).abs()[mask].mean()
        self.assertLess(error.item(), 0.03)

    def test_translation_exposes_border(self) -> None:
        transform = self.transform.clone()
        transform[:, 0, 3] = 10.0
        output = self.render(transform)
        holes = ~output["valid_mask"]
        self.assertGreater(holes.float().mean().item(), 0.05)
        self.assertGreater(
            holes[..., :6].float().mean().item(),
            holes[..., -6:].float().mean().item(),
        )

    def test_feature_gradient(self) -> None:
        rgb = self.rgb.clone().requires_grad_(True)
        output = self.renderer(
            rgb,
            self.depth,
            self.sigma,
            self.valid,
            self.confidence,
            self.intrinsics,
            self.intrinsics,
            self.transform,
        )
        output["features"].mean().backward()
        self.assertIsNotNone(rgb.grad)
        self.assertTrue(torch.isfinite(rgb.grad).all())

    def test_camera_plane_points_do_not_poison_depth_gradient(self) -> None:
        depth = torch.full_like(self.depth, 100.0)
        depth[..., : self.width // 2] = 99.0
        depth.requires_grad_(True)
        transform = self.transform.clone()
        transform[:, 2, 3] = -99.0
        output = self.renderer(
            self.rgb,
            depth,
            self.sigma,
            self.valid,
            self.confidence,
            self.intrinsics,
            self.intrinsics,
            transform,
        )
        output["support"].sum().backward()
        self.assertIsNotNone(depth.grad)
        self.assertTrue(torch.isfinite(depth.grad).all())

    def test_rigid_inverse(self) -> None:
        transform = self.transform.clone()
        transform[:, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
        identity = transform @ invert_rigid_transform(transform)
        self.assertTrue(torch.allclose(identity, torch.eye(4).unsqueeze(0)))

    def test_projection_taxonomy_is_mutually_exclusive(self) -> None:
        raw = torch.tensor([[[[1, 0, 0, 1]]]], dtype=torch.bool)
        completed = torch.tensor([[[[1, 1, 0, 1]]]], dtype=torch.bool)
        variance = torch.tensor([[[[0.0, 0.0, 0.0, 20.0]]]])
        entropy = torch.zeros_like(variance)
        labels = projection_defect_taxonomy(raw, completed, variance, entropy, 9.0, 0.5)
        self.assertEqual(tuple(labels.shape), tuple(raw.shape))
        self.assertTrue(((labels >= 0) & (labels <= 4)).all())


class ModelTests(unittest.TestCase):
    def test_configurable_synthesis_bottleneck_depth(self) -> None:
        for blocks in (1, 3):
            model = UncertaintyConditionedUFFC(
                10, 5, base_channels=8, condition_channels=6,
                bottleneck_blocks=blocks,
            )
            value = torch.rand(1, 10, 16, 16, requires_grad=True)
            condition = torch.rand(1, 6, 16, 16)
            output = model(value, condition)
            self.assertEqual(tuple(output.shape), (1, 5, 16, 16))
            output.mean().backward()
            self.assertTrue(torch.isfinite(value.grad).all())

    def test_small_region_repair_preserves_large_holes(self) -> None:
        prediction = torch.zeros((1, 3, 20, 20))
        warp = torch.full_like(prediction, 0.5)
        trusted = torch.ones((1, 1, 20, 20), dtype=torch.bool)
        trusted[..., 2, 2] = False
        trusted[..., 10:15, 10:15] = False
        repaired, mask = repair_small_synthesis_regions(
            prediction, warp, trusted, max_area_ratio=0.01, radius=1.0
        )
        self.assertTrue(mask[..., 2, 2].item())
        self.assertFalse(mask[..., 12, 12].item())
        self.assertGreater(repaired[..., 2, 2].mean().item(), 0.4)
        self.assertEqual(repaired[..., 12, 12].sum().item(), 0.0)

    def test_trust_cleanup_removes_islands_but_preserves_invalid_holes(self) -> None:
        raw_trusted = torch.ones((1, 1, 21, 21), dtype=torch.bool)
        raw_trusted[..., 4, 4] = False
        raw_trusted[..., 10:17, 10:17] = False
        render_valid = torch.ones_like(raw_trusted)
        render_valid[..., 2, 15] = False
        cleaned = spatially_consistent_trusted_mask(
            raw_trusted, render_valid, kernel_size=7
        )
        self.assertTrue(cleaned[..., 4, 4].item())
        self.assertFalse(cleaned[..., 13, 13].item())
        self.assertFalse(cleaned[..., 2, 15].item())

    def test_forward_loss_and_hard_composition(self) -> None:
        config = tiny_config()
        model = DistriSurg(config)
        batch, height, width = 1, config.data.height, config.data.width
        rgb = torch.rand(batch, 3, height, width)
        depth = torch.full((batch, 1, height, width), 100.0)
        valid = torch.ones_like(depth, dtype=torch.bool)
        intrinsics = torch.tensor(
            [[50.0, 0.0, width / 2], [0.0, 50.0, height / 2], [0.0, 0.0, 1.0]]
        ).unsqueeze(0)
        transform = torch.eye(4).unsqueeze(0)
        output = model(
            rgb, depth, valid, intrinsics, intrinsics, transform, return_cycle=True
        )
        self.assertEqual(tuple(output["target_rgb"].shape), tuple(rgb.shape))
        self.assertTrue(torch.isfinite(output["synthesis_logits"]).all())
        trusted = output["trusted_mask"].expand_as(rgb)
        self.assertTrue(
            torch.equal(output["target_rgb"][trusted], output["warped_rgb"][trusted])
        )
        self.assertIn("raw_trusted_mask", output)
        criterion = DistriSurgLoss(config)
        losses = criterion(
            output,
            {
                "source_rgb": rgb,
                "source_depth": depth,
                "source_depth_valid": valid,
                "target_rgb": rgb,
                "target_depth": depth,
                "target_depth_valid": valid,
                "target_tool_mask": torch.zeros_like(valid),
            },
        )
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()

    def test_router_loss_is_stable_for_saturated_logits(self) -> None:
        config = tiny_config()
        criterion = DistriSurgLoss(config)
        height, width = config.data.height, config.data.width
        prediction = torch.full((1, 3, height, width), 0.5, requires_grad=True)
        logits = torch.empty((1, 1, height, width))
        logits[..., ::2] = -20.0
        logits[..., 1::2] = 20.0
        logits.requires_grad_()
        zeros = torch.zeros((1, 1, height, width))
        bool_zeros = zeros.bool()
        output = {
            "target_rgb": prediction,
            "hole_mask": ~bool_zeros,
            "trusted_mask": bool_zeros,
            "warped_rgb": prediction.detach(),
            "source_depth_mean": torch.full_like(zeros, 100.0),
            "source_depth_sigma": torch.ones_like(zeros),
            "target_depth": torch.full_like(zeros, 100.0),
            "support": zeros,
            "render_variance": zeros,
            "collision_entropy": zeros,
            "synthesis_logits": logits,
            "risk": torch.full_like(zeros, 0.5),
        }
        losses = criterion(
            output,
            {
                "source_rgb": prediction.detach(),
                "source_depth": torch.full_like(zeros, 100.0),
                "source_depth_valid": ~bool_zeros,
                "target_rgb": prediction.detach(),
                "target_depth": torch.full_like(zeros, 100.0),
                "target_depth_valid": ~bool_zeros,
                "target_tool_mask": bool_zeros,
            },
        )
        losses["total"].backward()
        self.assertTrue(torch.isfinite(losses["router"]))
        self.assertTrue(torch.isfinite(logits.grad).all())


class AppearanceLossTests(unittest.TestCase):
    def test_low_frequency_identity_difference_empty_and_backward(self) -> None:
        target = torch.rand(1, 3, 16, 20)
        mask = torch.zeros(1, 1, 16, 20, dtype=torch.bool)
        mask[..., 4:12, 5:15] = True
        identical = target.clone().requires_grad_(True)
        identity_loss = multi_scale_low_frequency_loss(
            identical, target, mask, sigmas=(1.0, 2.0)
        )
        self.assertLess(identity_loss.item(), 1.1e-3)
        prediction = (target + 0.2).requires_grad_(True)
        different_loss = multi_scale_low_frequency_loss(
            prediction, target, mask, sigmas=(1.0, 2.0)
        )
        self.assertGreater(different_loss.item(), 0.0)
        different_loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        empty_loss = multi_scale_low_frequency_loss(
            prediction, target, torch.zeros_like(mask), sigmas=(1.0,)
        )
        self.assertTrue(torch.isfinite(empty_loss))
        self.assertEqual(empty_loss.item(), 0.0)

    def test_masked_perceptual_loss_without_external_weights(self) -> None:
        class TinyFeatures(nn.Module):
            def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
                return [value, F.avg_pool2d(value, 2)]

        criterion = MaskedPerceptualLoss(TinyFeatures())
        target = torch.rand(1, 3, 16, 20)
        prediction = (target + 0.1).requires_grad_(True)
        mask = torch.zeros(1, 1, 16, 20, dtype=torch.bool)
        mask[..., 3:13, 4:16] = True
        loss = criterion(prediction, target, mask)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        empty = criterion(prediction, target, torch.zeros_like(mask))
        self.assertTrue(torch.isfinite(empty))
        self.assertEqual(empty.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
