from __future__ import annotations

import unittest
from pathlib import Path

import torch

from distrisurg.config import ExperimentConfig, load_config
from distrisurg.data import StereoEndoscopyDataset, read_scene_list
from distrisurg.geometry import DistributionalSurfaceSplat, invert_rigid_transform
from distrisurg.geometry import projection_defect_taxonomy
from distrisurg.losses import DistriSurgLoss
from distrisurg.models import DistriSurg


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
    config.model.source_feature_channels = 4
    config.model.reliability_channels = 8
    config.dss.radius = 1
    config.dss.sigma_samples = (0.0,)
    config.dss.sigma_weights = (1.0,)
    config.model.trusted_support = 0.05
    return config.validate()


class ConfigAndDataTests(unittest.TestCase):
    def test_config_loads(self) -> None:
        value = load_config(PROJECT / "configs/distrisurg_dataset89.yaml")
        self.assertEqual(value.data.height, 512)
        self.assertEqual(len(value.dss.sigma_samples), 3)

    def test_partial_ablation_config_merges(self) -> None:
        value = load_config(
            PROJECT / "configs/distrisurg_dataset89.yaml",
            extra_paths=[PROJECT / "configs/ablations/a4_soft_splat.yaml"],
        )
        self.assertEqual(value.dss.sigma_samples, (0.0,))
        self.assertFalse(value.ablation.distributional_depth)

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
        trusted = output["trusted_mask"].expand_as(rgb)
        self.assertTrue(
            torch.equal(output["target_rgb"][trusted], output["warped_rgb"][trusted])
        )
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


if __name__ == "__main__":
    unittest.main()
