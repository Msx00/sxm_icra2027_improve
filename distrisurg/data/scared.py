"""Strict reader for the prepared SCARED/EndoVis stereo-pair layout."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


FRAME_PATTERN = re.compile(r"frame_(\d+)$")


def frame_id(path: Path) -> int:
    match = FRAME_PATTERN.match(path.stem)
    if match is None:
        raise ValueError(f"invalid frame name: {path}")
    return int(match.group(1))


def discover_scenes(root: str | Path) -> list[Path]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    scenes = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and (path / "K.txt").is_file()
        and (
            (path / "pose_pairs.txt").is_file()
            or (path / "pose.txt").is_file()
        )
    )
    if not scenes:
        raise RuntimeError(f"no stereo scenes found below {root}")
    return scenes


def read_intrinsics(path: str | Path) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    label: str | None = None
    rows: list[list[float]] = []

    def commit() -> None:
        nonlocal rows
        if label is None or not rows:
            return
        matrix = np.asarray(rows, dtype=np.float32)
        if matrix.shape != (3, 3):
            raise ValueError(f"{label} in {path} is not 3x3")
        matrices[label] = matrix
        rows = []

    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            commit()
            words = line[1:].strip().split()
            label = words[0] if words and words[0].startswith("K") else None
        elif label is not None:
            rows.append([float(value) for value in line.split()])
    commit()
    missing = {"K1_L", "K2_L"}.difference(matrices)
    if missing:
        raise KeyError(f"{path} misses {sorted(missing)}")
    return matrices


def read_pose_pairs(path: str | Path) -> dict[int, np.ndarray]:
    transforms: dict[int, np.ndarray] = {}
    for line_number, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 13:
            raise ValueError(f"invalid 3x4 pose at {path}:{line_number}")
        identifier = int(values[0])
        transform = np.eye(4, dtype=np.float32)
        transform[:3] = np.asarray(values[1:], dtype=np.float32).reshape(3, 4)
        transforms[identifier] = transform
    if not transforms:
        raise RuntimeError(f"no pose pairs in {path}")
    return transforms


def _quaternion_xyzw_to_matrix(values: list[float]) -> np.ndarray:
    x, y, z, w = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm([w, x, y, z])
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("zero-length quaternion")
    w, x, y, z = np.asarray([w, x, y, z]) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def read_fixed_source_to_target(path: str | Path) -> np.ndarray:
    """Read iMED pose.txt and return inv(c2w_target) @ c2w_source."""
    poses: dict[int, np.ndarray] = {}
    for line_number, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 8:
            raise ValueError(f"invalid pose at {path}:{line_number}")
        camera = int(values[0])
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = _quaternion_xyzw_to_matrix(values[4:8])
        transform[:3, 3] = np.asarray(values[1:4], dtype=np.float32)
        poses[camera] = transform
    if 0 not in poses or 1 not in poses:
        raise KeyError(f"{path} must contain camera 0 (source) and 1 (target)")
    return (np.linalg.inv(poses[1]) @ poses[0]).astype(np.float32)


def read_scene_list(path: str | Path) -> list[str]:
    values = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        value = raw.split("#", 1)[0].strip()
        if value:
            values.append(value)
    if not values:
        raise RuntimeError(f"empty scene list: {path}")
    if len(values) != len(set(values)):
        raise RuntimeError(f"duplicate names in scene list: {path}")
    return values


def scale_intrinsics(
    matrix: np.ndarray,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> np.ndarray:
    source_h, source_w = source_hw
    target_h, target_w = target_hw
    scale = np.asarray(
        [
            [target_w / source_w, 0.0, 0.0],
            [0.0, target_h / source_h, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return scale @ np.asarray(matrix, dtype=np.float32)


def _indexed_files(directory: Path, suffix: str) -> dict[int, Path]:
    return {
        frame_id(path): path
        for path in sorted(directory.glob(f"frame_*{suffix}"))
    }


@dataclass(frozen=True)
class StereoSample:
    scene: str
    frame_id: int
    source_rgb: Path
    source_depth: Path
    target_rgb: Path
    target_depth: Path | None
    target_tool_mask: Path | None
    source_intrinsics: np.ndarray
    target_intrinsics: np.ndarray
    source_to_target: np.ndarray


def _select_scene_names(
    scenes: Iterable[Path], selected: set[str] | None
) -> list[Path]:
    values = [path for path in scenes if selected is None or path.name in selected]
    if selected is not None:
        found = {path.name for path in values}
        missing = selected.difference(found)
        if missing:
            raise FileNotFoundError(f"requested scenes not found: {sorted(missing)}")
    return values


class StereoEndoscopyDataset(Dataset):
    """Paired right-to-left samples with no hidden frame alignment heuristics."""

    def __init__(
        self,
        root: str | Path,
        height: int = 512,
        width: int = 640,
        scene_names: Iterable[str] | None = None,
        require_target: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.height = int(height)
        self.width = int(width)
        self.require_target = bool(require_target)
        if min(self.height, self.width) <= 0:
            raise ValueError("height and width must be positive")
        selected = set(scene_names) if scene_names is not None else None
        scene_paths = _select_scene_names(discover_scenes(self.root), selected)
        self.samples: list[StereoSample] = []
        self.scene_metadata: dict[str, dict] = {}

        for scene in scene_paths:
            intrinsics = read_intrinsics(scene / "K.txt")
            pose_pairs_path = scene / "pose_pairs.txt"
            if pose_pairs_path.is_file():
                poses = read_pose_pairs(pose_pairs_path)
                fixed_pose = None
            else:
                poses = {}
                fixed_pose = read_fixed_source_to_target(scene / "pose.txt")
            source_rgb = _indexed_files(scene / "endoscope2" / "L", ".png")
            source_depth = _indexed_files(scene / "endoscope2" / "depthL", ".npy")
            target_rgb = _indexed_files(scene / "endoscope1" / "L", ".png")
            target_depth = _indexed_files(scene / "endoscope1" / "depthL", ".npy")
            target_tools = _indexed_files(scene / "endoscope1" / "toolL", ".png")
            identifiers = set(source_rgb) & set(source_depth)
            if poses:
                identifiers &= set(poses)
            if self.require_target:
                identifiers &= set(target_rgb)
            if not identifiers:
                raise RuntimeError(f"no aligned frames in {scene}")
            if self.require_target:
                expected = set(source_rgb) & set(source_depth)
                if poses:
                    expected &= set(poses)
                missing_targets = sorted(expected.difference(target_rgb))
                if missing_targets:
                    raise RuntimeError(
                        f"missing target frames in {scene}: {missing_targets[:8]}"
                    )

            first_path = source_rgb[min(identifiers)]
            with Image.open(first_path) as image:
                source_hw = (image.height, image.width)
            if target_rgb:
                with Image.open(target_rgb[min(target_rgb)]) as image:
                    target_hw = (image.height, image.width)
            else:
                target_hw = source_hw
            source_k = scale_intrinsics(
                intrinsics["K2_L"], source_hw, (self.height, self.width)
            )
            target_k = scale_intrinsics(
                intrinsics["K1_L"], target_hw, (self.height, self.width)
            )
            metadata_path = scene / "metadata.json"
            self.scene_metadata[scene.name] = (
                json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.is_file()
                else {}
            )
            for identifier in sorted(identifiers):
                self.samples.append(
                    StereoSample(
                        scene=scene.name,
                        frame_id=identifier,
                        source_rgb=source_rgb[identifier],
                        source_depth=source_depth[identifier],
                        target_rgb=target_rgb.get(identifier, source_rgb[identifier]),
                        target_depth=target_depth.get(identifier),
                        target_tool_mask=target_tools.get(identifier),
                        source_intrinsics=source_k,
                        target_intrinsics=target_k,
                        source_to_target=(
                            poses[identifier]
                            if fixed_pose is None
                            else fixed_pose
                        ),
                    )
                )
        if not self.samples:
            raise RuntimeError(f"no samples found below {self.root}")
        self.scenes = sorted({sample.scene for sample in self.samples})

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (self.width, self.height), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _load_depth(self, path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        array = np.squeeze(np.load(path, allow_pickle=False)).astype(np.float32)
        if array.ndim != 2:
            raise ValueError(f"depth must be 2-D: {path} -> {array.shape}")
        valid = np.isfinite(array) & (array > 0)
        clean = np.where(valid, array, 0.0)
        depth = torch.from_numpy(clean).unsqueeze(0).unsqueeze(0)
        mask = torch.from_numpy(valid.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        if tuple(array.shape) != (self.height, self.width):
            depth = F.interpolate(depth, (self.height, self.width), mode="nearest")
            mask = F.interpolate(mask, (self.height, self.width), mode="nearest")
        return depth[0], mask[0] > 0.5

    def _load_tool_mask(self, path: Path | None) -> torch.Tensor:
        if path is None:
            return torch.zeros((1, self.height, self.width), dtype=torch.bool)
        with Image.open(path) as image:
            image = image.convert("L").resize(
                (self.width, self.height), Image.Resampling.NEAREST
            )
            array = np.asarray(image, dtype=np.uint8).copy()
        return torch.from_numpy(array > 127).unsqueeze(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        sample = self.samples[index]
        source_depth, source_valid = self._load_depth(sample.source_depth)
        if sample.target_depth is not None:
            target_depth, target_depth_valid = self._load_depth(sample.target_depth)
        else:
            target_depth = torch.zeros_like(source_depth)
            target_depth_valid = torch.zeros_like(source_valid)
        return {
            "source_rgb": self._load_rgb(sample.source_rgb),
            "source_depth": source_depth,
            "source_depth_valid": source_valid,
            "target_rgb": self._load_rgb(sample.target_rgb),
            "target_depth": target_depth,
            "target_depth_valid": target_depth_valid,
            "target_tool_mask": self._load_tool_mask(sample.target_tool_mask),
            "source_intrinsics": torch.from_numpy(sample.source_intrinsics.copy()),
            "target_intrinsics": torch.from_numpy(sample.target_intrinsics.copy()),
            "source_to_target": torch.from_numpy(sample.source_to_target.copy()),
            "scene": sample.scene,
            "frame_id": sample.frame_id,
            "source_path": str(sample.source_rgb),
            "target_path": str(sample.target_rgb),
        }
