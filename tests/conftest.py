"""Shared fixtures: synthetic plant images and a stand-in leaf segmenter for the LSP tests."""

from __future__ import annotations

import numpy as np
import pytest

from phytovision.regions.base import region_from_mask
from phytovision.segmentation.leaves.instance import LeafInstanceSegmenter
from phytovision.types import Image, Mask, Region


def _blob_image(
    color: tuple[float, float, float],
    bg: tuple[float, float, float] = (0.10, 0.10, 0.10),
    size: int = 128,
    radius: int = 42,
    noise: float = 0.02,
    seed: int = 0,
) -> Image:
    """An image with a central elliptical 'plant' of ``color`` on a dark background."""
    rng = np.random.default_rng(seed)
    img = np.ones((size, size, 3), dtype=np.float32) * np.array(bg, dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    disk = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= radius**2
    img[disk] = np.array(color, dtype=np.float32)
    img = img + rng.normal(0.0, noise, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


@pytest.fixture
def healthy_image() -> Image:
    """A turgid green plant."""
    return _blob_image((0.15, 0.60, 0.15), seed=1)


@pytest.fixture
def stressed_image() -> Image:
    """A yellowing / browning plant."""
    img = _blob_image((0.62, 0.50, 0.12), seed=2)
    # a few brown necrotic patches
    for cy, cx in ((54, 54), (74, 70), (60, 80)):
        img[cy - 6 : cy + 6, cx - 6 : cx + 6] = np.array([0.40, 0.26, 0.10], dtype=np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


@pytest.fixture
def plant_mask() -> Mask:
    size, radius = 128, 42
    yy, xx = np.mgrid[0:size, 0:size]
    return (yy - size / 2) ** 2 + (xx - size / 2) ** 2 <= radius**2


@pytest.fixture
def plant_region(plant_mask: Mask) -> Region:
    return region_from_mask(region_id=0, label="plant", mask=plant_mask)


class HalfSplitLeafSegmenter(LeafInstanceSegmenter):
    """A trivial leaf segmenter that splits the plant into left/right halves.

    Not biologically meaningful — it exists purely to exercise the multi-region code path so the LSP
    substitution test can run without a trained model.
    """

    def segment_leaves(self, image: Image, plant_mask: Mask) -> list[Mask]:
        width = plant_mask.shape[1]
        left = plant_mask.copy()
        left[:, width // 2 :] = False
        right = plant_mask.copy()
        right[:, : width // 2] = False
        return [m for m in (left, right) if m.any()]


@pytest.fixture
def leaf_segmenter() -> HalfSplitLeafSegmenter:
    return HalfSplitLeafSegmenter()


@pytest.fixture
def image_path(tmp_path, healthy_image):
    """A healthy plant image written to disk (for path-based / CLI tests)."""
    from PIL import Image as PILImage

    path = tmp_path / "plant.png"
    PILImage.fromarray((healthy_image * 255).astype(np.uint8)).save(path)
    return path


@pytest.fixture
def dataset_dir(tmp_path, healthy_image, stressed_image):
    """A ``root/<label>/<image>`` folder with one healthy and one wilted plant, for dataset tests.

    Returns the root path; it holds 2 images across 2 label subfolders.
    """
    from PIL import Image as PILImage

    root = tmp_path / "dataset"
    for label, img in (("healthy", healthy_image), ("wilted", stressed_image)):
        class_dir = root / label
        class_dir.mkdir(parents=True)
        PILImage.fromarray((img * 255).astype(np.uint8)).save(class_dir / f"{label}.png")
    return root


@pytest.fixture
def training_dir(tmp_path):
    """A labelled folder with several images per class, big enough to fit a small model."""
    from PIL import Image as PILImage

    root = tmp_path / "train"
    for label, color, base_seed in (
        ("healthy", (0.15, 0.60, 0.15), 0),
        ("wilted", (0.62, 0.50, 0.12), 100),
    ):
        class_dir = root / label
        class_dir.mkdir(parents=True)
        for i in range(6):
            img = _blob_image(color, seed=base_seed + i, noise=0.03)
            PILImage.fromarray((img * 255).astype(np.uint8)).save(class_dir / f"{i}.png")
    return root
