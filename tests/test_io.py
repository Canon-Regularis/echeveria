"""Image loading and decoding: the single decode path shared by files and uploads."""

from __future__ import annotations

import io

import numpy as np
import pytest

from phytovision.exceptions import InvalidImageError
from phytovision.io import decode_rgb_bytes, load_image


def test_load_image_reads_an_rgb_array(tmp_path) -> None:
    from PIL import Image as PILImage

    path = tmp_path / "plant.png"
    PILImage.fromarray(np.zeros((6, 8, 3), dtype=np.uint8)).save(path)
    out = load_image(path)
    assert out.shape == (6, 8, 3)
    assert out.dtype == np.uint8


def test_load_image_missing_file_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nope.png")


def test_decode_rejects_non_image_bytes() -> None:
    with pytest.raises(InvalidImageError):
        decode_rgb_bytes(b"not an image")


def test_decode_honours_exif_orientation() -> None:
    from PIL import Image as PILImage

    # A portrait photo is stored landscape with an orientation tag; decoding must apply it so the
    # pixels match the photo as viewed, or geometry.orientation depends on how the camera stored it.
    arr = np.zeros((8, 16, 3), dtype=np.uint8)  # stored landscape (H, W) = (8, 16)
    arr[:, :4] = 255
    image = PILImage.fromarray(arr)
    exif = image.getexif()
    exif[0x0112] = 6  # orientation 6: rotate for upright portrait display
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)

    out = decode_rgb_bytes(buffer.getvalue())
    assert out.shape[:2] == (16, 8)  # transposed to the display orientation, not (8, 16)
