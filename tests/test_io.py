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


# Real photos and uploads arrive in many pixel modes, not just RGB: a grayscale scan, a PNG with an
# alpha channel, a CMYK JPEG from a print workflow, a palette GIF, a 1-bit fax. Every one must
# decode to a plain H x W x 3 uint8 array so the pipeline sees a single, uniform contract.
@pytest.mark.parametrize(
    ("mode", "fmt"),
    [("L", "PNG"), ("LA", "PNG"), ("RGBA", "PNG"), ("P", "PNG"), ("CMYK", "JPEG"), ("1", "PNG")],
)
def test_decode_normalizes_any_pixel_mode_to_rgb(mode, fmt) -> None:
    from PIL import Image as PILImage

    buffer = io.BytesIO()
    PILImage.new("RGB", (12, 8), (60, 150, 70)).convert(mode).save(buffer, format=fmt)
    out = decode_rgb_bytes(buffer.getvalue())
    assert out.shape == (8, 12, 3)
    assert out.dtype == np.uint8


def test_decode_rejects_truncated_image_bytes() -> None:
    from PIL import Image as PILImage

    # A half-finished upload or download must fail cleanly, not decode to a garbled buffer.
    buffer = io.BytesIO()
    PILImage.new("RGB", (64, 64), (30, 120, 40)).save(buffer, format="JPEG")
    whole = buffer.getvalue()
    with pytest.raises(InvalidImageError):
        decode_rgb_bytes(whole[: len(whole) // 2])


def test_load_image_on_a_directory_is_a_clean_error(tmp_path) -> None:
    # A path that exists but is a directory must not leak a raw OSError to the caller.
    with pytest.raises(InvalidImageError):
        load_image(tmp_path)
