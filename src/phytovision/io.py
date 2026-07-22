"""Image loading helpers. Kept separate so stages depend on arrays, not on file formats."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image as _PILImage
from PIL import ImageOps, UnidentifiedImageError
from PIL.Image import DecompressionBombError

from phytovision.exceptions import InvalidImageError
from phytovision.types import Image

# DecompressionBombError is a bare Exception, not an OSError, so it is listed explicitly.
_DECODE_ERRORS = (UnidentifiedImageError, DecompressionBombError, OSError)


def load_image(path: str | Path) -> Image:
    """Load an image from disk as an ``H x W x 3`` uint8 RGB array.

    :raises FileNotFoundError: if ``path`` does not exist.
    :raises InvalidImageError: if the file is not a decodable image.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {p}")
    try:
        with _PILImage.open(p) as im:
            # Honour the EXIF orientation tag so a portrait phone photo is analysed upright, not
            # sideways: geometry.orientation and every orientation-aware feature otherwise depend on
            # how the camera happened to store the pixels rather than on the plant as viewed.
            return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    except _DECODE_ERRORS as exc:
        raise InvalidImageError(f"could not decode image: {p} ({exc})") from exc


def decode_rgb_bytes(data: bytes) -> Image:
    """Decode raw image bytes as an ``H x W x 3`` uint8 RGB array.

    The single decode path for uploaded or in-memory images (API and dashboard).

    :raises InvalidImageError: if the bytes are not a decodable image.
    """
    try:
        with _PILImage.open(io.BytesIO(data)) as im:
            # Apply the EXIF orientation before conversion, matching load_image, so an uploaded
            # portrait photo is decoded upright rather than rotated.
            return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    except _DECODE_ERRORS as exc:
        raise InvalidImageError(f"invalid image: {exc}") from exc
