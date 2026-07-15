"""Image loading helpers. Kept separate so stages depend on arrays, not on file formats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image as _PILImage
from PIL import UnidentifiedImageError
from PIL.Image import DecompressionBombError

from phytovision.exceptions import InvalidImageError
from phytovision.types import Image


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
            return np.asarray(im.convert("RGB"))
    except (UnidentifiedImageError, DecompressionBombError, OSError) as exc:
        # DecompressionBombError is a bare Exception, not an OSError, so it must be listed.
        raise InvalidImageError(f"could not decode image: {p} ({exc})") from exc
