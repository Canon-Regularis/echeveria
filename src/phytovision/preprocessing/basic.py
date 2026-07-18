"""A straightforward resize + normalize preprocessor (the v1 default)."""

from __future__ import annotations

import numpy as np
from skimage.transform import resize

from phytovision.exceptions import ConfigError
from phytovision.preprocessing.base import Preprocessor
from phytovision.types import Image
from phytovision.validation import validate_rgb_image


class ResizeNormalizePreprocessor(Preprocessor):
    """Resize so the longest side is at most ``max_size`` and scale to float32 ``[0, 1]``.

    Resizing keeps compute bounded and, because it is applied consistently, leaves *relative*
    phenotypic traits (ratios, colour, texture) comparable across images.
    """

    def __init__(self, max_size: int = 1024) -> None:
        if max_size < 16:
            raise ConfigError("max_size must be at least 16")
        self.max_size = max_size

    def process(self, image: Image) -> Image:
        validate_rgb_image(image)

        img = image.astype(np.float32)
        if np.issubdtype(image.dtype, np.integer):  # scale by the dtype's range, not a hard 255
            img = img / float(np.iinfo(image.dtype).max)
        elif img.max() > 1.0:  # a float image outside [0, 1]: treat it as an 8-bit range
            img = img / 255.0

        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > self.max_size:
            scale = self.max_size / longest
            new_hw = (max(1, round(h * scale)), max(1, round(w * scale)))
            img = resize(img, new_hw, order=1, anti_aliasing=True, preserve_range=True)

        return np.clip(img, 0.0, 1.0).astype(np.float32)
