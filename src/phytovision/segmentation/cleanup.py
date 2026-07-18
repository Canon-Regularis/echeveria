"""Shared post-processing for plant masks.

Drops small specks, fills holes, and optionally keeps the largest blob. Segmenters call this so
the cleanup logic lives in one place.
"""

from __future__ import annotations

from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, remove_small_holes, remove_small_objects

from phytovision.types import Mask


def clean_mask(
    mask: Mask,
    image_shape: tuple[int, int],
    min_object_fraction: float = 0.002,
    closing_radius: int = 2,
    keep_largest: bool = False,
) -> Mask:
    """Drop specks and holes below ``min_object_fraction`` of the image, close, and optionally keep
    only the largest connected region."""
    min_size = max(1, int(min_object_fraction * image_shape[0] * image_shape[1]))
    # scikit-image >=0.26: max_size removes objects/holes up to AND INCLUDING that size, so pass
    # min_size - 1 to drop only components strictly below the fraction and keep one exactly at it.
    threshold = max(0, min_size - 1)
    mask = remove_small_objects(mask, max_size=threshold)
    mask = remove_small_holes(mask, max_size=threshold)
    if closing_radius > 0 and mask.any():
        mask = closing(mask, disk(closing_radius))
    if keep_largest and mask.any():
        mask = largest_component(mask)
    return mask


def largest_component(mask: Mask) -> Mask:
    """The single largest connected True region of a boolean mask."""
    labelled = label(mask)
    props = regionprops(labelled)
    if not props:
        return mask
    biggest = max(props, key=lambda region: region.area)
    return labelled == biggest.label
