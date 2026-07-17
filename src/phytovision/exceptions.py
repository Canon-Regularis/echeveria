"""Package exception hierarchy.

All library-raised errors derive from :class:`PhytoVisionError`, so callers can catch "a phytovision
failure" without also swallowing unrelated bugs. Where a built-in type is the natural super-class
(``ValueError`` for bad input, ``RuntimeError`` for bad state), the error multiply-inherits it
so existing ``except ValueError`` / ``except RuntimeError`` code keeps working.
"""

from __future__ import annotations


class PhytoVisionError(Exception):
    """Base class for every error raised by phytovision."""


class InvalidImageError(PhytoVisionError, ValueError):
    """Input is not a valid RGB image (wrong type, shape, or empty)."""


class ContractViolationError(PhytoVisionError, ValueError):
    """A pipeline component violated an interface invariant (e.g. an empty region set)."""


class SegmentationError(PhytoVisionError):
    """Segmentation could not produce a usable foreground."""


class ModelNotFittedError(PhytoVisionError, RuntimeError):
    """A trainable model was used for inference before ``fit`` was called."""


class ModelSchemaError(PhytoVisionError, ValueError):
    """The live feature schema does not match the schema a model was trained on."""


class ConfigError(PhytoVisionError, ValueError):
    """A pipeline configuration referenced an unknown component or bad parameters."""


class InsufficientDataError(PhytoVisionError, ValueError):
    """Not enough data to compute a result, e.g. a survival cohort with no repeated observations."""
