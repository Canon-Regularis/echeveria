"""A tiny generic registry so new implementations plug in by name (Open/Closed principle).

New segmenters, extractors, region providers or models register themselves; callers look them up by
name. Nothing in the orchestrator needs editing to add an implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from phytovision.exceptions import ConfigError

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator: register a factory (usually the class itself) under ``name``."""

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._factories:
                raise ConfigError(f"{self._kind} {name!r} is already registered")
            self._factories[name] = factory
            return factory

        return decorator

    def create(self, name: str, /, **kwargs: object) -> T:
        try:
            factory = self._factories[name]
        except KeyError:
            raise KeyError(
                f"unknown {self._kind} {name!r}; available: {sorted(self._factories)}"
            ) from None
        return factory(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        return name in self._factories
