"""The optional post-model head seam."""

from __future__ import annotations

from phytovision.pipeline import Pipeline
from phytovision.types import PlantFeatures


class _RegionCountHead:
    name = "region_count"

    def run(self, features: PlantFeatures) -> object:
        return features.region_count


def test_head_runs_and_is_recorded(healthy_image) -> None:
    report = Pipeline.default().add_head(_RegionCountHead()).analyze(healthy_image)
    assert report.head_outputs["region_count"] == 1
    assert "region_count" in report.summary()["heads"]


def test_no_heads_by_default(healthy_image) -> None:
    report = Pipeline.default().analyze(healthy_image)
    assert report.head_outputs == {}
