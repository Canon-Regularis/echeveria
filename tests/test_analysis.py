"""The shared dataset-analysis helper (F1) and its table export."""

from __future__ import annotations

import logging

from phytovision.analysis import AnalysisRow, analyze_dataset, feature_table
from phytovision.datasets.directory import ImageDirectoryLoader
from phytovision.datasets.folder import FolderClassificationLoader
from phytovision.pipeline import Pipeline


def test_image_directory_loader_labels_from_subfolder(dataset_dir) -> None:
    loader = ImageDirectoryLoader(dataset_dir)
    assert len(loader) == 2
    assert {s.label for s in loader} == {"healthy", "wilted"}


def test_analyze_dataset_one_row_per_image(dataset_dir) -> None:
    rows = list(analyze_dataset(Pipeline.default(), FolderClassificationLoader(dataset_dir)))
    assert len(rows) == 2
    for row in rows:
        assert isinstance(row, AnalysisRow)
        assert 0.0 <= row.score <= 1.0
        assert row.label in {"healthy", "wilted"}
        assert "colour.gcc_mean" in row.features


def test_analyze_dataset_skips_unreadable_images(tmp_path, caplog) -> None:
    (tmp_path / "bad.png").write_bytes(b"not an image")
    with caplog.at_level(logging.WARNING):
        rows = list(analyze_dataset(Pipeline.default(), ImageDirectoryLoader(tmp_path)))
    assert rows == []
    assert "skipping" in caplog.text


def test_feature_table_header_and_records(dataset_dir) -> None:
    rows = analyze_dataset(Pipeline.default(), FolderClassificationLoader(dataset_dir))
    fieldnames, records = feature_table(rows)
    assert fieldnames[:4] == ["image_path", "label", "split", "source"]
    assert "target" in fieldnames  # the measured-value column is part of the base schema
    assert "colour.gcc_mean" in fieldnames
    assert len(records) == 2
    assert set(records[0]) <= set(fieldnames)


def test_analysis_row_record_carries_the_target() -> None:
    row = AnalysisRow(
        image_path="x",
        label="healthy",
        split=None,
        source=None,
        score=0.2,
        confidence=0.5,
        stress_label="healthy",
        model="m",
        features={"colour.gcc_mean": 0.4},
        target=0.7,
    )
    assert row.as_record()["target"] == 0.7
