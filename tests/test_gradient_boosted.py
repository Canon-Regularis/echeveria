"""The trained model is a drop-in for the heuristic (same StressModel contract) and explainable.

Requires the ``ml`` extra; skipped automatically if scikit-learn is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from phytovision.exceptions import (
    ConfigError,
    ModelNotFittedError,
    ModelSchemaError,
)
from phytovision.models.base import ContributionModel, StressModel
from phytovision.models.stress.gradient_boosted import GradientBoostedStressModel
from phytovision.types import PlantFeatures

_KEYS = ["colour.gcc_mean", "colour.yellow_fraction", "texture.entropy"]


def _training_data(seed: int = 0, per_class: int = 60):
    """Noisy, separable synthetic samples so the gradient-boosted model can actually learn."""
    rng = np.random.default_rng(seed)
    dicts: list[dict[str, float]] = []
    labels: list[int] = []
    for _ in range(per_class):  # healthy: green, no yellowing, low texture entropy
        dicts.append(
            {
                "colour.gcc_mean": float(rng.normal(0.40, 0.02)),
                "colour.yellow_fraction": float(abs(rng.normal(0.03, 0.02))),
                "texture.entropy": float(rng.normal(2.5, 0.3)),
            }
        )
        labels.append(0)
    for _ in range(per_class):  # stressed: less green, yellowing, high texture entropy
        dicts.append(
            {
                "colour.gcc_mean": float(rng.normal(0.30, 0.02)),
                "colour.yellow_fraction": float(abs(rng.normal(0.40, 0.05))),
                "texture.entropy": float(rng.normal(4.5, 0.3)),
            }
        )
        labels.append(1)
    return dicts, labels


def test_gbm_is_a_substitutable_stress_model() -> None:
    dicts, labels = _training_data()
    model = GradientBoostedStressModel(feature_keys=_KEYS).fit(dicts, labels)

    assert isinstance(model, StressModel)
    assert isinstance(model, ContributionModel)

    stressed = PlantFeatures(
        values={"colour.gcc_mean": 0.30, "colour.yellow_fraction": 0.45, "texture.entropy": 4.6},
        region_count=1,
    )
    healthy = PlantFeatures(
        values={"colour.gcc_mean": 0.41, "colour.yellow_fraction": 0.01, "texture.entropy": 2.4},
        region_count=1,
    )

    assert model.predict(stressed).score > model.predict(healthy).score
    for assessment in (model.predict(stressed), model.predict(healthy)):
        assert 0.0 <= assessment.score <= 1.0
        assert 0.0 <= assessment.confidence <= 1.0


def test_shap_completeness_holds_when_positive_label_is_class_zero() -> None:
    pytest.importorskip("shap")
    dicts, labels = _training_data()
    model = GradientBoostedStressModel(feature_keys=_KEYS, positive_label=0).fit(dicts, labels)
    probe = PlantFeatures(
        values={"colour.gcc_mean": 0.34, "colour.yellow_fraction": 0.2, "texture.entropy": 3.5},
        region_count=1,
    )
    result = model.shap_attribution(probe)
    # Completeness: base + sum(values) == model output, kept consistent with the SHAP orientation
    # even when the positive label is class 0 (the output is oriented to match the values and base).
    assert abs(result.base_value + sum(result.values.values()) - result.model_output) < 1e-4


def test_label_uses_the_shared_bucket_cuts_not_a_binary_split() -> None:
    from phytovision.models.base import bucket_label

    dicts, labels = _training_data()
    model = GradientBoostedStressModel(feature_keys=_KEYS).fit(dicts, labels)
    # Every verdict must agree with bucket_label at its score, so this model does not drift from the
    # heuristic/ensemble (which would happen with a hardcoded 0.5 cut that never says "mild").
    for gcc in (0.28, 0.32, 0.36, 0.40):
        features = PlantFeatures(
            values={"colour.gcc_mean": gcc, "colour.yellow_fraction": 0.2, "texture.entropy": 3.5},
            region_count=1,
        )
        assessment = model.predict(features)
        assert assessment.label == bucket_label(assessment.score)


def test_contributions_skip_a_missing_feature() -> None:
    dicts, labels = _training_data()
    model = GradientBoostedStressModel(feature_keys=_KEYS).fit(dicts, labels)
    partial = PlantFeatures(
        values={"colour.gcc_mean": 0.35, "texture.entropy": 3.0}, region_count=1
    )  # colour.yellow_fraction is absent (schema drift)
    contributions = model.contributions(partial)
    assert "colour.yellow_fraction" not in contributions  # the absent feature earns no attribution
    assert all(np.isfinite(value) for value in contributions.values())  # and no NaN leaks


def test_shap_completeness_holds_for_a_flipped_positive_label() -> None:
    pytest.importorskip("shap")
    dicts, labels = _training_data()
    model = GradientBoostedStressModel(feature_keys=_KEYS, positive_label=0).fit(
        dicts, [1 - y for y in labels]
    )
    features = PlantFeatures(
        values={"colour.gcc_mean": 0.30, "colour.yellow_fraction": 0.4, "texture.entropy": 4.0},
        region_count=1,
    )
    result = model.shap_attribution(features)
    total = result.base_value + sum(result.values.values())
    # Completeness must hold for positive_label=0 too: the 2-D single-margin SHAP path is now
    # oriented to the positive label rather than left in class-1 terms.
    assert abs(total - result.model_output) < 1e-6


def test_seeded_fit_is_reproducible() -> None:
    dicts, labels = _training_data()
    probe = PlantFeatures(
        values={"colour.gcc_mean": 0.34, "colour.yellow_fraction": 0.2, "texture.entropy": 3.5},
        region_count=1,
    )
    first = GradientBoostedStressModel(feature_keys=_KEYS, random_state=7).fit(dicts, labels)
    second = GradientBoostedStressModel(feature_keys=_KEYS, random_state=7).fit(dicts, labels)
    assert first.predict(probe).score == second.predict(probe).score


def test_fit_rejects_an_all_missing_feature_with_a_clear_error() -> None:
    # A feature that is NaN for every sample crashes the histogram learner cryptically; it must be a
    # clean ConfigError naming the feature (real for library callers and cross-dataset CV).
    dicts = [{"colour.gcc_mean": 0.4}, {"colour.gcc_mean": 0.3}, {"colour.gcc_mean": 0.4}]
    labels = [0, 1, 0]
    model = GradientBoostedStressModel(feature_keys=["colour.gcc_mean", "colour.yellow_fraction"])
    with pytest.raises(ConfigError, match="colour.yellow_fraction"):
        model.fit(dicts, labels)


def test_fit_trains_when_a_feature_is_only_partially_missing() -> None:
    # A feature missing for SOME samples is fine (the learner handles per-sample NaN); only an
    # entirely-missing column is rejected.
    dicts, labels = _training_data(per_class=20)
    for row in dicts[::3]:  # drop one feature from a third of the rows
        row.pop("texture.entropy", None)
    model = GradientBoostedStressModel(feature_keys=_KEYS).fit(dicts, labels)
    assert 0.0 <= model.predict(PlantFeatures(values=dicts[0], region_count=1)).score <= 1.0


def test_fit_rejects_empty_training_data() -> None:
    with pytest.raises(ConfigError):
        GradientBoostedStressModel(feature_keys=_KEYS).fit([], [])


def test_gbm_contributions_cover_all_features() -> None:
    dicts, labels = _training_data()
    model = GradientBoostedStressModel(feature_keys=_KEYS).fit(dicts, labels)
    features = PlantFeatures(
        values={"colour.gcc_mean": 0.30, "colour.yellow_fraction": 0.45, "texture.entropy": 4.6},
        region_count=1,
    )
    contributions = model.contributions(features)
    assert set(contributions) == set(_KEYS)


def test_gbm_predict_before_fit_raises() -> None:
    model = GradientBoostedStressModel(feature_keys=_KEYS)
    features = PlantFeatures(values={k: 0.0 for k in _KEYS}, region_count=1)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(features)


def test_gbm_fit_rejects_positive_label_absent_from_classes() -> None:
    dicts, _ = _training_data()
    # Relabel to classes {10, 20}; positive_label=1 is then absent, which would otherwise make
    # predict() silently read an arbitrary class probability.
    labels = [10] * (len(dicts) // 2) + [20] * (len(dicts) - len(dicts) // 2)
    with pytest.raises(ConfigError, match="positive_label"):
        GradientBoostedStressModel(feature_keys=_KEYS).fit(dicts, labels)


def test_gbm_save_load_roundtrip(tmp_path) -> None:
    dicts, labels = _training_data()
    model = GradientBoostedStressModel(feature_keys=_KEYS).fit(dicts, labels)
    features = PlantFeatures(
        values={"colour.gcc_mean": 0.30, "colour.yellow_fraction": 0.45, "texture.entropy": 4.6},
        region_count=1,
    )
    before = model.predict(features).score

    path = tmp_path / "model.joblib"
    model.save(path)
    reloaded = GradientBoostedStressModel.load(path)

    assert reloaded.feature_keys == model.feature_keys
    assert reloaded.predict(features).score == before


def test_gbm_save_before_fit_raises(tmp_path) -> None:
    with pytest.raises(ModelNotFittedError):
        GradientBoostedStressModel(feature_keys=_KEYS).save(tmp_path / "model.joblib")


def _fitted():
    dicts, labels = _training_data()
    return GradientBoostedStressModel(feature_keys=_KEYS).fit(dicts, labels)


def test_gbm_strict_schema_raises_on_drift() -> None:
    model = _fitted()
    model.strict_schema = True
    partial = PlantFeatures(values={"colour.gcc_mean": 0.3}, region_count=1)  # missing 2 of 3 keys
    with pytest.raises(ModelSchemaError, match="missing"):
        model.predict(partial)


def test_gbm_tolerant_schema_warns_once_and_predicts(caplog) -> None:
    import logging

    model = _fitted()  # strict_schema defaults False
    partial = PlantFeatures(values={"colour.gcc_mean": 0.3}, region_count=1)
    with caplog.at_level(logging.WARNING, logger="phytovision.models.stress.gradient_boosted"):
        first = model.predict(partial)
        model.predict(partial)  # a second drifting predict must not warn again
    assert 0.0 <= first.score <= 1.0  # still produces a (NaN-tolerant) prediction
    warnings = [r for r in caplog.records if "schema mismatch" in r.getMessage()]
    assert len(warnings) == 1


def test_gbm_full_schema_does_not_warn(caplog) -> None:
    import logging

    model = _fitted()
    full = PlantFeatures(values=dict.fromkeys(_KEYS, 0.3), region_count=1)
    with caplog.at_level(logging.WARNING, logger="phytovision.models.stress.gradient_boosted"):
        model.predict(full)
    assert not any("schema mismatch" in r.getMessage() for r in caplog.records)


def test_gbm_strict_schema_survives_round_trip(tmp_path) -> None:
    model = _fitted()
    model.strict_schema = True
    path = tmp_path / "strict.joblib"
    model.save(path)
    assert GradientBoostedStressModel.load(path).strict_schema is True
