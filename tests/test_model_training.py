from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from deployguard.generate_data import (
    SEED,
    generate_dataset,
    write_dataset,
)
from deployguard.train import (
    DECISION_THRESHOLD,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_candidate_models,
    build_metadata,
    evaluate_model,
    split_features_target,
    split_train_test,
)
from deployguard.validate_data import sha256_file


@pytest.fixture(scope="module")
def trained_logistic_bundle():
    dataframe = generate_dataset(
        n_rows=2_500,
        seed=SEED,
    )

    features, target = split_features_target(dataframe)

    (
        train_features,
        test_features,
        train_target,
        test_target,
    ) = split_train_test(
        features,
        target,
    )

    model = build_candidate_models()["LogisticRegression"]

    model.fit(
        train_features,
        train_target,
    )

    metrics, probabilities = evaluate_model(
        model,
        test_features,
        test_target,
    )

    return {
        "model": model,
        "train_features": train_features,
        "train_target": train_target,
        "test_features": test_features,
        "test_target": test_target,
        "metrics": metrics,
        "probabilities": probabilities,
    }


def test_split_is_deterministic() -> None:
    dataframe = generate_dataset(
        n_rows=600,
        seed=SEED,
    )

    features, target = split_features_target(dataframe)

    first_split = split_train_test(
        features,
        target,
    )

    second_split = split_train_test(
        features,
        target,
    )

    assert first_split[0].index.tolist() == (second_split[0].index.tolist())

    assert first_split[1].index.tolist() == (second_split[1].index.tolist())


def test_model_training_is_deterministic() -> None:
    dataframe = generate_dataset(
        n_rows=800,
        seed=SEED,
    )

    features, target = split_features_target(dataframe)

    (
        train_features,
        test_features,
        train_target,
        _,
    ) = split_train_test(
        features,
        target,
    )

    first_model = build_candidate_models()["LogisticRegression"]

    second_model = build_candidate_models()["LogisticRegression"]

    first_model.fit(
        train_features,
        train_target,
    )

    second_model.fit(
        train_features,
        train_target,
    )

    first_probabilities = first_model.predict_proba(test_features)[:, 1]

    second_probabilities = second_model.predict_proba(test_features)[:, 1]

    np.testing.assert_allclose(
        first_probabilities,
        second_probabilities,
    )


def test_feature_contract_excludes_identifier_and_target() -> None:
    assert "deployment_id" not in FEATURE_COLUMNS
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 9


def test_pipeline_can_be_persisted_and_loaded(
    trained_logistic_bundle,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.joblib"

    model = trained_logistic_bundle["model"]

    joblib.dump(
        model,
        model_path,
    )

    loaded_model = joblib.load(model_path)

    assert isinstance(
        loaded_model,
        Pipeline,
    )

    original_predictions = model.predict(trained_logistic_bundle["test_features"])

    loaded_predictions = loaded_model.predict(trained_logistic_bundle["test_features"])

    np.testing.assert_array_equal(
        original_predictions,
        loaded_predictions,
    )


def test_prediction_shape(
    trained_logistic_bundle,
) -> None:
    probabilities = trained_logistic_bundle["probabilities"]

    test_rows = len(trained_logistic_bundle["test_features"])

    assert probabilities.shape == (test_rows,)


def test_probabilities_are_valid(
    trained_logistic_bundle,
) -> None:
    probabilities = trained_logistic_bundle["probabilities"]

    assert (probabilities >= 0.0).all()

    assert (probabilities <= 1.0).all()


def test_metadata_structure(
    trained_logistic_bundle,
) -> None:
    metadata = build_metadata(
        model_name="LogisticRegression",
        dataset_sha256="a" * 64,
        evaluation_metrics=(trained_logistic_bundle["metrics"]),
        comparison={
            "LogisticRegression": {
                "f1_mean": 0.40,
            }
        },
    )

    required_keys = {
        "model_name",
        "model_version",
        "trained_at_utc",
        "feature_contract",
        "target_name",
        "identifier_excluded",
        "random_seed",
        "dataset_sha256",
        "python_version",
        "scikit_learn_version",
        "decision_threshold",
        "evaluation_metrics",
        "selection",
    }

    assert required_keys.issubset(metadata)

    assert metadata["decision_threshold"] == DECISION_THRESHOLD


def test_metadata_connects_to_dataset_hash(
    tmp_path: Path,
    trained_logistic_bundle,
) -> None:
    dataset_path = tmp_path / "deployments.csv"

    dataframe = generate_dataset(
        n_rows=500,
        seed=SEED,
    )

    write_dataset(
        dataframe,
        dataset_path,
    )

    fingerprint = sha256_file(dataset_path)

    metadata = build_metadata(
        model_name="LogisticRegression",
        dataset_sha256=fingerprint,
        evaluation_metrics=(trained_logistic_bundle["metrics"]),
        comparison={
            "LogisticRegression": {
                "f1_mean": 0.40,
            }
        },
    )

    assert metadata["dataset_sha256"] == fingerprint


def test_model_quality_is_reasonable(
    trained_logistic_bundle,
) -> None:
    metrics = trained_logistic_bundle["metrics"]

    assert metrics["roc_auc"] >= 0.60
    assert metrics["recall"] >= 0.40
    assert metrics["f1"] >= 0.30
