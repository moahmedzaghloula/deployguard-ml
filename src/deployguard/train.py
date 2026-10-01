import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from deployguard.validate_data import (
    RANGE_RULES,
    sha256_file,
    validate_dataframe,
)

DATA_PATH = Path("data/raw/deployments.csv")
MODEL_PATH = Path("artifacts/deployguard_model.joblib")
METADATA_PATH = Path("artifacts/metadata.json")
METRICS_PATH = Path("reports/metrics.json")
FIGURE_DIR = Path("reports/figures")

FEATURE_COLUMNS = [
    "files_changed",
    "lines_added",
    "lines_deleted",
    "test_coverage_percent",
    "failed_tests",
    "previous_deployment_failures",
    "deployment_hour",
    "is_weekend",
    "team_experience_months",
]

TARGET_COLUMN = "deployment_failed"
IDENTIFIER_COLUMN = "deployment_id"

RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
DECISION_THRESHOLD = 0.50
PRIMARY_METRIC = "f1"
MODEL_VERSION = "1.0.0"

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def load_validated_dataset() -> tuple[pd.DataFrame, str]:
    """Load the dataset only after validating its Stage 2 contract."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset does not exist: {DATA_PATH}")

    dataframe = pd.read_csv(DATA_PATH)
    validate_dataframe(dataframe)

    return dataframe, sha256_file(DATA_PATH)


def split_features_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the prediction target."""
    features = dataframe[FEATURE_COLUMNS].copy()
    target = dataframe[TARGET_COLUMN].copy()

    return features, target


def split_train_test(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Create one reproducible stratified holdout test set."""
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=target,
    )


def build_candidate_models() -> dict[str, Pipeline]:
    """Build the two candidate model pipelines."""
    logistic_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2_000,
                    random_state=RANDOM_SEED,
                    solver="liblinear",
                ),
            ),
        ]
    )

    random_forest_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                "passthrough",
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=10,
                    class_weight="balanced_subsample",
                    max_features="sqrt",
                    n_jobs=1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    return {
        "LogisticRegression": logistic_pipeline,
        "RandomForestClassifier": random_forest_pipeline,
    }


def compare_candidates(
    candidates: dict[str, Pipeline],
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, dict[str, float]]:
    """Compare candidates using stratified cross-validation."""
    cross_validator = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    comparison: dict[str, dict[str, float]] = {}

    for model_name, pipeline in candidates.items():
        scores = cross_validate(
            pipeline,
            features,
            target,
            cv=cross_validator,
            scoring=SCORING,
            n_jobs=1,
            return_train_score=False,
        )

        model_results: dict[str, float] = {}

        for metric_name in SCORING:
            metric_values = scores[f"test_{metric_name}"]

            model_results[f"{metric_name}_mean"] = float(metric_values.mean())
            model_results[f"{metric_name}_std"] = float(metric_values.std())

        comparison[model_name] = model_results

    return comparison


def select_model(
    comparison: dict[str, dict[str, float]],
) -> str:
    """Select a candidate using validation F1 only."""
    selection_key = f"{PRIMARY_METRIC}_mean"

    return max(
        comparison,
        key=lambda model_name: comparison[model_name][selection_key],
    )


def evaluate_model(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[dict[str, float | int], pd.Series]:
    """Evaluate the selected model at the fixed decision threshold."""
    probabilities = pd.Series(
        model.predict_proba(features)[:, 1],
        index=features.index,
        name="failure_probability",
    )

    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)

    matrix = confusion_matrix(
        target,
        predictions,
        labels=[0, 1],
    )

    true_negative, false_positive, false_negative, true_positive = matrix.ravel()

    false_positive_rate = false_positive / (false_positive + true_negative)

    false_negative_rate = false_negative / (false_negative + true_positive)

    metrics: dict[str, float | int] = {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(
            precision_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(recall_score(target, predictions)),
        "f1": float(f1_score(target, predictions)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
    }

    return metrics, probabilities


def build_feature_contract() -> list[dict[str, Any]]:
    """Create machine-readable metadata for model inputs."""
    contract = []

    for feature_name in FEATURE_COLUMNS:
        minimum, maximum = RANGE_RULES[feature_name]

        feature_type = "float" if feature_name == "test_coverage_percent" else "integer"

        contract.append(
            {
                "name": feature_name,
                "type": feature_type,
                "minimum": minimum,
                "maximum": maximum,
            }
        )

    return contract


def build_metadata(
    model_name: str,
    dataset_sha256: str,
    evaluation_metrics: dict[str, float | int],
    comparison: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Build model lineage and reproducibility metadata."""
    return {
        "model_name": model_name,
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_contract": build_feature_contract(),
        "target_name": TARGET_COLUMN,
        "identifier_excluded": IDENTIFIER_COLUMN,
        "random_seed": RANDOM_SEED,
        "dataset_sha256": dataset_sha256,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "decision_threshold": DECISION_THRESHOLD,
        "evaluation_metrics": evaluation_metrics,
        "selection": {
            "primary_metric": PRIMARY_METRIC,
            "cross_validation_folds": CV_FOLDS,
            "test_size": TEST_SIZE,
            "candidate_results": comparison,
        },
    }


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write deterministic, human-readable JSON formatting."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file_handle:
        json.dump(
            payload,
            file_handle,
            indent=2,
            sort_keys=True,
        )

        file_handle.write("\n")


def save_evaluation_figures(
    target: pd.Series,
    probabilities: pd.Series,
) -> None:
    """Save confusion-matrix and ROC figures without a GUI."""
    plt.switch_backend("Agg")

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)

    matrix = confusion_matrix(
        target,
        predictions,
        labels=[0, 1],
    )

    figure, axis = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Success (0)",
            "Failure (1)",
        ],
    ).plot(
        ax=axis,
        values_format="d",
        colorbar=False,
    )

    axis.set_title("DeployGuard ML Confusion Matrix")

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "confusion_matrix.png",
        dpi=150,
    )

    plt.close(figure)

    false_positive_rate, true_positive_rate, _ = roc_curve(
        target,
        probabilities,
    )

    figure, axis = plt.subplots(figsize=(7, 5))

    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label="Selected model",
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random baseline",
    )

    axis.set_title("DeployGuard ML ROC Curve")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "roc_curve.png",
        dpi=150,
    )

    plt.close(figure)


def print_comparison(
    comparison: dict[str, dict[str, float]],
) -> None:
    """Print a compact candidate comparison table."""
    table = pd.DataFrame.from_dict(
        comparison,
        orient="index",
    )

    columns = [
        "accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "roc_auc_mean",
    ]

    print("\nCross-validation comparison:")
    print(table[columns].round(4).to_string())


def train_and_save() -> None:
    """Run candidate selection, final fitting, and test evaluation."""
    dataframe, dataset_sha256 = load_validated_dataset()

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

    candidates = build_candidate_models()

    comparison = compare_candidates(
        candidates,
        train_features,
        train_target,
    )

    selected_model_name = select_model(comparison)

    selected_model = clone(candidates[selected_model_name])

    selected_model.fit(
        train_features,
        train_target,
    )

    evaluation_metrics, probabilities = evaluate_model(
        selected_model,
        test_features,
        test_target,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        selected_model,
        MODEL_PATH,
        compress=3,
    )

    metadata = build_metadata(
        model_name=selected_model_name,
        dataset_sha256=dataset_sha256,
        evaluation_metrics=evaluation_metrics,
        comparison=comparison,
    )

    metrics_report = {
        "selected_model": selected_model_name,
        "selection_metric": PRIMARY_METRIC,
        "cross_validation": comparison,
        "test_metrics": evaluation_metrics,
    }

    write_json(
        METADATA_PATH,
        metadata,
    )

    write_json(
        METRICS_PATH,
        metrics_report,
    )

    save_evaluation_figures(
        test_target,
        probabilities,
    )

    print(f"Dataset SHA-256: {dataset_sha256}")
    print(f"Training rows: {len(train_features):,}")
    print(f"Test rows: {len(test_features):,}")

    print_comparison(comparison)

    print(f"\nSelected model: {selected_model_name}")
    print(f"Primary selection metric: {PRIMARY_METRIC}")

    print("\nHeld-out test metrics:")

    for metric_name, metric_value in evaluation_metrics.items():
        if isinstance(metric_value, float):
            print(f"{metric_name}: " f"{metric_value:.4f}")
        else:
            print(f"{metric_name}: " f"{metric_value}")

    print(f"\nModel artifact: {MODEL_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Metrics report: {METRICS_PATH}")


def show_saved_evaluation() -> None:
    """Print the already-recorded final evaluation."""
    if not METRICS_PATH.exists():
        raise FileNotFoundError("Metrics report is missing. Run training first.")

    with METRICS_PATH.open(encoding="utf-8") as file_handle:
        metrics = json.load(file_handle)

    print(
        json.dumps(
            metrics,
            indent=2,
            sort_keys=True,
        )
    )


def check_model_contract() -> None:
    """Validate local model artifacts and their dataset lineage."""
    required_paths = [
        MODEL_PATH,
        METADATA_PATH,
        METRICS_PATH,
        FIGURE_DIR / "confusion_matrix.png",
        FIGURE_DIR / "roc_curve.png",
    ]

    missing_paths = [str(path) for path in required_paths if not path.exists()]

    if missing_paths:
        raise FileNotFoundError("Missing model artifacts: " + ", ".join(missing_paths))

    dataframe, current_dataset_sha256 = load_validated_dataset()

    with METADATA_PATH.open(encoding="utf-8") as file_handle:
        metadata = json.load(file_handle)

    if metadata["dataset_sha256"] != (current_dataset_sha256):
        raise ValueError(
            "Model metadata does not match " "the current dataset SHA-256."
        )

    metadata_features = [feature["name"] for feature in metadata["feature_contract"]]

    if metadata_features != FEATURE_COLUMNS:
        raise ValueError("Model feature contract is invalid.")

    if metadata["decision_threshold"] != (DECISION_THRESHOLD):
        raise ValueError("Decision threshold metadata is invalid.")

    model = joblib.load(MODEL_PATH)

    model_features = list(
        getattr(
            model,
            "feature_names_in_",
            [],
        )
    )

    if model_features != FEATURE_COLUMNS:
        raise ValueError(
            "Persisted pipeline feature names " "do not match the contract."
        )

    sample = dataframe[FEATURE_COLUMNS].head(3)

    probabilities = model.predict_proba(sample)[:, 1]

    if probabilities.shape != (3,):
        raise ValueError("Persisted model returned " "an invalid probability shape.")

    if not ((probabilities >= 0.0).all() and (probabilities <= 1.0).all()):
        raise ValueError("Persisted model returned " "invalid probabilities.")

    print("Model contract: PASS")
    print(f"Dataset SHA-256: " f"{current_dataset_sha256}")
    print(f"Model: {metadata['model_name']}")
    print(f"Model version: " f"{metadata['model_version']}")
    print(f"Decision threshold: " f"{metadata['decision_threshold']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Train and validate the DeployGuard ML model.")
    )

    parser.add_argument(
        "action",
        choices=[
            "train",
            "evaluate",
            "check",
        ],
    )

    arguments = parser.parse_args()

    if arguments.action == "train":
        train_and_save()
    elif arguments.action == "evaluate":
        show_saved_evaluation()
    else:
        check_model_contract()


if __name__ == "__main__":
    main()
