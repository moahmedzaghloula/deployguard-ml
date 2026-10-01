import hashlib
from pathlib import Path

import pandas as pd
from pandas.api.types import (
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

DATA_PATH = Path("data/raw/deployments.csv")
EXPECTED_ROW_COUNT = 5_000

EXPECTED_COLUMNS = [
    "deployment_id",
    "files_changed",
    "lines_added",
    "lines_deleted",
    "test_coverage_percent",
    "failed_tests",
    "previous_deployment_failures",
    "deployment_hour",
    "is_weekend",
    "team_experience_months",
    "deployment_failed",
]

INTEGER_COLUMNS = [
    "files_changed",
    "lines_added",
    "lines_deleted",
    "failed_tests",
    "previous_deployment_failures",
    "deployment_hour",
    "is_weekend",
    "team_experience_months",
    "deployment_failed",
]

RANGE_RULES = {
    "files_changed": (1, 150),
    "lines_added": (1, 5_000),
    "lines_deleted": (0, 4_000),
    "test_coverage_percent": (30.0, 100.0),
    "failed_tests": (0, 40),
    "previous_deployment_failures": (0, 10),
    "deployment_hour": (0, 23),
    "is_weekend": (0, 1),
    "team_experience_months": (1, 120),
    "deployment_failed": (0, 1),
}

MIN_FAILURE_RATE = 0.10
MAX_FAILURE_RATE = 0.40


class DataValidationError(ValueError):
    """Raised when the deployment dataset violates its contract."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 fingerprint of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for chunk in iter(
            lambda: file_handle.read(8192),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def validate_dataframe(dataframe: pd.DataFrame) -> dict[str, float | int]:
    """Validate the complete deployment dataset contract."""
    errors: list[str] = []

    if list(dataframe.columns) != EXPECTED_COLUMNS:
        errors.append("Columns or column order do not match the expected schema.")

    if len(dataframe) != EXPECTED_ROW_COUNT:
        errors.append(
            f"Expected {EXPECTED_ROW_COUNT} rows, " f"found {len(dataframe)}."
        )

    if not (
        is_object_dtype(dataframe["deployment_id"])
        or is_string_dtype(dataframe["deployment_id"])
    ):
        errors.append("deployment_id must be a string-like column.")

    for column in INTEGER_COLUMNS:
        if not is_integer_dtype(dataframe[column]):
            errors.append(f"{column} must use an integer data type.")

    if not is_numeric_dtype(dataframe["test_coverage_percent"]):
        errors.append("test_coverage_percent must be numeric.")

    missing_value_count = int(dataframe.isna().sum().sum())

    if missing_value_count != 0:
        errors.append(f"Dataset contains {missing_value_count} missing values.")

    duplicate_id_count = int(dataframe["deployment_id"].duplicated().sum())

    if duplicate_id_count != 0:
        errors.append(
            f"Dataset contains {duplicate_id_count} duplicate deployment IDs."
        )

    valid_id_format = dataframe["deployment_id"].str.fullmatch(r"dep-\d{5}")

    if not valid_id_format.all():
        errors.append("One or more deployment IDs use an invalid format.")

    for column, (minimum, maximum) in RANGE_RULES.items():
        invalid_mask = ~dataframe[column].between(
            minimum,
            maximum,
            inclusive="both",
        )

        if invalid_mask.any():
            errors.append(
                f"{column} contains values outside " f"[{minimum}, {maximum}]."
            )

    target_values = set(dataframe["deployment_failed"].unique().tolist())

    if not target_values.issubset({0, 1}):
        errors.append("deployment_failed contains values other than 0 and 1.")

    if target_values != {0, 1}:
        errors.append("deployment_failed must contain both target classes.")

    failure_rate = float(dataframe["deployment_failed"].mean())

    if not MIN_FAILURE_RATE <= failure_rate <= MAX_FAILURE_RATE:
        errors.append(
            "deployment_failed balance is outside the "
            f"accepted range: {failure_rate:.2%}."
        )

    if errors:
        formatted_errors = "\n- ".join(errors)
        raise DataValidationError(f"Dataset validation failed:\n- {formatted_errors}")

    return {
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "missing_values": missing_value_count,
        "duplicate_ids": duplicate_id_count,
        "failure_rate": failure_rate,
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset does not exist: {DATA_PATH}")

    dataframe = pd.read_csv(DATA_PATH)
    summary = validate_dataframe(dataframe)
    fingerprint = sha256_file(DATA_PATH)

    print("Dataset validation: PASS")
    print(f"Rows: {summary['rows']:,}")
    print(f"Columns: {summary['columns']}")
    print(f"Missing values: {summary['missing_values']}")
    print(f"Duplicate deployment IDs: " f"{summary['duplicate_ids']}")
    print(f"Failure rate: " f"{summary['failure_rate']:.2%}")
    print(f"SHA-256: {fingerprint}")


if __name__ == "__main__":
    main()
