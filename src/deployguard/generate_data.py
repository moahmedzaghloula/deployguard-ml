from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 5_000
RAW_DATA_PATH = Path("data/raw/deployments.csv")

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


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Convert risk scores into probabilities between zero and one."""
    safe_values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-safe_values))


def generate_dataset(
    n_rows: int = N_ROWS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate deterministic synthetic pre-deployment records."""
    rng = np.random.default_rng(seed)

    files_changed = np.clip(
        rng.poisson(lam=20, size=n_rows) + 1,
        1,
        150,
    ).astype(int)

    lines_added = np.clip(
        np.rint(
            files_changed
            * rng.gamma(
                shape=2.0,
                scale=12.0,
                size=n_rows,
            )
        ),
        1,
        5_000,
    ).astype(int)

    lines_deleted = np.clip(
        np.rint(
            files_changed
            * rng.gamma(
                shape=1.6,
                scale=8.0,
                size=n_rows,
            )
        ),
        0,
        4_000,
    ).astype(int)

    test_coverage_percent = np.clip(
        rng.normal(
            loc=78.0,
            scale=12.0,
            size=n_rows,
        ),
        30.0,
        100.0,
    )
    test_coverage_percent = np.round(test_coverage_percent, 2)

    failed_test_rate = (
        0.25
        + files_changed / 45.0
        + np.maximum(
            0.0,
            70.0 - test_coverage_percent,
        )
        / 20.0
    )

    failed_tests = np.clip(
        rng.poisson(
            lam=failed_test_rate,
            size=n_rows,
        ),
        0,
        40,
    ).astype(int)

    previous_deployment_failures = np.clip(
        rng.poisson(
            lam=0.8,
            size=n_rows,
        ),
        0,
        10,
    ).astype(int)

    deployment_hour = rng.integers(
        low=0,
        high=24,
        size=n_rows,
        dtype=np.int64,
    )

    is_weekend = rng.binomial(
        n=1,
        p=0.22,
        size=n_rows,
    ).astype(int)

    team_experience_months = np.clip(
        np.rint(
            rng.gamma(
                shape=3.0,
                scale=12.0,
                size=n_rows,
            )
        ),
        1,
        120,
    ).astype(int)

    off_hours = ((deployment_hour < 8) | (deployment_hour >= 19)).astype(int)

    large_change_low_coverage = (
        (files_changed >= 30) & (test_coverage_percent < 70.0)
    ).astype(int)

    risk_score = (
        -2.50
        + 0.018 * (files_changed - 20)
        + 0.0006 * (lines_added - 500)
        + 0.0008 * (lines_deleted - 250)
        - 0.045 * (test_coverage_percent - 75.0)
        + 0.38 * failed_tests
        + 0.42 * previous_deployment_failures
        + 0.50 * is_weekend
        + 0.25 * off_hours
        - 0.012 * (team_experience_months - 24)
        + 0.60 * large_change_low_coverage
    )

    failure_probability = sigmoid(risk_score)

    deployment_failed = rng.binomial(
        n=1,
        p=failure_probability,
        size=n_rows,
    ).astype(int)

    return pd.DataFrame(
        {
            "deployment_id": [f"dep-{index:05d}" for index in range(1, n_rows + 1)],
            "files_changed": files_changed,
            "lines_added": lines_added,
            "lines_deleted": lines_deleted,
            "test_coverage_percent": test_coverage_percent,
            "failed_tests": failed_tests,
            "previous_deployment_failures": (previous_deployment_failures),
            "deployment_hour": deployment_hour,
            "is_weekend": is_weekend,
            "team_experience_months": team_experience_months,
            "deployment_failed": deployment_failed,
        },
        columns=EXPECTED_COLUMNS,
    )


def write_dataset(
    dataframe: pd.DataFrame,
    output_path: Path = RAW_DATA_PATH,
) -> Path:
    """Write the dataset using stable CSV settings."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
        lineterminator="\n",
        float_format="%.2f",
    )

    return output_path


def main() -> None:
    dataframe = generate_dataset()
    output_path = write_dataset(dataframe)

    failure_rate = dataframe["deployment_failed"].mean()

    print(f"Dataset path: {output_path}")
    print(f"Rows: {len(dataframe):,}")
    print(f"Random seed: {SEED}")
    print(f"Failure rate: {failure_rate:.2%}")


if __name__ == "__main__":
    main()
