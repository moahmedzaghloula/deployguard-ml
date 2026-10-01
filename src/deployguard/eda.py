from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path("data/raw/deployments.csv")
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


def save_target_distribution(
    dataframe: pd.DataFrame,
    output_dir: Path,
) -> None:
    counts = dataframe["deployment_failed"].value_counts().sort_index()

    figure, axis = plt.subplots(figsize=(7, 5))
    counts.plot(
        kind="bar",
        ax=axis,
    )

    axis.set_title("Deployment Outcome Distribution")
    axis.set_xlabel("deployment_failed")
    axis.set_ylabel("Deployments")
    axis.set_xticklabels(
        ["Success (0)", "Failure (1)"],
        rotation=0,
    )

    figure.tight_layout()
    figure.savefig(
        output_dir / "target_distribution.png",
        dpi=150,
    )
    plt.close(figure)


def save_feature_distributions(
    dataframe: pd.DataFrame,
    output_dir: Path,
) -> None:
    for column in FEATURE_COLUMNS:
        figure, axis = plt.subplots(figsize=(7, 5))

        dataframe[column].plot(
            kind="hist",
            bins=25,
            ax=axis,
        )

        axis.set_title(f"Distribution of {column}")
        axis.set_xlabel(column)
        axis.set_ylabel("Frequency")

        figure.tight_layout()
        figure.savefig(
            output_dir / f"feature_distribution_{column}.png",
            dpi=150,
        )
        plt.close(figure)


def save_correlation_heatmap(
    dataframe: pd.DataFrame,
    output_dir: Path,
) -> pd.Series:
    correlation = dataframe[FEATURE_COLUMNS + ["deployment_failed"]].corr()

    figure, axis = plt.subplots(figsize=(12, 9))

    image = axis.imshow(
        correlation,
        vmin=-1.0,
        vmax=1.0,
        cmap="coolwarm",
    )

    axis.set_xticks(range(len(correlation.columns)))
    axis.set_yticks(range(len(correlation.index)))

    axis.set_xticklabels(
        correlation.columns,
        rotation=70,
        ha="right",
    )
    axis.set_yticklabels(
        correlation.index,
    )

    for row_index in range(len(correlation.index)):
        for column_index in range(len(correlation.columns)):
            axis.text(
                column_index,
                row_index,
                f"{correlation.iloc[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )

    axis.set_title("Feature Correlation Heatmap")

    figure.colorbar(
        image,
        ax=axis,
        label="Pearson correlation",
    )

    figure.tight_layout()
    figure.savefig(
        output_dir / "correlation_heatmap.png",
        dpi=150,
    )
    plt.close(figure)

    return correlation["deployment_failed"].drop("deployment_failed").sort_values()


def failure_rate_table(
    dataframe: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    return dataframe.groupby(
        group_column,
        observed=False,
    )["deployment_failed"].agg(
        failure_rate="mean",
        deployments="size",
    )


def save_grouped_failure_rate(
    table: pd.DataFrame,
    title: str,
    filename: str,
    output_dir: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))

    table["failure_rate"].plot(
        kind="bar",
        ax=axis,
    )

    axis.set_title(title)
    axis.set_xlabel("")
    axis.set_ylabel("Failure rate")
    axis.tick_params(
        axis="x",
        rotation=30,
    )

    figure.tight_layout()
    figure.savefig(
        output_dir / filename,
        dpi=150,
    )
    plt.close(figure)


def build_grouped_analyses(
    dataframe: pd.DataFrame,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    analysis_frame = dataframe.copy()

    analysis_frame["coverage_band"] = pd.cut(
        analysis_frame["test_coverage_percent"],
        bins=[29.99, 60, 70, 80, 90, 100],
        labels=[
            "30-60",
            "60-70",
            "70-80",
            "80-90",
            "90-100",
        ],
        include_lowest=True,
    )

    analysis_frame["failed_tests_band"] = pd.cut(
        analysis_frame["failed_tests"],
        bins=[-0.1, 0.5, 1.5, 2.5, 4.5, 40],
        labels=[
            "0",
            "1",
            "2",
            "3-4",
            "5+",
        ],
        include_lowest=True,
    )

    analysis_frame["previous_failures_band"] = pd.cut(
        analysis_frame["previous_deployment_failures"],
        bins=[-0.1, 0.5, 1.5, 2.5, 10],
        labels=[
            "0",
            "1",
            "2",
            "3+",
        ],
        include_lowest=True,
    )

    analysis_frame["experience_band"] = pd.cut(
        analysis_frame["team_experience_months"],
        bins=[0, 12, 24, 48, 72, 120],
        labels=[
            "1-12",
            "13-24",
            "25-48",
            "49-72",
            "73-120",
        ],
        include_lowest=True,
    )

    analysis_frame["change_size_band"] = pd.cut(
        analysis_frame["files_changed"],
        bins=[0, 15, 20, 25, 30, 150],
        labels=[
            "1-15",
            "16-20",
            "21-25",
            "26-30",
            "31+",
        ],
        include_lowest=True,
    )

    analysis_frame["off_hours"] = (
        (analysis_frame["deployment_hour"] < 8)
        | (analysis_frame["deployment_hour"] >= 19)
    ).astype(int)

    tables = {
        "coverage": failure_rate_table(
            analysis_frame,
            "coverage_band",
        ),
        "failed_tests": failure_rate_table(
            analysis_frame,
            "failed_tests_band",
        ),
        "previous_failures": failure_rate_table(
            analysis_frame,
            "previous_failures_band",
        ),
        "experience": failure_rate_table(
            analysis_frame,
            "experience_band",
        ),
        "change_size": failure_rate_table(
            analysis_frame,
            "change_size_band",
        ),
        "weekend": failure_rate_table(
            analysis_frame,
            "is_weekend",
        ),
        "off_hours": failure_rate_table(
            analysis_frame,
            "off_hours",
        ),
    }

    plot_config = {
        "coverage": (
            "Failure Rate by Test Coverage",
            "failure_rate_by_coverage.png",
        ),
        "failed_tests": (
            "Failure Rate by Failed Tests",
            "failure_rate_by_failed_tests.png",
        ),
        "previous_failures": (
            "Failure Rate by Previous Failures",
            "failure_rate_by_previous_failures.png",
        ),
        "experience": (
            "Failure Rate by Team Experience",
            "failure_rate_by_experience.png",
        ),
        "change_size": (
            "Failure Rate by Change Size",
            "failure_rate_by_change_size.png",
        ),
        "weekend": (
            "Failure Rate by Weekend Flag",
            "failure_rate_by_weekend.png",
        ),
        "off_hours": (
            "Failure Rate by Deployment Period",
            "failure_rate_by_off_hours.png",
        ),
    }

    for name, table in tables.items():
        title, filename = plot_config[name]

        save_grouped_failure_rate(
            table,
            title,
            filename,
            output_dir,
        )

    return tables


def main() -> None:
    plt.switch_backend("Agg")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset does not exist: {DATA_PATH}")

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.read_csv(DATA_PATH)

    save_target_distribution(
        dataframe,
        FIGURE_DIR,
    )

    save_feature_distributions(
        dataframe,
        FIGURE_DIR,
    )

    correlations = save_correlation_heatmap(
        dataframe,
        FIGURE_DIR,
    )

    grouped_tables = build_grouped_analyses(
        dataframe,
        FIGURE_DIR,
    )

    target_counts = dataframe["deployment_failed"].value_counts().sort_index()

    print("\nTarget distribution:")
    print(target_counts.to_string())

    print("\nFailure rate:")
    print(f"{dataframe['deployment_failed'].mean():.2%}")

    print("\nCorrelation with deployment_failed:")
    print(correlations.to_string())

    for name, table in grouped_tables.items():
        printable_table = table.copy()
        printable_table["failure_rate"] = printable_table["failure_rate"] * 100

        print(f"\n{name}:")
        print(printable_table.round(2).to_string())

    print(f"\nEDA figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
