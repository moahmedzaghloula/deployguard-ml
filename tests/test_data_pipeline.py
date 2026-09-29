from pathlib import Path

import pandas as pd

from deployguard.generate_data import (
    EXPECTED_COLUMNS,
    N_ROWS,
    SEED,
    generate_dataset,
    write_dataset,
)
from deployguard.validate_data import (
    RANGE_RULES,
    sha256_file,
    validate_dataframe,
)


def test_generation_is_deterministic() -> None:
    first = generate_dataset(seed=SEED)
    second = generate_dataset(seed=SEED)

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_exact_row_count() -> None:
    dataframe = generate_dataset()

    assert len(dataframe) == N_ROWS


def test_schema_and_column_order() -> None:
    dataframe = generate_dataset()

    assert list(dataframe.columns) == EXPECTED_COLUMNS


def test_values_are_within_allowed_ranges() -> None:
    dataframe = generate_dataset()

    for column, (minimum, maximum) in RANGE_RULES.items():
        assert (
            dataframe[column]
            .between(
                minimum,
                maximum,
                inclusive="both",
            )
            .all()
        )


def test_dataset_has_no_null_values() -> None:
    dataframe = generate_dataset()

    assert not dataframe.isna().any().any()


def test_deployment_ids_are_unique() -> None:
    dataframe = generate_dataset()

    assert dataframe["deployment_id"].is_unique


def test_target_contains_both_classes() -> None:
    dataframe = generate_dataset()

    assert set(dataframe["deployment_failed"].unique()) == {0, 1}


def test_complete_dataset_contract() -> None:
    dataframe = generate_dataset()

    summary = validate_dataframe(dataframe)

    assert summary["rows"] == N_ROWS
    assert summary["missing_values"] == 0
    assert summary["duplicate_ids"] == 0


def test_generated_csv_fingerprint_is_reproducible(
    tmp_path: Path,
) -> None:
    dataframe = generate_dataset(seed=SEED)

    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"

    write_dataset(
        dataframe,
        first_path,
    )

    write_dataset(
        generate_dataset(seed=SEED),
        second_path,
    )

    assert sha256_file(first_path) == sha256_file(second_path)
