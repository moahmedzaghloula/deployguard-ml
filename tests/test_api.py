from pathlib import Path

import joblib
import pytest

from deployguard.app import create_app
from deployguard.generate_data import (
    SEED,
    generate_dataset,
)
from deployguard.train import (
    build_candidate_models,
    build_metadata,
    evaluate_model,
    split_features_target,
    write_json,
)


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    temporary_directory = tmp_path_factory.mktemp("deployguard-api")

    model_path = temporary_directory / "model.joblib"

    metadata_path = temporary_directory / "metadata.json"

    dataframe = generate_dataset(
        n_rows=800,
        seed=SEED,
    )

    features, target = split_features_target(dataframe)

    model = build_candidate_models()["LogisticRegression"]

    model.fit(
        features,
        target,
    )

    metrics, _ = evaluate_model(
        model,
        features,
        target,
    )

    joblib.dump(
        model,
        model_path,
    )

    metadata = build_metadata(
        model_name=("LogisticRegression"),
        dataset_sha256="a" * 64,
        evaluation_metrics=metrics,
        comparison={"LogisticRegression": {"f1_mean": float(metrics["f1"])}},
    )

    write_json(
        metadata_path,
        metadata,
    )

    return create_app(
        {
            "TESTING": True,
            "MODEL_PATH": str(model_path),
            "METADATA_PATH": str(metadata_path),
        }
    )


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def valid_payload():
    return {
        "files_changed": 32,
        "lines_added": 850,
        "lines_deleted": 180,
        "test_coverage_percent": 71.5,
        "failed_tests": 2,
        "previous_deployment_failures": 1,
        "deployment_hour": 22,
        "is_weekend": 1,
        "team_experience_months": 18,
    }


def test_root_service_discovery(
    client,
) -> None:
    response = client.get("/")

    assert response.status_code == 200

    payload = response.get_json()

    assert payload["service"] == "DeployGuard ML"

    assert "example_request" in payload
    assert "endpoints" in payload


def test_health(
    client,
) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_ready(
    client,
) -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.get_json()["status"] == "ready"


def test_metadata_is_safe(
    client,
) -> None:
    response = client.get("/metadata")

    assert response.status_code == 200

    payload = response.get_json()

    assert "model_name" in payload
    assert "model_version" in payload
    assert "feature_contract" in payload
    assert "evaluation_metrics" in payload

    assert "python_version" not in payload
    assert "scikit_learn_version" not in payload
    assert "MODEL_PATH" not in payload
    assert "METADATA_PATH" not in payload


def test_valid_prediction(
    client,
    valid_payload,
) -> None:
    response = client.post(
        "/predict",
        json=valid_payload,
    )

    assert response.status_code == 200

    payload = response.get_json()

    assert payload["risk_level"] in {
        "low",
        "medium",
        "high",
    }

    assert 0.0 <= payload["failure_probability"] <= 1.0

    assert payload["prediction"] in {0, 1}

    assert payload["decision_threshold"] == 0.5


def test_predict_rejects_wrong_content_type(
    client,
) -> None:
    response = client.post(
        "/predict",
        data="not-json",
        content_type="text/plain",
    )

    assert response.status_code == 415
    assert response.get_json()["error"]["code"] == ("unsupported_media_type")


def test_predict_rejects_malformed_json(
    client,
) -> None:
    response = client.post(
        "/predict",
        data='{"files_changed":',
        content_type="application/json",
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_json"


def test_predict_requires_json_object(
    client,
) -> None:
    response = client.post(
        "/predict",
        json=[
            1,
            2,
            3,
        ],
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_body"


def test_predict_rejects_missing_fields(
    client,
    valid_payload,
) -> None:
    payload = valid_payload.copy()
    payload.pop("failed_tests")

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "missing_fields"


def test_predict_rejects_unexpected_fields(
    client,
    valid_payload,
) -> None:
    payload = valid_payload.copy()
    payload["rollback_status"] = 0

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "unexpected_fields"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("files_changed", True),
        ("files_changed", 4.5),
        ("lines_added", "850"),
        (
            "test_coverage_percent",
            "71.5",
        ),
        ("is_weekend", False),
    ],
)
def test_predict_rejects_invalid_types(
    client,
    valid_payload,
    field,
    value,
) -> None:
    payload = valid_payload.copy()
    payload[field] = value

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "validation_error"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("files_changed", 0),
        ("lines_added", 5001),
        (
            "test_coverage_percent",
            101,
        ),
        ("deployment_hour", 24),
        ("is_weekend", 2),
        (
            "team_experience_months",
            121,
        ),
    ],
)
def test_predict_rejects_out_of_range_values(
    client,
    valid_payload,
    field,
    value,
) -> None:
    payload = valid_payload.copy()
    payload[field] = value

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "validation_error"


def test_not_found_returns_json(
    client,
) -> None:
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.is_json


def test_wrong_method_returns_json(
    client,
) -> None:
    response = client.get("/predict")

    assert response.status_code == 405
    assert response.is_json


def test_health_stays_up_when_model_is_missing(
    tmp_path: Path,
) -> None:
    unavailable_app = create_app(
        {
            "TESTING": True,
            "MODEL_PATH": str(tmp_path / "missing.joblib"),
            "METADATA_PATH": str(tmp_path / "missing.json"),
        }
    )

    client = unavailable_app.test_client()

    assert client.get("/health").status_code == 200

    assert client.get("/ready").status_code == 503

    assert (
        client.post(
            "/predict",
            json={},
        ).status_code
        == 503
    )
