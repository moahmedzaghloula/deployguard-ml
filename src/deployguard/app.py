import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest

from deployguard.train import FEATURE_COLUMNS
from deployguard.validate_data import RANGE_RULES

DEFAULT_MODEL_PATH = Path("artifacts/deployguard_model.joblib")
DEFAULT_METADATA_PATH = Path("artifacts/metadata.json")

FLOAT_FEATURES = {"test_coverage_percent"}
INTEGER_FEATURES = set(FEATURE_COLUMNS) - FLOAT_FEATURES


def error_response(
    code: str,
    message: str,
    status_code: int,
    details: dict[str, Any] | None = None,
):
    payload: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
        }
    }

    if details:
        payload["error"]["details"] = details

    return jsonify(payload), status_code


def load_runtime(
    model_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    if not model_path.is_file() or not metadata_path.is_file():
        return {
            "ready": False,
            "model": None,
            "metadata": None,
            "load_error": "artifact_missing",
        }

    try:
        with metadata_path.open(encoding="utf-8") as file_handle:
            metadata = json.load(file_handle)

        feature_contract = metadata["feature_contract"]
        metadata_features = [feature["name"] for feature in feature_contract]

        if metadata_features != FEATURE_COLUMNS:
            raise ValueError(
                "Model metadata feature contract does not match runtime contract."
            )

        model = joblib.load(model_path)

        model_features = list(
            getattr(
                model,
                "feature_names_in_",
                [],
            )
        )

        if model_features and model_features != FEATURE_COLUMNS:
            raise ValueError(
                "Persisted model feature contract does not match runtime contract."
            )

        decision_threshold = float(metadata["decision_threshold"])

        if not 0.0 <= decision_threshold <= 1.0:
            raise ValueError(
                "Decision threshold is outside the valid probability range."
            )

        return {
            "ready": True,
            "model": model,
            "metadata": metadata,
            "load_error": None,
        }

    except Exception as error:
        return {
            "ready": False,
            "model": None,
            "metadata": None,
            "load_error": type(error).__name__,
        }


def validate_prediction_payload(
    payload: dict[str, Any],
):
    received_fields = set(payload)
    required_fields = set(FEATURE_COLUMNS)

    missing_fields = sorted(required_fields - received_fields)

    unexpected_fields = sorted(received_fields - required_fields)

    if missing_fields:
        return None, error_response(
            code="missing_fields",
            message=("Request body is missing required fields."),
            status_code=422,
            details={
                "fields": missing_fields,
            },
        )

    if unexpected_fields:
        return None, error_response(
            code="unexpected_fields",
            message=("Request body contains unexpected fields."),
            status_code=422,
            details={
                "fields": unexpected_fields,
            },
        )

    normalized: dict[str, int | float] = {}
    field_errors: dict[str, str] = {}

    for feature_name in FEATURE_COLUMNS:
        value = payload[feature_name]

        if isinstance(value, bool):
            field_errors[feature_name] = "Boolean values are not accepted."
            continue

        if feature_name in INTEGER_FEATURES:
            if not isinstance(value, int):
                field_errors[feature_name] = "An integer value is required."
                continue

            normalized_value: int | float = value

        else:
            if not isinstance(value, (int, float)):
                field_errors[feature_name] = "A numeric value is required."
                continue

            normalized_value = float(value)

            if not math.isfinite(normalized_value):
                field_errors[feature_name] = "A finite numeric value is required."
                continue

        minimum, maximum = RANGE_RULES[feature_name]

        if not minimum <= normalized_value <= maximum:
            field_errors[feature_name] = (
                f"Value must be between {minimum} and {maximum}."
            )
            continue

        normalized[feature_name] = normalized_value

    if field_errors:
        return None, error_response(
            code="validation_error",
            message=("One or more feature values are invalid."),
            status_code=422,
            details={
                "fields": field_errors,
            },
        )

    return normalized, None


def public_metadata(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "trained_at_utc": metadata["trained_at_utc"],
        "feature_contract": metadata["feature_contract"],
        "target_name": metadata["target_name"],
        "dataset_sha256": metadata["dataset_sha256"],
        "decision_threshold": metadata["decision_threshold"],
        "evaluation_metrics": metadata["evaluation_metrics"],
    }


def risk_level(
    probability: float,
    threshold: float,
) -> str:
    if probability >= threshold:
        return "high"

    if probability >= threshold / 2:
        return "medium"

    return "low"


def create_app(
    config: dict[str, Any] | None = None,
) -> Flask:
    app = Flask(__name__)

    app.config.from_mapping(
        MODEL_PATH=str(DEFAULT_MODEL_PATH),
        METADATA_PATH=str(DEFAULT_METADATA_PATH),
    )

    if config:
        app.config.update(config)

    runtime = load_runtime(
        Path(app.config["MODEL_PATH"]),
        Path(app.config["METADATA_PATH"]),
    )

    app.extensions["deployguard"] = runtime

    @app.get("/")
    def index():
        current_runtime = app.extensions["deployguard"]

        metadata = current_runtime["metadata"] if current_runtime["ready"] else None

        return jsonify(
            {
                "service": "DeployGuard ML",
                "model_version": (metadata["model_version"] if metadata else None),
                "endpoints": {
                    "health": "GET /health",
                    "ready": "GET /ready",
                    "metadata": "GET /metadata",
                    "predict": "POST /predict",
                },
                "example_request": {
                    "files_changed": 32,
                    "lines_added": 850,
                    "lines_deleted": 180,
                    "test_coverage_percent": 71.5,
                    "failed_tests": 2,
                    "previous_deployment_failures": 1,
                    "deployment_hour": 22,
                    "is_weekend": 1,
                    "team_experience_months": 18,
                },
            }
        )

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
            }
        )

    @app.get("/ready")
    def ready():
        current_runtime = app.extensions["deployguard"]

        if not current_runtime["ready"]:
            return (
                jsonify(
                    {
                        "status": "not_ready",
                    }
                ),
                503,
            )

        return jsonify(
            {
                "status": "ready",
                "model_version": (current_runtime["metadata"]["model_version"]),
            }
        )

    @app.get("/metadata")
    def metadata():
        current_runtime = app.extensions["deployguard"]

        if not current_runtime["ready"]:
            return error_response(
                code="model_unavailable",
                message=("The model is not ready."),
                status_code=503,
            )

        return jsonify(public_metadata(current_runtime["metadata"]))

    @app.post("/predict")
    def predict():
        current_runtime = app.extensions["deployguard"]

        if not current_runtime["ready"]:
            return error_response(
                code="model_unavailable",
                message=("The model is not ready."),
                status_code=503,
            )

        if not request.is_json:
            return error_response(
                code="unsupported_media_type",
                message=("Content-Type must be application/json."),
                status_code=415,
            )

        try:
            payload = request.get_json()
        except BadRequest:
            return error_response(
                code="invalid_json",
                message=("Request body contains malformed JSON."),
                status_code=400,
            )

        if not isinstance(payload, dict):
            return error_response(
                code="invalid_body",
                message=("JSON request body must be an object."),
                status_code=400,
            )

        normalized, validation_error = validate_prediction_payload(payload)

        if validation_error is not None:
            return validation_error

        input_frame = pd.DataFrame(
            [normalized],
            columns=FEATURE_COLUMNS,
        )

        try:
            probability = float(
                current_runtime["model"].predict_proba(input_frame)[0, 1]
            )

            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError("Invalid model probability.")

        except Exception:
            app.logger.exception("Model inference failed.")

            return error_response(
                code="inference_error",
                message=("The prediction could not be completed."),
                status_code=500,
            )

        metadata_payload = current_runtime["metadata"]

        threshold = float(metadata_payload["decision_threshold"])

        prediction = int(probability >= threshold)

        return jsonify(
            {
                "risk_level": risk_level(
                    probability,
                    threshold,
                ),
                "failure_probability": round(
                    probability,
                    6,
                ),
                "prediction": prediction,
                "decision_threshold": threshold,
                "model_version": (metadata_payload["model_version"]),
            }
        )

    @app.errorhandler(404)
    def not_found(_error):
        return error_response(
            code="not_found",
            message="Endpoint not found.",
            status_code=404,
        )

    @app.errorhandler(405)
    def method_not_allowed(_error):
        return error_response(
            code="method_not_allowed",
            message=("HTTP method is not allowed for this endpoint."),
            status_code=405,
        )

    return app
