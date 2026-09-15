"""Flask API and web entry point for the Student Perceptron Lab."""

from __future__ import annotations

import logging
import math
import os
from threading import Lock
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from perceptron import Perceptron


app = Flask(__name__)
app.config.update(
    JSON_SORT_KEYS=False,
    MAX_CONTENT_LENGTH=64 * 1024,
)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

MAX_POINTS = 50
MAX_EPOCHS = 100

# This small in-memory snapshot makes a point-only /api/predict request useful
# during local development.  The browser normally sends explicit parameters,
# which is the reliable stateless approach on serverless platforms such as
# Vercel.  The lock keeps simultaneous local requests from observing half of an
# update.
_model_lock = Lock()
_model_state: dict[str, Any] = {
    "weights": [0.0, 0.0],
    "bias": 0.0,
    "trained": False,
}


class ApiError(ValueError):
    """An expected client error with a stable machine-readable code."""

    def __init__(self, message: str, *, code: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


def _json_body() -> Any:
    """Read a JSON body or raise a consistent API error."""

    payload = request.get_json(silent=True)
    if payload is None:
        raise ApiError(
            "Send a valid JSON request body.",
            code="invalid_json",
        )
    return payload


def _finite_number(value: Any, field: str, *, minimum: float, maximum: float) -> float:
    """Validate a JSON number, rejecting booleans, NaN, infinity, and outliers."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError(f"{field} must be a number.", code="invalid_point")
    number = float(value)
    if not math.isfinite(number):
        raise ApiError(f"{field} must be finite.", code="invalid_point")
    if number < minimum or number > maximum:
        raise ApiError(
            f"{field} must be between {minimum:g} and {maximum:g}.",
            code="point_out_of_range",
        )
    return number


def _target(value: Any, index: int) -> int:
    """Validate one target without silently converting strings or booleans."""

    if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
        raise ApiError(
            f"points[{index}].target must be the integer 0 or 1.",
            code="invalid_target",
        )
    return value


def _parse_training_payload(payload: Any) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate both supported training payload shapes and build NumPy arrays."""

    if isinstance(payload, list):
        points = payload
        max_epochs = MAX_EPOCHS
    elif isinstance(payload, dict):
        points = payload.get("points")
        max_epochs = payload.get("max_epochs", MAX_EPOCHS)
    else:
        raise ApiError(
            "The training body must be a points array or an object containing points.",
            code="invalid_payload",
        )

    if not isinstance(points, list) or not points:
        raise ApiError(
            "Add at least one passing and one failing student before training.",
            code="empty_dataset",
            status=422,
        )
    if len(points) > MAX_POINTS:
        raise ApiError(
            f"Use at most {MAX_POINTS} students in one training run.",
            code="dataset_too_large",
            status=413,
        )
    if isinstance(max_epochs, bool) or not isinstance(max_epochs, int):
        raise ApiError("max_epochs must be an integer.", code="invalid_max_epochs")
    if max_epochs < 1 or max_epochs > MAX_EPOCHS:
        raise ApiError(
            f"max_epochs must be between 1 and {MAX_EPOCHS}.",
            code="invalid_max_epochs",
        )

    samples: list[list[float]] = []
    targets: list[int] = []
    labels_by_coordinate: dict[tuple[float, float], int] = {}

    for index, point in enumerate(points):
        if not isinstance(point, dict):
            raise ApiError(
                f"points[{index}] must be an object with p1, p2, and target.",
                code="invalid_point",
            )
        p1 = _finite_number(point.get("p1"), f"points[{index}].p1", minimum=0, maximum=10)
        p2 = _finite_number(point.get("p2"), f"points[{index}].p2", minimum=0, maximum=10)
        target = _target(point.get("target"), index)

        coordinate = (p1, p2)
        previous_target = labels_by_coordinate.get(coordinate)
        if previous_target is not None and previous_target != target:
            raise ApiError(
                "The same student coordinates cannot be labeled both Pass and Fail; "
                "no classifier can satisfy that contradiction.",
                code="conflicting_labels",
                status=422,
            )
        labels_by_coordinate[coordinate] = target
        samples.append([p1, p2])
        targets.append(target)

    if len(set(targets)) < 2:
        raise ApiError(
            "Training needs both classes. Add at least one passing and one failing student.",
            code="single_class_dataset",
            status=422,
        )

    return (
        np.asarray(samples, dtype=np.float64),
        np.asarray(targets, dtype=np.int64),
        max_epochs,
    )


def _boundary(weights: list[float], bias: float) -> dict[str, Any]:
    """Describe w1*p1 + w2*p2 + b = 0 without dividing by a near-zero value."""

    w1, w2 = (float(weights[0]), float(weights[1]))
    epsilon = 1e-12
    result: dict[str, Any] = {
        "w1": w1,
        "w2": w2,
        "bias": float(bias),
        "equation": "w1*p1 + w2*p2 + b = 0",
        "positive_region": "w1*p1 + w2*p2 + b >= 0",
    }

    if abs(w1) < epsilon and abs(w2) < epsilon:
        result.update({"type": "undefined", "slope": None, "intercept": None})
    elif abs(w2) < epsilon:
        result.update(
            {
                "type": "vertical",
                "p1_intercept": -float(bias) / w1,
                "slope": None,
                "intercept": None,
            }
        )
    else:
        result.update(
            {
                "type": "slope_intercept",
                "slope": -w1 / w2,
                "intercept": -float(bias) / w2,
            }
        )
    return result


def _parse_prediction_model(payload: dict[str, Any]) -> tuple[list[float], float, str, bool]:
    """Prefer explicit model parameters, with a local-memory compatibility fallback."""

    supplied_weights = payload.get("weights", payload.get("W"))
    supplied_bias = payload.get("bias", payload.get("b"))

    if supplied_weights is None and supplied_bias is None:
        with _model_lock:
            return (
                list(_model_state["weights"]),
                float(_model_state["bias"]),
                "server_memory",
                bool(_model_state["trained"]),
            )
    if supplied_weights is None or supplied_bias is None:
        raise ApiError(
            "Provide both weights and bias, or omit both to use the last local model.",
            code="incomplete_model",
        )
    if (
        not isinstance(supplied_weights, list)
        or len(supplied_weights) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in supplied_weights)
    ):
        raise ApiError("weights must be an array of two finite numbers.", code="invalid_model")

    weights = [float(value) for value in supplied_weights]
    if not all(math.isfinite(value) for value in weights):
        raise ApiError("weights must contain only finite numbers.", code="invalid_model")
    if isinstance(supplied_bias, bool) or not isinstance(supplied_bias, (int, float)):
        raise ApiError("bias must be a finite number.", code="invalid_model")
    bias = float(supplied_bias)
    if not math.isfinite(bias):
        raise ApiError("bias must be a finite number.", code="invalid_model")
    return weights, bias, "request", True


@app.get("/")
def index() -> str:
    """Render the interactive learning laboratory."""

    return render_template("index.html")


@app.get("/api/health")
def health() -> Any:
    """Provide a tiny deployment health check without exposing internal details."""

    return jsonify(ok=True, service="student-perceptron")


@app.post("/api/train")
def train() -> Any:
    """Train a fresh deterministic perceptron and return every animation state."""

    samples, targets, max_epochs = _parse_training_payload(_json_body())
    model = Perceptron(input_dim=2, learning_rate=1.0)
    result = model.fit(samples, targets, max_epochs=max_epochs)

    final_weights = result["final"]["W"]
    final_bias = result["final"]["b"]
    with _model_lock:
        _model_state.update(
            {
                "weights": list(final_weights),
                "bias": float(final_bias),
                "trained": True,
            }
        )

    class_counts = {
        "pass": int(np.sum(targets == 1)),
        "fail": int(np.sum(targets == 0)),
    }
    converged = bool(result["converged"])
    message = (
        "Perfect separation found. Every student is classified correctly."
        if converged
        else (
            f"No perfect separating line was found in {max_epochs} epochs. "
            "The data may be linearly inseparable, as in the XOR preset."
        )
    )

    return jsonify(
        ok=True,
        status="converged" if converged else "not_converged",
        message=message,
        dataset={"size": len(samples), "class_counts": class_counts},
        boundary=_boundary(final_weights, final_bias),
        **result,
    )


@app.post("/api/predict")
def predict() -> Any:
    """Classify one student with explicit or last-seen model parameters."""

    payload = _json_body()
    if not isinstance(payload, dict):
        raise ApiError("The prediction body must be a JSON object.", code="invalid_payload")

    p1 = _finite_number(payload.get("p1"), "p1", minimum=0, maximum=10)
    p2 = _finite_number(payload.get("p2"), "p2", minimum=0, maximum=10)
    weights, bias, model_source, trained = _parse_prediction_model(payload)

    model = Perceptron(input_dim=2)
    model.weights = np.asarray(weights, dtype=np.float64)
    model.bias = bias
    net = float(model.net_input([p1, p2]))
    prediction = int(model.predict([p1, p2]))

    return jsonify(
        ok=True,
        point={"p1": p1, "p2": p2},
        prediction=prediction,
        label="Pass" if prediction == 1 else "Fail",
        net_input=net,
        model={"weights": weights, "bias": bias, "source": model_source, "trained": trained},
    )


@app.after_request
def add_security_headers(response: Any) -> Any:
    """Apply conservative browser headers to API and document responses."""

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; base-uri 'self'; "
        "form-action 'self'; frame-ancestors 'none'"
    )
    return response


@app.errorhandler(ApiError)
def handle_api_error(error: ApiError) -> tuple[Any, int]:
    """Return expected validation failures in one predictable JSON envelope."""

    return jsonify(ok=False, error={"code": error.code, "message": error.message}), error.status


@app.errorhandler(413)
def handle_payload_too_large(_error: Any) -> tuple[Any, int]:
    """Explain Flask's request-size rejection as JSON."""

    return (
        jsonify(
            ok=False,
            error={
                "code": "payload_too_large",
                "message": "The request is too large. Use at most 50 student points.",
            },
        ),
        413,
    )


@app.errorhandler(Exception)
def handle_unexpected_error(error: Exception) -> tuple[Any, int] | HTTPException:
    """Preserve HTTP errors and hide implementation details for server failures."""

    if isinstance(error, HTTPException):
        return error
    logger.exception("Unhandled application error")
    return (
        jsonify(
            ok=False,
            error={
                "code": "internal_error",
                "message": "The server could not complete that request. Please try again.",
            },
        ),
        500,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
