"""Flask entry point for the Sightline object-detection web app."""

from __future__ import annotations

import base64
import io
import os
import time
from typing import Any

from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError

from detector import DetectionError, YoloDetector


MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_SIDE = 1920
MAX_SOURCE_PIXELS = 40_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}


def _read_image(upload: Any) -> Image.Image:
    """Validate, orient, and bound an uploaded image before inference."""
    # Read one byte beyond the limit so an oversized file is easy to detect.
    raw = upload.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise ValueError("The selected image is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("The image is larger than the 12 MB limit.")

    try:
        image = Image.open(io.BytesIO(raw))
        if image.width * image.height > MAX_SOURCE_PIXELS:
            raise ValueError("The image dimensions are too large. Use a photo under 40 megapixels.")
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("That file is not a supported or readable image.") from exc

    if image.format not in ALLOWED_FORMATS:
        raise ValueError("Use a JPG, PNG, WEBP, or BMP image.")

    # Correct phone-camera rotation and give YOLO one predictable color format.
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max(image.size) > MAX_IMAGE_SIDE:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    return image


def _parse_confidence(value: str | None) -> float:
    try:
        confidence = float(value or 0.35)
    except (TypeError, ValueError) as exc:
        raise ValueError("Confidence must be a number from 0.10 to 0.90.") from exc
    if not 0.10 <= confidence <= 0.90:
        raise ValueError("Confidence must be between 0.10 and 0.90.")
    return confidence


def create_app(detector: YoloDetector | None = None) -> Flask:
    """Create the web app; tests may inject a tiny fake detector."""

    app = Flask(__name__)
    # Leave room for multipart field overhead while enforcing the exact file limit above.
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + (256 * 1024)
    app.config["DETECTOR"] = detector or YoloDetector()

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        engine = app.config["DETECTOR"]
        return jsonify({"status": "ready", "model": engine.model_name, "model_state": engine.state})

    @app.post("/api/detect")
    def detect():
        # The browser sends multipart/form-data with image and confidence fields.
        upload = request.files.get("image")
        if upload is None:
            return jsonify({"error": "Add a photo or capture a camera frame first."}), 400

        try:
            confidence = _parse_confidence(request.form.get("confidence"))
            image = _read_image(upload)
            started = time.perf_counter()
            result = app.config["DETECTOR"].detect(image, confidence)
            elapsed_ms = round((time.perf_counter() - started) * 1000)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except DetectionError as exc:
            return jsonify({"error": str(exc)}), 503

        # A data URL lets the browser display the returned JPEG without a second
        # file request or saving the user's photo on disk.
        encoded = base64.b64encode(result.image_bytes).decode("ascii")
        return jsonify(
            {
                "image": f"data:image/jpeg;base64,{encoded}",
                "detections": result.detections,
                "count": len(result.detections),
                "elapsed_ms": elapsed_ms,
                "width": image.width,
                "height": image.height,
                "model": app.config["DETECTOR"].model_name,
                "confidence": confidence,
            }
        )

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"error": "The image is larger than the 12 MB limit."}), 413

    @app.errorhandler(500)
    def server_error(_error):
        return jsonify({"error": "Detection failed unexpectedly. Check the server log and try again."}), 500

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=os.getenv("FLASK_DEBUG") == "1")
