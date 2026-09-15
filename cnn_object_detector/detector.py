"""Run YOLO object detection and draw its answers on an image.

Kid-sized picture: YOLO looks at one whole photograph like a quick-eyed spotter.
Its already-trained CNN changes pixels into layers of clues—first edges, then
textures, parts, and object shapes. It proposes rectangles, names what may be in
each rectangle, and gives each guess a confidence from 0 to 1. Ultralytics owns
that large trained network; this file clearly shows the numbers going into and
coming out of it.
"""

# These hints let types refer to names defined later in the file.
from __future__ import annotations

# io provides an in-memory byte bucket; os reads optional settings; threading
# keeps two web requests from trying to use the shared model at the same time.
import io
import os
import threading
from dataclasses import dataclass
from typing import Any

# Pillow supplies the photograph type and simple rectangle/text drawing tools.
from PIL import Image, ImageDraw, ImageFont


# Eight high-contrast label colors are reused in a loop when many objects appear.
PALETTE = (
    "#70F0C0",
    "#FFC857",
    "#7AB8FF",
    "#FF7A90",
    "#B79CFF",
    "#F59E5B",
    "#57D6E8",
    "#E879F9",
)


class DetectionError(RuntimeError):
    """A safe, friendly error that the API is allowed to show to a user."""


@dataclass(frozen=True)
class DetectionResult:
    """The two things returned after one photograph is analyzed.

    ``image_bytes`` is the finished JPEG. ``detections`` is a list of readable
    records such as label="dog", confidence=0.91, box=[left, top, right, bottom].
    """

    image_bytes: bytes
    detections: list[dict[str, Any]]


class YoloDetector:
    """Load one pretrained YOLO model only when needed, then share it safely."""

    def __init__(self, model_name: str | None = None) -> None:
        # Prefer an explicit filename, then an environment setting, then the
        # small default YOLOv8 nano checkpoint bundled with this project.
        self.model_name = model_name or os.getenv("YOLO_MODEL", "yolov8n.pt")
        # None means the heavy neural network has not been opened yet.
        self._model = None
        # If loading fails once, remember the safe message instead of retrying
        # an expensive broken operation on every request.
        self._load_error: str | None = None
        # A re-entrant lock is a one-person doorway around shared model work.
        self._lock = threading.RLock()

    @property
    def state(self) -> str:
        """Return a tiny health label: loaded, error, or cold (not opened yet)."""

        if self._model is not None:
            return "loaded"
        if self._load_error:
            return "error"
        return "cold"

    def _get_model(self):
        """Return the shared YOLO network, loading it on the first request."""

        # Only one request may enter the loading doorway at a time.
        with self._lock:
            # A previous request already loaded the network, so reuse it.
            if self._model is not None:
                return self._model
            # A previous load failed; return the same clean explanation.
            if self._load_error:
                raise DetectionError(self._load_error)
            try:
                # Import here so the Flask page can start before the heavy ML
                # package and weight file are needed.
                from ultralytics import YOLO

                # Read millions of learned numbers from yolov8n.pt. Those numbers
                # are like tuned knobs that turn pixel patterns into object guesses.
                self._model = YOLO(self.model_name)
                return self._model
            except Exception as exc:  # library/download failures vary by platform
                # Do not expose a long internal traceback through the public API.
                self._load_error = (
                    "The YOLO model could not start. Make sure the dependencies are installed "
                    "and the computer is online for the first model download."
                )
                # ``from exc`` keeps the original technical cause in server logs.
                raise DetectionError(self._load_error) from exc

    def detect(self, image: Image.Image, confidence: float) -> DetectionResult:
        """Find objects above ``confidence`` and return JSON-like boxes + JPEG.

        Input example: a 1280×720 RGB image and confidence 0.35.
        Output example: ``[{label: 'dog', confidence: .91, box: [20,40,500,690]}]``
        plus a JPEG with that rectangle drawn on it.
        """

        # Load or reuse the pretrained CNN.
        model = self._get_model()
        try:
            # Keep one request inside the model doorway. Shared GPU/CPU model
            # objects are not assumed to be safe for simultaneous prediction.
            with self._lock:
                results = model.predict(
                    # source is the RGB Pillow image supplied by app.py.
                    source=image,
                    # conf is the smallest allowed certainty: 0.35 means 35%.
                    conf=confidence,
                    # YOLO resizes its working canvas; 640 is the normal default.
                    imgsz=int(os.getenv("YOLO_IMAGE_SIZE", "640")),
                    # Keep library progress chatter out of the web-server console.
                    verbose=False,
                )
        except Exception as exc:
            raise DetectionError("The model could not analyze this image. Try another photo.") from exc

        # This app sends one image, so the first result is the only result.
        result = results[0]
        # names maps numeric class IDs such as 16 to words such as "dog".
        names = result.names
        # Start an empty basket for clean, browser-friendly detection records.
        detections: list[dict[str, Any]] = []

        # Some photos have no accepted guesses; in that case boxes may be None.
        if result.boxes is not None:
            # Each box contains a class ID, confidence, and four pixel edges.
            for index, box in enumerate(result.boxes):
                # Tensor values are tiny containers; item() takes out one number.
                class_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                # xyxy means left x, top y, right x, bottom y, measured in pixels.
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
                detections.append(
                    {
                        # Change class 16 into a readable label such as "dog".
                        "label": str(names[class_id]),
                        "class_id": class_id,
                        # Four decimals preserve useful detail without noisy tails.
                        "confidence": round(score, 4),
                        # Whole pixels are enough for drawing a screen rectangle.
                        "box": [round(x1), round(y1), round(x2), round(y2)],
                        # Modulo wraps object 9 back to palette color 1.
                        "color": PALETTE[index % len(PALETTE)],
                    }
                )

        # Put the strongest guesses first in the result list.
        detections.sort(key=lambda item: item["confidence"], reverse=True)
        # Reassign colors after sorting so list order and color order match.
        for index, detection in enumerate(detections):
            detection["color"] = PALETTE[index % len(PALETTE)]

        # Draw on a copy so the caller's original image stays untouched.
        annotated = self._annotate(image.copy(), detections)
        # Use a memory bucket instead of saving an uploaded photo to disk.
        output = io.BytesIO()
        # JPEG quality 90 is clear while smaller than an uncompressed image.
        annotated.save(output, format="JPEG", quality=90, optimize=True)
        # Return raw bytes and the structured explanation together.
        return DetectionResult(image_bytes=output.getvalue(), detections=detections)

    @staticmethod
    def _annotate(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
        """Draw every box and its label on the supplied image copy."""

        # RGBA mode permits transparent drawing colors when needed.
        draw = ImageDraw.Draw(image, "RGBA")
        # Make text scale with the smaller image side, but never below 14 pixels.
        font = ImageFont.load_default(size=max(14, round(min(image.size) / 42)))
        # Make box lines scale too, with a minimum width of 3 pixels.
        stroke = max(3, round(min(image.size) / 220))
        # Leave readable breathing room around each label word.
        padding = max(5, stroke * 2)

        # Paint detections from strongest to weakest.
        for detection in detections:
            # Unpack the rectangle edges in image pixels.
            x1, y1, x2, y2 = detection["box"]
            color = detection["color"]
            # Turn 0.9134 into a compact "dog  91%" tag.
            label = f'{detection["label"]}  {detection["confidence"] * 100:.0f}%'

            # Outline the object without covering the photograph inside the box.
            draw.rounded_rectangle(
                (x1, y1, x2, y2),
                radius=stroke * 2,
                outline=color,
                width=stroke,
            )
            # Ask Pillow how much space the label text needs.
            text_box = draw.textbbox((0, 0), label, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            # Prefer a tag above the object, but never draw above pixel 0.
            label_top = max(0, y1 - text_height - padding * 2)
            # Keep the label background inside the image's right edge.
            label_right = min(image.width, x1 + text_width + padding * 2)
            draw.rounded_rectangle(
                (x1, label_top, label_right, y1),
                radius=stroke * 2,
                fill=color,
            )
            # Dark text contrasts with every bright palette color.
            draw.text(
                (x1 + padding, label_top + padding - 1),
                label,
                font=font,
                fill="#07120F",
            )
        # Hand the finished annotated picture back to detect().
        return image
