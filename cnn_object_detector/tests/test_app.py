import io

from PIL import Image

from app import create_app
from detector import DetectionResult


class StubDetector:
    model_name = "stub-model.pt"
    state = "loaded"

    def detect(self, image, confidence):
        output = io.BytesIO()
        image.save(output, "JPEG")
        return DetectionResult(
            image_bytes=output.getvalue(),
            detections=[
                {
                    "label": "test object",
                    "class_id": 0,
                    "confidence": confidence,
                    "box": [0, 0, image.width, image.height],
                    "color": "#70F0C0",
                }
            ],
        )


def image_file(format_name="PNG"):
    stream = io.BytesIO()
    Image.new("RGB", (24, 16), "white").save(stream, format_name)
    stream.seek(0)
    return stream


def test_home_page_loads():
    client = create_app(StubDetector()).test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert b"SIGHTLINE" in response.data


def test_detection_returns_annotation_and_metadata():
    client = create_app(StubDetector()).test_client()
    response = client.post(
        "/api/detect",
        data={"image": (image_file(), "frame.png"), "confidence": "0.45"},
        content_type="multipart/form-data",
    )
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["count"] == 1
    assert payload["detections"][0]["label"] == "test object"
    assert payload["image"].startswith("data:image/jpeg;base64,")


def test_detection_rejects_missing_image():
    client = create_app(StubDetector()).test_client()
    response = client.post("/api/detect", data={"confidence": "0.35"})
    assert response.status_code == 400
    assert "photo" in response.get_json()["error"].lower()


def test_detection_rejects_invalid_threshold():
    client = create_app(StubDetector()).test_client()
    response = client.post(
        "/api/detect",
        data={"image": (image_file(), "frame.png"), "confidence": "0.99"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
