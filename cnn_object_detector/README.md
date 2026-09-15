# Sightline — CNN Object Detector

Sightline is a simple local web app that detects common objects in an uploaded photo or a frame captured from the device camera. A Flask API sends the image to a pretrained Ultralytics YOLOv8 nano model in Python, draws labeled bounding boxes, and returns the annotated image plus structured detection data to the browser.

## Quick start

Python 3.10 or newer is recommended.

```powershell
cd C:\Users\User\machine_learning\cnn_object_detector
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. The first detection downloads the small `yolov8n.pt` checkpoint if it is not already cached, so keep an internet connection available for that one-time step.

## Use the app

1. Select **Upload photo**, or select **Use camera** and allow browser permission.
2. Choose/capture a frame.
3. Adjust **Minimum confidence** if needed. Higher values reduce uncertain predictions.
4. Select **Run detection**.
5. Review the bounding boxes and confidence scores, then save the annotated image if useful.

Camera access works on `localhost` and secure HTTPS pages. The browser captures a still frame; the Python server performs the CNN inference. Photos are handled in memory and are not written to disk by this app.

## Configuration

Optional environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `YOLO_MODEL` | `yolov8n.pt` | Use another compatible Ultralytics detection checkpoint. |
| `YOLO_IMAGE_SIZE` | `640` | Model inference size. Larger can help small objects but costs time and memory. |
| `PORT` | `5000` | Local HTTP port. |
| `FLASK_DEBUG` | unset | Set to `1` for Flask development reloads. |

## Test

The tests use a tiny stub detector, so they do not download model weights.

```powershell
python -m pytest -q
```

For architecture, behavior, limitations, and improvement ideas, see [REPORT.md](REPORT.md).
