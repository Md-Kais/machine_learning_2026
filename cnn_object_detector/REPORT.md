# Sightline Object Detector — Project Report

## 1. Problem Title & Purpose

### Title

**Finding and Drawing Boxes Around Everyday Objects in a Photo**

### Architecture checklist

**Type B: Server-Backed ML.** The browser sends an image to Flask with a `POST
/api/detect` request. Python runs a pretrained YOLOv8n convolutional neural
network (CNN), draws boxes, and returns JSON plus an annotated image. Inference
cannot run from `static/app.js` alone.

### Why it matters

Finding both *what* is in an image and *where* it is can help people organize
photos, count common items, and learn how computer vision works. Sightline turns
that difficult job into one upload and one button while keeping the model's
boxes and confidence visible. It is a general learning demo, not a tool for
safety, identity, medical, surveillance, or access-control decisions.

## 2. How the Algorithm Works (10-Year-Old Explanation)

### The story metaphor

Imagine a super-fast picture detective looking through a window covered by a
grid. The detective does not begin by understanding a whole dog or bicycle.
Instead, many small CNN filters slide across the image like tiny magnifying
glasses:

1. Early filters notice simple clues such as light/dark edges.
2. Deeper filters combine edges into textures, corners, and parts.
3. Later filters combine parts into object-shaped clues.
4. YOLO looks at the whole picture in one trip and proposes labeled rectangles.

**YOLO** means “You Only Look Once.” That name describes its one-pass style: the
network predicts likely object boxes and labels together instead of slowly
searching the same image again and again.

### Inputs and outputs

| Item | Everyday meaning | Numbers used by the app |
|---|---|---|
| Image pixels | Tiny colored tiles making the photograph | RGB values; source under 40 megapixels |
| Minimum confidence | How sure a guess must be before it is shown | 0.10–0.90 (10–90%) |
| Inference size | Working canvas fed to YOLO | 640 by default |
| Class ID | A numeric name tag learned from COCO | Mapped to one of 80 words |
| Box | Rectangle around an object | `[left x, top y, right x, bottom y]` pixels |
| Confidence | Model strength for one guess | Decimal from 0 to 1; not a guarantee |
| Annotated output | The original view with labels and boxes | In-memory JPEG data URL |

The app accepts JPG, PNG, WEBP, or BMP files up to 12 MB. Large sides are shrunk
to at most 1,920 pixels before inference so ordinary computers do not have to
carry an unnecessarily huge picture.

### Step-by-step magic

1. The user uploads a photo or captures one camera frame.
2. JavaScript checks the browser MIME type and 12 MB limit.
3. `FormData` packs the image and confidence setting.
4. `fetch()` sends them to `POST /api/detect`.
5. Flask reads only up to the allowed size plus one byte.
6. Pillow verifies the real image format and rejects unreadable or enormous data.
7. EXIF rotation is applied so phone photos face the correct way.
8. The image becomes RGB and is downscaled when needed.
9. `YoloDetector` lazily loads `yolov8n.pt` on the first detection.
10. The pretrained CNN changes pixels into feature maps—stacks of clue grids.
11. YOLO proposes boxes, object classes, and confidence values in one network pass.
12. The Ultralytics prediction pipeline filters weak guesses using the selected
    threshold and removes many overlapping duplicates.
13. Python extracts each numeric class, confidence, and `x1,y1,x2,y2` box.
14. Detections are sorted from strongest to weakest.
15. Pillow draws a bright rectangle and plain label on an image copy.
16. The JPEG is encoded as text-safe base64 and returned with structured JSON.
17. JavaScript shows a result badge, annotated image, object count, elapsed time,
    and one confidence bar per detected object.

### Key code breakdown

```python
results = model.predict(
    source=image,
    conf=confidence,
    imgsz=640,
    verbose=False,
)
```

The RGB picture and a threshold such as `0.35` go into the pretrained detective.
It works on a 640-sized canvas and gives back boxes that survived the confidence
and overlap filters.

```python
class_id = int(box.cls[0].item())
score = float(box.conf[0].item())
x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
```

One model answer is unpacked like a parcel. `class_id` is the object's number
tag, `score` is the guess strength, and the four coordinates are the left, top,
right, and bottom rectangle edges measured in pixels.

```python
"label": str(names[class_id]),
"confidence": round(score, 4),
"box": [round(x1), round(y1), round(x2), round(y2)],
```

The number tag becomes a word such as `dog`, the score becomes a compact decimal,
and fuzzy coordinates become whole pixels that are easy to draw.

```javascript
const form = new FormData();
form.append("image", state.file);
form.append("confidence", String(Number(el.confidenceInput.value) / 100));
const response = await fetch("/api/detect", { method: "POST", body: form });
```

The browser puts the photo and slider number into a digital envelope. Dividing
35 by 100 changes `35%` into `0.35`, the number expected by Python.

### What the pretrained model learned

The included YOLOv8n checkpoint was pretrained for 80 common COCO categories. It
does not learn from photos uploaded to this app. Small, dark, blurred, partly
hidden, unusual, or unsupported objects may be missed or mislabeled, and a high
confidence is still not proof.

## 3. Simple Web Development Guide

### Folder and file map

```text
cnn_object_detector/
├── static/
│   ├── app.js              Upload, camera, API call, and result rendering
│   └── styles.css          Responsive detector interface and status colors
├── templates/
│   └── index.html          Image controls, presets, result card, and object list
├── tests/
│   └── test_app.py         Flask contract tests with a fake detector
├── app.py                  Flask routes and image/request guardrails
├── detector.py             YOLO loading, inference extraction, and box drawing
├── yolov8n.pt              Pretrained network weights
├── requirements.txt        Flask, Pillow, Ultralytics, and test dependencies
├── README.md               Setup instructions
└── REPORT.md               This guide
```

### How it works together

1. `index.html` provides upload, camera, confidence, and run controls. Two preset
   buttons set balanced or cautious confidence instantly.
2. `static/app.js` validates a selected file, then sends multipart form data to
   `/api/detect` with `fetch()`.
3. `app.py` validates and prepares the image before calling
   `YoloDetector.detect()`.
4. `detector.py` returns image bytes and detection records. Flask wraps those in
   JSON, and JavaScript renders the badge, image, statistics, and list.

### Key functions

| Function | Role |
|---|---|
| `chooseFile()` | Checks and previews an uploaded/captured image. |
| `runDetection()` | Builds the POST request and handles success/errors. |
| `_read_image()` | Enforces server-side size, format, orientation, and dimensions. |
| `YoloDetector.detect()` | Runs YOLO and returns boxes plus an annotated JPEG. |

## 4. Kid-Friendly User Manual

### Quick summary

Choose a clear photo, then press **Run detection**. The app draws labeled boxes
around guesses and lists how confident the model is about each one.

### Three steps

1. **Choose an image** — upload JPG/PNG/WEBP/BMP under 12 MB, or capture a camera
   frame. Optionally choose **Balanced 35%** or **Cautious 60%**.
2. **Click the button** — press **Run detection**.
3. **Read the badge** — inspect the “objects found” badge, boxes, labels, and list.

### Understanding the result

| Badge/color | Plain meaning |
|---|---|
| Green — “objects found” | One or more guesses passed the selected threshold. |
| Yellow — “No strong match” | No guess was strong enough; try a clearer photo or lower threshold. |
| Yellow — “Analyzing image…” | The Python model is working. |
| Red — “Could not analyze” | The request failed; read the friendly warning and try again. |
| Colored object bar | The label's confidence strength; it is not a guarantee. |

Higher confidence hides weaker guesses but can also hide real objects. Lower
confidence shows more guesses but may include more mistakes.
