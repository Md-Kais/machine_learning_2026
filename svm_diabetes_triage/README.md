# SVM Triage Lab — a tiny linear SVM project

This separate project teaches a **Support Vector Machine (SVM)** using two
features:

1. Fasting blood glucose in milligrams per decilitre (`mg/dL`)
2. Body Mass Index (`BMI`)

The Python model learns a straight separator between two groups of teaching
examples. The space between the two margin lines becomes the third teaching
result: **Pre-diabetic / Further Diagnostic Needed**. The browser draws the
dataset, support examples, separator, margin, clinical reference lines, and the
user's input on the same x/y graph.

> **Medical safety:** This synthetic-data project explains SVM; it cannot
> diagnose or exclude diabetes. A clinician should interpret laboratory
> results. NIDDK notes that diagnosis usually requires a second abnormal test.
> BMI is a risk factor, not a blood-glucose test.

## Run it

Open PowerShell in this folder:

```powershell
cd C:\Users\User\machine_learning\svm_diabetes_triage
```

Train the SVM and create the browser model:

```powershell
python train_model.py
```

Start the website:

```powershell
python start_app.py
```

Open the exact address printed by the command. It normally starts at:

```text
http://127.0.0.1:8001
```

The server tries the next available port automatically if 8001 is already in
use. Keep the terminal open while using the page. Press `Ctrl+C` to stop it.

## The SVM idea in child-friendly words

Imagine blue dots standing on one side of a playground and red squares on the
other side. SVM tries to draw a straight road between them:

1. The **center stripe** is the decision line.
2. The **road edges** are the two margin lines.
3. The dots touching or closest to the road edges are **support vectors**. They
   are important because they decide where the road goes.
4. During training, a point on the wrong side pushes the road a tiny step.
5. After many tiny steps, the model remembers two weights and one bias.

The learned score is:

```text
score = glucose_weight × scaled_glucose
      + BMI_weight     × scaled_BMI
      + bias
```

The result rule is:

```text
score <= -1      → Lower screening signal
-1 < score < 1   → Pre-diabetic / Further Diagnostic Needed
score >= 1       → Higher screening signal
```

The score is **not a probability**. It only says where the point sits relative
to the learned line and its two margin edges.

## Why the dataset has no middle training rows

The CSV deliberately contains only two outside reference groups:

- lower examples have fasting glucose at or below 99 mg/dL;
- higher examples have fasting glucose at or above 126 mg/dL.

The model learns between these groups. The unused space becomes the SVM margin,
which this teaching app names the follow-up zone. This makes the meaning of a
margin visible without hiding the idea inside a large ML library.

The dataset is **synthetic educational data**, not patient records and not a
validated clinical model.

## How to read the graph

- The **x-axis** is fasting blood glucose in `mg/dL`.
- The **y-axis** is BMI.
- Blue circles are lower-reference training examples.
- Red squares are higher-reference training examples.
- Dark rings mark the six examples closest to their margin edges—the support
  examples.
- The solid dark line is the learned SVM separator.
- The orange dashed lines are the SVM margin edges.
- The orange band is the **Pre-diabetic / Further Diagnostic Needed** teaching
  zone.
- The green diamond is the current user input.
- The thin horizontal dashed segment shows the input's side and distance from
  the center separator.
- Vertical dotted lines at 100 and 126 are clinical fasting-glucose references.
  They are separate from the SVM prediction.

The learned line tilts slightly because BMI is a risk clue in the teaching data,
but glucose has much more influence. Real diagnosis does not come from this
two-feature SVM.

## File-by-file summary

### `data/diabetes_triage.csv`

This is the small synthetic training dataset. Each row has fasting glucose, BMI,
and one of two outside labels. CSV looks like a table, so a beginner can open it
and understand every example. The clinically important middle glucose rows are
left out so the SVM margin can represent the follow-up zone.

### `svm_model.py`

This is the actual machine-learning code. It loads the CSV, scales the two
features, trains a linear soft-margin SVM with repeated gradient steps, finds
support examples, predicts one of three teaching zones, and exports the learned
numbers. It uses only normal Python—no scikit-learn or NumPy—so every important
step is visible.

### `train_model.py`

This is the short file you run to train the model. It loads all 40 examples,
calls `fit()`, prints training-side accuracy and the two learned weights, and
saves `dist/model.json`. Keeping these actions separate makes the program order
easy to follow.

### `start_app.py`

This starts a local static-file server using Python's standard library. It uses
port 8001 so it can run beside the original BMI KNN project on port 8000. If the
port is busy, it tries another one and prints the exact address to open.

### `dist/model.json`

This generated file is the trained SVM that JavaScript can read. It contains the
feature means and scales, two weights, bias, margin values, clinical reference
numbers, all graph points, and six support-example markers. Do not edit it by
hand; run `python train_model.py` instead.

### `dist/index.html`

This is the structure and plain-language text of the UI. It contains the two
inputs, animated result, responsive graph, tiny SVM lesson, safety warning, and
authoritative medical links. Labels and live regions help keyboard and
screen-reader users.

### `dist/styles.css`

This creates the responsive clinical-notebook design. It styles the two-column
workspace, large inputs, three result states, graph axes, region colors, point
shapes, line types, and small animations. Color is always paired with shape or
text, and reduced-motion settings are respected.

### `dist/svm.js`

This is the small browser version of the SVM maths. It standardizes glucose and
BMI, calculates the decision score, chooses a zone, and calculates where a
margin line crosses the graph. Keeping maths here makes it testable without the
buttons or SVG drawing code.

### `dist/app.js`

This connects the form to `svm.js`. It validates input, updates the visible
result, explains the clinical fasting-glucose reference, and redraws the SVG
when the input or screen size changes. Browsers with WebMCP support also receive
one structured `screen_diabetes_triage` action that updates the same visible UI.

### `tests/test_svm.py`

These Python tests check the dataset, training accuracy, all three zones, bad
input, model export, and support-example count. They use Python's built-in
`unittest` module.

### `tests/test_browser_model.js`

These Node.js tests check the exact SVM maths used by the webpage. They verify
the three teaching zones, boundary calculations, exported points, support
examples, graph elements, responsive CSS, and invalid-input behavior.

### `.gitignore`

This keeps automatic Python cache files and local server logs out of version
control because they are not source code.

## Run the tests

Python tests:

```powershell
python -m unittest discover -s tests -v
```

Browser-model tests:

```powershell
node tests/test_browser_model.js
```

## Medical references used for the teaching labels

- [CDC: Diabetes Testing](https://www.cdc.gov/diabetes/diabetes-testing/index.html)
- [NIDDK: Diabetes Tests & Diagnosis](https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis)
- [NIDDK: Risk Factors for Type 2 Diabetes](https://www.niddk.nih.gov/health-information/diabetes/overview/risk-factors-type-2-diabetes)

For non-pregnant people, these sources list fasting plasma glucose of 99 mg/dL
or below as normal, 100–125 mg/dL as prediabetes, and 126 mg/dL or above as a
diabetes threshold. Testing conditions and confirmation matter; this page is a
programming lesson, not a replacement for medical care.
