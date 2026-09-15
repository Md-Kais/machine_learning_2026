# Decision Tree ER Lab — a tiny white-box ML project

This folder contains a Decision Tree written from scratch in normal Python. It
uses a **synthetic emergency-room cardiac triage scenario** to teach two numeric
features:

1. Systolic blood pressure (`mmHg`) on the graph's x-axis
2. Maximum heart rate (`bpm`) on the graph's y-axis

The requested target labels are:

- **Low Risk / Discharge** — blue circles and blue regions
- **High Risk / Admit to ICU** — red squares and red regions

These are **toy labels**, not medical instructions. The dataset is made up, and
the app must never be used to discharge, admit, diagnose, or treat a person.

## Run it on localhost

Open PowerShell in this folder:

```powershell
cd C:\Users\User\machine_learning\decisionTree
```

Train and export the Decision Tree:

```powershell
python train_model.py
```

Start the website:

```powershell
python start_app.py
```

Open the exact address printed in the terminal. It normally uses:

```text
http://127.0.0.1:8002
```

Port `8002` lets this page run beside the KNN app on `8000` and the SVM app on
`8001`. If `8002` is busy, the script tries the next available port. Keep the
terminal open and press `Ctrl+C` when you want to stop the server.

## Decision Trees in words a child can understand

Imagine a guessing game. You ask one yes-or-no question, then choose the next
branch from the answer:

```text
Is systolic BP <= 140?
├─ Yes → blue toy label
└─ No  → Is maximum heart rate <= 115?
         ├─ Yes → blue toy label
         └─ No  → red toy label
```

Each final answer box is called a **leaf**. The entire model is only nested
`if`, `else`, and `return` instructions, so a person can inspect every decision.

## How training works

The Python program does not begin with `140` or `115`. It learns them:

1. Try cuts halfway between every neighboring number.
2. Put values less than or equal to the cut on the left.
3. Put larger values on the right.
4. Measure how mixed the two new groups are with **Gini impurity**.
5. Keep the cut producing the purest groups.
6. Repeat inside each child group until depth two or a pure leaf is reached.

Gini impurity is a mix meter:

```text
gini = 1 - (blue_fraction² + red_fraction²)
```

`0` means the box contains only one label. A larger number means the labels are
mixed. Because the CSV was designed as a clear teaching grid, the learned tree
classifies all 48 synthetic rows correctly.

## How to read the visualization

- Blue circles show synthetic rows with the blue toy label.
- Red squares show synthetic rows with the red toy label.
- The x-axis shows systolic blood pressure.
- The y-axis shows maximum heart rate.
- **Split 1** is a vertical line at `140` because it asks about an x-axis value.
- **Split 2** is a horizontal line at `115` and exists only to the right of
  Split 1 because only that branch asks the second question.
- Blue and red rectangles are the three final leaves.
- The active leaf gets a bright green outline.
- The green diamond is the current user input.
- The decision-path list repeats every yes/no answer in plain text.

This shows why Decision Trees make axis-aligned, stair-step boundaries instead
of the diagonal line of a linear SVM or distance neighborhoods of KNN.

## File-by-file summary

### `data/cardiac_triage.csv`

This is the 48-row synthetic dataset. It contains a simple grid of blood
pressure and heart-rate values plus the requested toy label. CSV is used because
it looks like an ordinary table that a beginner can edit. No rows are patient
records and no threshold is a clinical rule.

### `decision_tree_model.py`

This is the actual machine-learning algorithm. It loads and validates the CSV,
calculates Gini impurity, tries every possible numeric split, recursively grows
a depth-two tree, follows a prediction path, prints readable rules, and exports
the model. It uses no scikit-learn, NumPy, or other package, so every important
step is visible.

### `train_model.py`

This is the short training entry point. It loads the 48 rows, calls `fit()`,
prints training accuracy and the complete learned `if/else` rules, and creates
`dist/model.json`. Separating this file makes the order of training easy to see.

### `start_app.py`

This starts localhost using Python's built-in web server. It serves only the
`dist` folder, begins at port `8002`, automatically tries another port if needed,
and prints the exact address. No package installation is required.

### `dist/model.json`

This generated JSON file contains the complete trained tree, feature names,
units, target-label names, node counts, impurity, thresholds, child nodes, and
all graph points. JavaScript can read JSON directly. Do not edit it by hand; run
`python train_model.py` after changing the CSV.

### `dist/index.html`

This defines the app's accessible structure: emergency warning, two labeled
inputs, teaching result, yes/no path, responsive graph, complete learned rule,
medical-safety explanation, and authoritative reference links. Live regions
announce changing results to assistive technology.

### `dist/styles.css`

This creates the monitor-inspired visual design, readable inputs, blue and red
result states, rectangular leaf regions, active branch styling, graph labels,
responsive mobile layout, and short animations. Colors are paired with circles,
squares, labels, and line styles so color is never the only clue. Reduced-motion
settings are respected.

### `dist/tree.js`

This is the browser's small Decision Tree brain. It follows exported nodes,
returns the selected label and complete path, converts leaf nodes into graph
rectangles, and converts question nodes into vertical or horizontal split lines.
Keeping it separate from the page makes the logic easy to test.

### `dist/app.js`

This connects the form, `tree.js`, and SVG visualization. It validates the two
inputs, updates the toy result, creates the human-readable path, colors the
active rectangle, plots the green input diamond, and redraws on screen-size
changes. Browsers supporting WebMCP also get one structured
`evaluate_synthetic_cardiac_tree` action that updates the same visible state.

### `tests/test_decision_tree.py`

These Python tests check the dataset, 100% training fit, learned `140` and `115`
questions, all three leaf paths, invalid values, safety warning, and JSON export.
They use Python's built-in `unittest` module.

### `tests/test_browser_tree.js`

These Node.js tests check the exact model used by the webpage. They verify the
learned questions, blue and red predictions, path lengths, three rectangular
regions, vertical and horizontal splits, important page elements, responsive
CSS, safety wording, and invalid-input behavior.

### `.gitignore`

This ignores automatic Python cache files and local server logs because they are
not source code.

## Run the tests

Python:

```powershell
python -m unittest discover -s tests -v
```

Browser model:

```powershell
node tests/test_browser_tree.js
```

## Medical safety references

- [ACC summary of the 2021 AHA/ACC Chest Pain Guideline](https://www.acc.org/Latest-in-Cardiology/ten-points-to-remember/2021/10/27/14/06/2021-Guideline-for-Chest-Pain-gl_chestpain)
- [NHLBI: Heart Attack Symptoms](https://www.nhlbi.nih.gov/health/heart-attack/symptoms)
- [NHLBI: Heart Attack Diagnosis](https://www.nhlbi.nih.gov/health/heart-attack/diagnosis)

The guideline emphasizes ECG, serial cardiac troponin, structured risk
assessment, symptoms, and other findings. If a real person may be having a heart
attack, contact local emergency services immediately instead of entering values
into this programming demo.
