# Decision Tree ER Lab — Project Report

## 1. Problem Title & Purpose

### Title

**Learning a Yes-or-No Decision Tree with Pretend Cardiac-Triage Cards**

### Architecture checklist

**Type A: Client-Side Browser ML.** Python grows the tree and exports its nested
questions to `dist/model.json`. JavaScript follows those questions locally in
the browser through `dist/tree.js`; it does not send measurements to an API.

### Why it matters

Decision Trees are useful for teaching because every choice can be read like an
ordinary `if`/`else` story. This demo makes that story visible with two synthetic
measurements. Its data, thresholds, and labels are made up for programming
practice and must never be used to discharge, admit, diagnose, or treat anyone.

## 2. How the Algorithm Works (10-Year-Old Explanation)

### The story metaphor

Imagine 48 pretend cards. Each card shows two numbers and wears either a blue or
red sticker. We want to build a short hallway of yes/no gates that sends cards
with the same sticker toward the same final room.

At one gate we might ask, “Is the first number 140 or less?” Values on or below
the fence walk left; larger values walk right. Training is a fence contest: the
computer tries every sensible fence and keeps the one that makes the new rooms
least mixed. A final room is called a **leaf**.

### Inputs and outputs

| Item | Physical meaning | Teaching range |
|---|---|---:|
| Systolic blood pressure | The top pressure number measured by a cuff | 90–180 mmHg |
| Maximum heart rate | The fastest heart-beat count in one minute | 60–160 bpm |
| Output | The synthetic sticker on the reached leaf | Blue toy label or red toy label |

The interface uses the requested names “Low Risk / Discharge” and “High Risk /
Admit to ICU,” but they are only toy class names. Two vital signs are nowhere
near enough for real emergency care.

### Step-by-step magic

1. Python reads each CSV row and rejects missing, non-positive, or unknown values.
2. The root begins as one bag holding every card.
3. The computer measures how mixed the two sticker colors are with Gini impurity.
4. For each number column, it sorts the unique values.
5. It places candidate fences halfway between neighboring numbers.
6. Each fence splits the cards into a left basket and a right basket.
7. The two baskets receive a size-weighted messiness score.
8. The fence that removes the most mess becomes the node's question.
9. The same search repeats inside the two child baskets.
10. Growth stops at depth two, in a pure basket, or when a basket is too small.
11. The final tree is exported to `dist/model.json`.
12. In the browser, a new case starts at the root and follows Yes/left or No/right
    until it reaches a leaf.
13. The result card prints the toy label and `app.js` highlights the exact path
    and rectangular region on the graph.

### Key code breakdown

```python
return 1.0 - sum((count / total) ** 2 for count in counts.values())
```

Gini impurity is a “mixed-marble meter.” If all marbles in a bag are blue, the
score is `1 - 1² = 0`, perfectly sorted. If blue and red are evenly mixed, the
score is larger. Squaring each color's share makes a strong majority stand out.

```python
thresholds = [
    (left_value + right_value) / 2
    for left_value, right_value in zip(values, values[1:])
]
```

The fence is placed halfway between neighboring cards. If the numbers are 130
and 150, the candidate fence is `(130 + 150) / 2 = 140`. No card balances on the
fence.

```python
weighted_impurity = (
    len(left) / len(examples) * gini_impurity(left)
    + len(right) / len(examples) * gini_impurity(right)
)
gain = parent_impurity - weighted_impurity
```

A room holding many cards gets more say than a room holding one card. `gain` is
the amount of mess removed by the fence. The largest positive gain wins.

```javascript
const value = values[node.feature];
const goLeft = value <= node.threshold;
node = goLeft ? node.left : node.right;
```

The browser opens the drawer named by the question, compares its number with the
fence, and walks to exactly one child. It repeats until `node.is_leaf` is true.

### Safety boundary

The dataset is synthetic, the labels are deliberately toy labels, and the model
uses only two values. Real chest-pain assessment can involve symptoms, ECG,
laboratory tests, medical history, examination, and professional judgment. For
possible real heart-attack symptoms, contact local emergency services.

## 3. Simple Web Development Guide

### Folder and file map

```text
decisionTree/
├── data/
│   └── cardiac_triage.csv       Synthetic labeled teaching cards
├── dist/
│   ├── index.html               Two inputs, presets, path, result, and graph
│   ├── styles.css               Responsive monitor theme and label colors
│   ├── app.js                   Validation, rendering, and SVG drawing
│   ├── tree.js                  Browser tree walking and region helpers
│   └── model.json               Exported learned questions and leaves
├── tests/
│   ├── test_decision_tree.py    Python training and path checks
│   └── test_browser_tree.js     Browser model and visualization checks
├── decision_tree_model.py       From-scratch training algorithm
├── train_model.py               Training/export entry point
├── start_app.py                 Local static-file server
├── README.md                    Setup and safety notes
└── REPORT.md                    This guide
```

### How it works together

1. The form collects the two synthetic measurements. Helper text gives the
   teaching ranges, and two quick buttons provide ready-made blue/red examples.
2. `app.js` fetches `model.json`. Submitting the form runs range checks and calls
   `DecisionTree.predict()` inside `tree.js`.
3. The returned label and breadcrumb path update the result card and path list.
   Region helpers turn leaves and thresholds into SVG rectangles and fence lines.

### Key functions

| Function | Role |
|---|---|
| `loadModel()` | Loads the exported tree and enables evaluation. |
| `validateInput()` | Displays friendly missing/out-of-range messages. |
| `predict()` | Walks from root question to final leaf. |
| `renderPrediction()` | Shows the label, path, and highlighted graph region. |

## 4. Kid-Friendly User Manual

### Quick summary

Enter two pretend measurements or choose a sample, then press **Follow the tree**.
Watch the page answer each yes/no question and land in a colored final room.

### Three steps

1. **Type numbers** — enter values within the ranges printed below the boxes, or
   press **Try blue toy case** / **Try red toy case**.
2. **Click the button** — press **Follow the tree**.
3. **Read the badge** — read the large toy label, then follow its numbered path.

### Understanding the result

| Badge | What it means here |
|---|---|
| Blue — Low Risk / Discharge | The inputs reached a blue leaf in the made-up lesson. It is not permission to discharge anyone. |
| Red — High Risk / Admit to ICU | The inputs reached a red leaf in the made-up lesson. It is not an ICU decision. |
| Green outline on graph | This is the rectangular leaf reached by the current inputs. |
| Green diamond | This is the current synthetic case on the two-number map. |

Use the colors to understand tree branches only. Do not use this application for
a real person or real medical decision.
