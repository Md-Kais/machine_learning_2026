# SVM Triage Lab — Project Report

## 1. Problem Title & Purpose

### Title

**Drawing a Wide Straight Road Between Two Diabetes-Screening Example Groups**

### Architecture checklist

**Type A: Client-Side Browser ML.** Python trains the linear SVM and exports its
means, scales, weights, bias, and margin to `dist/model.json`. `dist/svm.js`
performs inference directly in the browser. No measurement is posted to a server.

### Why it matters

People learning about diabetes screening often see cutoffs without seeing how a
machine-learning boundary behaves. This project makes a straight SVM boundary
and its uncertain margin visible. It is a synthetic programming lesson, not a
diagnosis; laboratory conditions, confirmation, symptoms, pregnancy, medicines,
and professional interpretation all matter in real care.

## 2. How the Algorithm Works (10-Year-Old Explanation)

### The story metaphor

Imagine blue players on one side of a playground and red players on the other.
We want to paint the widest straight road that fits between the teams:

- The stripe down the road's center is score `0`.
- One road edge is score `-1`.
- The other road edge is score `+1`.
- The players closest to the edges are the **support examples**. Like tent pegs,
  they have the strongest say in where the road can sit.

The SVM repeatedly nudges its road. A player on the wrong side or inside the road
pushes it away. A gentle rubber band stops the road from tilting too wildly.

### Inputs and outputs

| Item | Physical meaning | UI range |
|---|---|---:|
| Fasting blood glucose | Sugar measured in a fasting laboratory blood sample | 50–300 mg/dL |
| BMI | Weight compared with squared height; a broad risk clue | 12–60 |
| SVM score | Signed map position relative to the center road | Any decimal; not a probability |
| Output | Which side or road area contains the dot | Lower, follow-up margin, or higher signal |

The outside training rows have two labels. The middle result is deliberately made
from the learned SVM margin; it is not a third learned medical class.

### Step-by-step magic

1. Python loads the two numeric columns and outside-group labels from the CSV.
2. Labels become signs: lower group is `-1`, higher group is `+1`.
3. The mean of each feature is calculated—its balance point.
4. The usual spread of each feature is calculated.
5. Each value becomes `(value - mean) / spread`. This gives glucose and BMI
   comparable “map steps” even though their original rulers are different.
6. Both weights and the bias start at zero.
7. For 6,000 practice rounds, every dot receives a score.
8. If `label × score` is below `1`, that dot is wrong or inside the road and adds
   a corrective push.
9. Regularization adds a small pull toward simpler, smaller weights.
10. The learning rate makes the push smaller over time so the road settles.
11. Python exports the learned means, scales, weights, bias, and edge scores.
12. Browser JavaScript standardizes a new glucose/BMI pair with those same rulers.
13. It calculates the signed score and selects the lower side (`≤ -1`), road
    margin (`-1 to +1`), or higher side (`≥ +1`).
14. `app.js` shows a colored result and plots the point, road, support examples,
    and separate clinical glucose reference lines.

### Key code breakdown

```python
variance = self._mean([(value - mean) ** 2 for value in column])
self.scales.append(max(math.sqrt(variance), 1e-9))
```

Each number's gap from the balance point is squared so left and right gaps both
count positively. Averaging the squares measures spread; the square root turns
that spread back into normal ruler units. `1e-9` is a microscopic safety ruler
that prevents division by zero.

```python
score = weight_glucose * scaled_glucose + weight_bmi * scaled_bmi + bias
```

Each scaled map step is multiplied by a steering strength. Adding the two pushes
sets the road's tilt, while the bias slides the road without changing the tilt.
The result is a signed position, not “72% likely.”

```python
if label * score < 1:
    weight_gradient[0] -= label * features[0] / count
    weight_gradient[1] -= label * features[1] / count
    bias_gradient -= label / count
```

Multiplying the team sign by the score checks whether a dot is safely beyond its
correct edge. A result below 1 means it should push the road during this round.

```javascript
if (score <= model.margin.lower) return model.zones.lower;
if (score >= model.margin.upper) return model.zones.higher;
return model.zones.margin;
```

This is the final map lookup: left of the road, right of the road, or standing
inside it.

### Safety boundary

The model is trained on synthetic educational examples and cannot diagnose or
exclude diabetes. BMI is not a blood test, an SVM score is not a probability,
and the page's learned line must not replace clinical testing or advice.

## 3. Simple Web Development Guide

### Folder and file map

```text
svm_diabetes_triage/
├── data/
│   └── diabetes_triage.csv    Synthetic outside-group training dots
├── dist/
│   ├── index.html             Inputs, presets, result card, graph, and warning
│   ├── styles.css             Responsive three-zone visual design
│   ├── app.js                 Validation and SVG chart controller
│   ├── svm.js                 Browser scaling, scoring, and zone engine
│   └── model.json             Exported SVM parameters and graph points
├── tests/
│   ├── test_svm.py            Python training/export checks
│   └── test_browser_model.js  Browser math and interface checks
├── svm_model.py               From-scratch linear soft-margin SVM
├── train_model.py             Training/export entry point
├── start_app.py               Local static-file server
├── README.md                  Setup and safety notes
└── REPORT.md                  This guide
```

### How it works together

1. The form accepts fasting glucose and BMI. Helper text explains units and
   meaning; range attributes and inline warnings reject impossible demo values.
2. `app.js` fetches `model.json`. On submit, it calls `SimpleSvm.predict()` in
   `svm.js`, entirely inside the browser.
3. The result updates the zone badge, score, reference message, summary, and SVG
   plot. Two quick presets travel through the same validated submit handler.

### Key functions

| Function | Role |
|---|---|
| `loadModel()` | Loads the exported learned parameters. |
| `validateInput()` | Shows friendly warnings outside the UI ranges. |
| `decisionScore()` | Standardizes both inputs and calculates signed position. |
| `renderPrediction()` | Updates the badge, explanation, and graph. |

## 4. Kid-Friendly User Manual

### Quick summary

Enter a fasting laboratory glucose value and BMI, then press **Screen and plot**.
The page places the dot beside the wide SVM road and names its teaching zone.

### Three steps

1. **Type numbers** — use the two boxes or choose a lower/higher sample.
2. **Click the button** — press **Screen and plot**.
3. **Read the badge** — read the large zone name and the short explanation under it.

### Understanding the result

| Badge/color | Plain meaning in this lesson |
|---|---|
| Blue — Lower screening signal | The dot sits beyond the lower SVM road edge. |
| Orange — Further diagnostic needed | The dot sits inside the SVM margin, the wide road. |
| Red — Higher screening signal | The dot sits beyond the higher SVM road edge. |
| Green diamond | Your current glucose/BMI point. |
| Dark-ringed dot | A support example close to a road edge. |

These are teaching locations on a learned graph, not medical diagnoses. A
qualified clinician should interpret real test results.
