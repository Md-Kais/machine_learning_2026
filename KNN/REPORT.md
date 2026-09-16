# BMI Neighbor — Project Report

## 1. Problem Title & Purpose

### Title

**Sorting an Adult BMI by Asking the Five Nearest Examples**

### Architecture checklist

**Type A: Client-Side Browser ML.** Python reads the teaching data and exports
`dist/model.json`. The browser loads that file, and `dist/knn.js` performs the
BMI calculation, distance measurement, and neighbor vote without calling an API.

### Why it matters

Height and weight can be combined into a quick adult screening measurement called
BMI. This little app helps a learner see how a computer can compare one person
with familiar examples. It is a programming lesson, not a medical diagnosis;
children need age- and sex-specific growth charts, and adults need more than BMI
for a complete health picture.

## 2. How the Algorithm Works (10-Year-Old Explanation)

### The story metaphor

Imagine a long strip of graph paper with labeled toy people standing on it. Each
person stands at their BMI number and wears one of four team shirts:
Underweight, Normal, Overweight, or Obese.

A new person arrives. The computer uses a ruler to measure the gap from the new
person to every toy person. It invites the closest five to a tiny election. Each
neighbor votes for the team printed on its shirt, and the team with the most
votes becomes the answer. This is **K-Nearest Neighbors**, or **KNN**:

- **K** is how many neighbors may vote. Here, K is 5.
- **Nearest** means the smallest ruler gap between BMI numbers.
- **Neighbors** are the saved examples closest to the new BMI.

### Inputs and outputs

| Item | What it means in everyday life | Number used by the app |
|---|---|---:|
| Height | How tall the adult is, like measuring against a wall | 100–250 cm |
| Weight | How heavy the adult is on a scale | 25–350 kg |
| BMI | Weight compared with the square of height | Calculated by the app |
| Output | The shirt worn by most of the five nearest examples | One of four category words |

The result also includes the five neighbors, their measurements, BMI values, and
ruler gaps. Those details let a learner inspect the vote instead of trusting a
mystery answer.

### Step-by-step magic

1. Python opens `data/bmi_data.csv` and checks every row.
2. It changes height from centimetres to metres.
3. It calculates each example's BMI.
4. Every fifth row is hidden for a small practice test.
5. KNN remembers the remaining study rows. That memory is its training.
6. Each hidden row is predicted, and correct answers are counted.
7. The final KNN remembers every row and exports them with `k = 5` to
   `dist/model.json`.
8. The browser reads the user's height and weight and calculates a new BMI.
9. JavaScript measures the absolute BMI gap to every exported example.
10. It sorts the gaps from smallest to largest and keeps five.
11. The five labels vote. Most votes wins; a tied vote goes to the team with the
    smaller total gap.
12. The page shows a plain-English, color-coded card and draws the five neighbors.

### Key code breakdown

```python
height_m = height_cm / 100
bmi = weight_kg / (height_m * height_m)
```

Centimetres are too small for the BMI recipe, so `170` becomes `1.70` metres.
Multiplying `1.70 × 1.70` makes a square, like a floor tile with two equal sides.
For 65 kg, the answer is `65 / 2.89`, or about `22.49`.

```python
distances = [(abs(example.bmi - bmi), example) for example in self.examples]
distances.sort(key=lambda item: item[0])
neighbors = distances[:5]
```

`abs` is a ruler that ignores direction. The gap between 22 and 25 is 3 whether
we walk left or right. Sorting puts the shortest gaps first, and `[:5]` invites
only the first five dots.

```javascript
votes[neighbor.category].count += 1;
votes[neighbor.category].totalDistance += neighbor.distance;
```

The first line drops one ballot into the neighbor's team box. The second line
adds that neighbor's ruler gap. The gap is used only if two team boxes hold the
same number of ballots.

```javascript
const category = Object.keys(votes).sort((first, second) => {
  const voteDifference = votes[second].count - votes[first].count;
  return voteDifference || votes[first].totalDistance - votes[second].totalDistance;
})[0];
```

The computer places the team with more ballots first. If `voteDifference` is
zero, it places the team standing closer first. Position zero is the winner.

### Important limits

- This model uses only BMI, so it cannot see muscle, pregnancy, medicines,
  growth, illness, or medical history.
- Its examples are a small teaching dataset, not a population study.
- BMI categories are screening descriptions, not diagnoses or treatment advice.

## 3. Simple Web Development Guide

### Folder and file map

```text
KNN/
├── data/
│   └── bmi_data.csv          Labeled height, weight, and category examples
├── dist/
│   ├── index.html            Inputs, quick presets, result card, and chart
│   ├── styles.css            Responsive layout and category colors
│   ├── app.js                Validation, model loading, and screen updates
│   ├── knn.js                Browser BMI, distance, and voting engine
│   ├── model.json            Exported KNN memory read by the browser
│   ├── sw.js                 Saves required files for offline reopening
│   └── manifest.webmanifest Installable-app name, colors, and start page
├── tests/
│   ├── test_knn.py           Python model checks
│   └── test_browser_model.js Browser-engine checks
├── knn_model.py              From-scratch Python KNN implementation
├── train_model.py            Test, final training, and JSON export
├── start_app.py              Small local static-file server
├── vercel.json               Publishes dist and safely refreshes the worker
├── README.md                 Setup notes
└── REPORT.md                 This learning guide
```

### How it works together

1. `index.html` provides height and weight boxes. Each box shows its unit, allowed
   range, and an error area. Two preset buttons fill realistic examples.
2. `app.js` downloads `model.json` with `fetch()`. On submit, it validates both
   values and calls `BmiKnn.calculateBmi()` and `BmiKnn.predictWithKnn()`.
3. `knn.js` returns the winning category and neighbors. `app.js` writes them into
   the result card, moves the BMI marker, and redraws the SVG neighbor map.
4. On localhost or Vercel, `sw.js` caches the local page and model files for
   later offline reopening.

No health measurements are sent to a prediction server. Once the static files
are loaded, prediction happens inside the browser tab.

### Key functions

| Function | Role |
|---|---|
| `loadModel()` | Fetches `model.json` and enables the form. |
| `validateInput()` | Shows a friendly warning for blank or out-of-range values. |
| `calculateBmi()` | Changes height and weight into one BMI number. |
| `predictWithKnn()` | Measures gaps, chooses five neighbors, and counts votes. |

## 4. Kid-Friendly User Manual

### Quick summary

Enter an adult's height and weight, then press **Check with KNN**. The page shows
the BMI team selected by five nearby examples and draws those voters on a map.

### Three steps

1. **Type numbers** — enter height in centimetres and weight in kilograms, or
   choose one of the two sample buttons.
2. **Click the button** — press **Check with KNN**.
3. **Read the badge** — look at the category word, BMI number, explanation, and
   five numbered neighbors.

### Understanding the result

| Badge/color | Plain meaning |
|---|---|
| Blue — Underweight | Below the usual adult BMI screening range. |
| Green — Normal | Inside the usual adult BMI screening range. |
| Amber — Overweight | Above the usual adult BMI screening range. |
| Red — Obese | Inside the adult obesity screening range. |

The colors help you read the lesson quickly; they do not declare that a person
is healthy or unhealthy. Ask a qualified health professional for personal advice.
