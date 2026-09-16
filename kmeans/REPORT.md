# Customer Team Finder — Project Report

## 1. Problem Title & Purpose

### Title

**Grouping Pretend Customers into Two Nearby Shopping Teams**

### Architecture checklist

**Type A: Client-Side Browser ML, with a small variation.** There is no Python
training script or exported `model.json`; the tiny fixed dataset and complete
K-Means training loop live in `dist/app.js`. `server.js` only serves static files
and never runs inference.

### Why it matters

Shops often want to understand broad patterns in unlabeled customer data. K-Means
shows one simple way to discover groups by closeness instead of starting with
answer labels. This demo uses pretend data and playful team names; it must not be
used to judge, price, include, or exclude a real person.

## 2. How the Algorithm Works (10-Year-Old Explanation)

### The story metaphor

Imagine customer marbles scattered on graph paper. Moving right means an older
age; moving up means a higher shopping score. We place two empty bowls on the
paper because **K = 2**.

K-Means plays three jobs:

1. **Pick** two starting bowl positions.
2. **Put** every marble with the nearest bowl.
3. **Push** each bowl to the average, or balance point, of its marbles.

Then it repeats Put and Push. When no marble changes bowls, the teams are stable
and training stops. The bowl positions are called **centroids**, which simply
means “team middles.”

### Inputs and outputs

| Item | Physical meaning | App range |
|---|---|---:|
| Age | Years lived by the pretend customer; graph x-position | 18–70 |
| Shopping score | Pretend interest in shopping; graph y-position | 1–100 |
| K | Number of bowls/teams | Fixed at 2 |
| Output | The closest learned team middle | Budget Shopper or Premium Shopper |

Unlike a classifier, K-Means receives no “correct team” column. It makes groups
from the positions of the dots. The human-friendly names are attached afterward:
the centroid with the lower shopping score is called Budget, and the other is
called Premium.

### Step-by-step magic

1. JavaScript turns every `[age, score]` pair into a named point object.
2. Two starting centers are picked at `(25, 22)` and `(56, 76)`.
3. Age and shopping gaps are divided by their full ranges. That prevents the
   1–100 score ruler from overpowering the shorter 18–70 age ruler.
4. For each dot, the computer measures straight-line distance to both centers.
5. Team 0 or team 1 is chosen using the smaller distance.
6. All ages in one team are added and divided by the number of dots.
7. All shopping scores in that team are averaged the same way.
8. Those two averages become the team's new center.
9. Assignment and center snapshots are saved for the Replay animation.
10. Steps 4–8 repeat until every dot keeps its previous team, or 12 safety rounds
    have run.
11. A new user-created customer is measured against the two final centers.
12. The closer center wins, and the page shows a red/blue plain-English card.

### Key code breakdown

```javascript
const ageGap = (point.age - center.age) / 52;
const spendGap = (point.spend - center.spend) / 99;
```

Age travels 52 steps from 18 to 70; score travels 99 steps from 1 to 100.
Dividing by those spans is like printing both rulers at the same size. Each
feature then gets a fair voice.

```javascript
return Math.sqrt((ageGap * ageGap) + (spendGap * spendGap));
```

This is a straight diagonal ruler between two graph-paper dots. It uses the same
idea as finding a ladder length from its sideways and upward gaps: square both,
add them, and take the square root. This is Euclidean distance.

```javascript
const nextAssignments = points.map(point => {
  const first = normalizedDistance(point, centers[0]);
  const second = normalizedDistance(point, centers[1]);
  return first <= second ? 0 : 1;
});
```

Every marble checks both bowl distances. The smaller one selects team 0 or team
1. An exact tie always picks team 0 so the result is repeatable.

```javascript
age: team.reduce((sum, point) => sum + point.age, 0) / team.length,
spend: team.reduce((sum, point) => sum + point.spend, 0) / team.length
```

Add all team values, then divide by the number of team members. That is the mean,
or the point where the team's number-seesaw balances.

```javascript
const nobodyMoved = nextAssignments.every(
  (team, index) => team === assignments[index]
);
if (nobodyMoved) break;
```

If every marble chose the same bowl as last time, another round would change
nothing. `break` ends the practice loop early.

### Limits

- Different starting centers can sometimes lead to different final groups.
- K is fixed at two even if another number of groups could fit different data.
- Age and shopping score are not enough to understand a real customer.
- The labels “Budget” and “Premium” are teaching names, not facts about a person.

## 3. Simple Web Development Guide

### Folder and file map

```text
kmeans/
├── dist/
│   ├── index.html          Inputs, presets, result card, chart, and lesson
│   ├── styles.css          Responsive playful layout and team colors
│   ├── app.js              K-Means training, validation, prediction, and canvas
│   ├── sw.js               Saves required files for offline reopening
│   └── manifest.webmanifest Installable-app name, colors, and start page
├── server.js               Safe local static-file server
├── package.json            Node start command; no packages required
├── vercel.json             Publishes dist and safely refreshes the worker
├── README.md               Local setup instructions
└── REPORT.md               This guide
```

### How it works together

1. `index.html` collects age and shopping score through linked sliders and number
   boxes. Helper text, inline errors, and two sample buttons make the ranges clear.
2. When `app.js` loads, `runKMeans(customers)` trains the two centroids in memory.
3. Submitting the form validates both numbers, measures the new point to the two
   centers, updates the result card, and redraws the Canvas chart.
4. `server.js` only returns files from `dist`; it does not receive customer input.
5. On localhost or Vercel, `sw.js` caches every public file for later offline
   reopening.

### Key functions

| Function | Role |
|---|---|
| `runKMeans()` | Repeats assignment and centroid movement until stable. |
| `normalizedDistance()` | Measures a fair two-feature ruler gap. |
| `validateNumberInput()` | Shows friendly blank/out-of-range warnings. |
| `classifyCustomer()` | Chooses the nearest final center and updates the UI. |

## 4. Kid-Friendly User Manual

### Quick summary

Choose an age and shopping score, then press **Find the customer’s team**. The
page places the yellow dot on the map and shows which team middle is closer.

### Three steps

1. **Type numbers** — use age 18–70 and score 1–100, move the sliders, or choose
   one of the budget/premium samples.
2. **Click the button** — press **Find the customer’s team**.
3. **Read the badge** — the red or blue result card names the closest team and
   compares both distances.

### Understanding the result

| Badge/color | Plain meaning |
|---|---|
| Red — Budget Shopper | The yellow dot is closer to the lower-score team middle. |
| Blue — Premium Shopper | The yellow dot is closer to the higher-score team middle. |
| Yellow dot | The customer values entered in the controls. |
| Red/blue star | A learned team middle, or centroid. |
| Solid connecting line | The winning shorter distance. |
| Dashed connecting line | The longer distance to the other team. |

Press **Replay K-means** to watch the stars move toward their team averages. The
pretend result is an algorithm lesson, not a judgment about a real shopper.
