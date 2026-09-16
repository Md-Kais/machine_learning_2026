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


# How KMEANS work? 

 ## Main answer

  The actual K-means algorithm is implemented in dist/app.js:59, especially:

  - normalizedDistance() at line 59
  - runKMeans() at line 71
  - Training invocation at line 129
  - New-customer classification at line 197

  This is a fully client-side JavaScript application. server.js only serves static files—it does not train or execute the model.

  ## Project structure

  kmeans/
  ├── dist/
  │   ├── index.html              Page structure, form, buttons, canvas
  │   ├── styles.css              Layout, colors, responsive design
  │   ├── app.js                  Dataset, K-means, classification, chart
  │   ├── sw.js                   Offline caching
  │   └── manifest.webmanifest   Installable PWA configuration
  ├── server.js                   Local static-file web server
  ├── package.json                npm start command; no dependencies
  ├── vercel.json                 Vercel deployment configuration
  ├── README.md                   Running and deployment instructions
  └── REPORT.md                   Project and algorithm documentation

  Despite the directory being named dist, there is no separate src directory or build process. These are the directly executed application files.

  ### Runtime flow

  npm start
     ↓
  server.js serves dist/
     ↓
  index.html loads app.js
     ↓
  runKMeans(customers) trains once
     ↓
  Final centroids are stored in model
     ↓
  User enters age and shopping score
     ↓
  classifyCustomer() finds the nearest centroid
     ↓
  drawChart() updates the visualization

  ## How the K-means implementation works

  ### 1. Training data

  The fixed dataset is defined at dist/app.js:15.

  const customers = [
    [19, 18],
    [22, 27],
    // ...
    [68, 89]
  ].map(([age, spend]) => ({ age, spend }));

  There are 27 pretend customers. Each customer has two features:

  - age
  - spend, meaning shopping score

  There are no target labels because K-means is unsupervised.

  ### 2. Distance calculation

  dist/app.js:59 calculates normalized Euclidean distance:

  const ageGap = (point.age - center.age) / 52;
  const spendGap = (point.spend - center.spend) / 99;

  return Math.sqrt(
    ageGap * ageGap +
    spendGap * spendGap
  );

  Mathematically:

  [
  d(p,c)=\sqrt{
  \left(\frac{p_{age}-c_{age}}{52}\right)^2+
  \left(\frac{p_{spend}-c_{spend}}{99}\right)^2
  }
  ]

  The divisions normalize the feature ranges:

  - Age range: 70 - 18 = 52
  - Shopping-score range: 100 - 1 = 99

  Without this normalization, shopping score would have more influence simply because its numeric range is larger.

  ### 3. Starting centroids

  Inside dist/app.js:71, K is effectively fixed at 2:

  let centers = [
    { age: 25, spend: 22 },
    { age: 56, spend: 76 }
  ];

  These are manually selected initial guesses. The application does not use random initialization or K-means++.

  ### 4. Assign points to their closest centroid

  For every customer, the code measures the distance to both centers:

  const nextAssignments = points.map(point => {
    const distanceToFirst = normalizedDistance(point, centers[0]);
    const distanceToSecond = normalizedDistance(point, centers[1]);

    return distanceToFirst <= distanceToSecond ? 0 : 1;
  });

  The point receives cluster 0 or 1. An exact tie goes to cluster 0, making execution deterministic.

  ### 5. Recalculate centroids

  For each cluster, the code collects its customers and calculates their mean age and mean shopping score:

  const team = points.filter(
    (point, index) => nextAssignments[index] === clusterIndex
  );

  return {
    age: team.reduce((sum, point) => sum + point.age, 0) / team.length,
    spend: team.reduce((sum, point) => sum + point.spend, 0) / team.length
  };

  Therefore:

  [
  centroid_{age}=\frac{\sum ages}{number\ of\ customers}
  ]

  [
  centroid_{spend}=\frac{\sum scores}{number\ of\ customers}
  ]

  If a cluster becomes empty, its previous center is retained.

  ### 6. Stop when assignments no longer change

  const nobodyMoved = nextAssignments.every(
    (team, index) => team === assignments[index]
  );

  if (nobodyMoved) break;

  There is also a maximum of 12 rounds to prevent an unlimited loop.

  For this dataset, the verified execution is:

   Round    Cluster sizes    Centroids                         Status
  ━━━━━━━  ━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━
   Start            0 / 0    (25, 22), (56, 76)                Initial guesses
  ───────  ───────────────  ────────────────────────────────  ─────────────────
   1              12 / 15    (31.67, 32.08), (49.27, 65.93)    Points assigned
  ───────  ───────────────  ────────────────────────────────  ─────────────────
   2              12 / 15    Same centroids                    Converged

  The second round confirms that no assignments changed.

  ### 7. Save the trained model

  Training runs immediately when the browser loads dist/app.js:129:

  const model = runKMeans(customers);

  The returned object contains:

  {
    centers,      // final learned centroids
    assignments,  // cluster of each training customer
    history       // snapshots used by the replay animation
  }

  ## Budget and Premium labels

  K-means itself only produces cluster numbers 0 and 1. It does not know the words “Budget” or “Premium.”

  dist/app.js:132 assigns those names afterward:

  const budgetIndex =
    centers[0].spend <= centers[1].spend ? 0 : 1;

  The centroid with the lower average shopping score becomes “Budget Shopper.” The other becomes “Premium Shopper.”

  ## Classifying a user-entered customer

  dist/app.js:197 does not rerun K-means and does not add the entered customer to the training data.

  It:

  1. Validates age and shopping score.
  2. Calculates distance to both final centroids.
  3. Chooses the nearest centroid.
  4. Converts its cluster number to Budget or Premium.
  5. Updates the result card and chart.

  const distances = model.centers.map(
    center => normalizedDistance(point, center)
  );

  winningCluster = distances[0] <= distances[1] ? 0 : 1;

  So this part is nearest-centroid prediction using the already-trained clusters.

  ## Supporting pieces

  - dist/index.html:1 defines the sliders, number inputs, result card, replay button, and canvas.
  - dist/app.js:306 draws points, centroids, decision regions, and the user’s customer.
  - dist/app.js:248 plays the saved model.history; it does not retrain.
  - server.js:1 is only a dependency-free static server with path-traversal protection.
  - dist/sw.js:1 caches the app for offline use.

  In short: this is a standard two-cluster Lloyd-style K-means loop with fixed initialization, normalized Euclidean distance, mean-based centroid updates, and assignment-based convergence.