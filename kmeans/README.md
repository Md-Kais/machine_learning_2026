# K-means Dataset Lab

This replaces the old customer example with **CSV-based training and a user-selected cluster count**. There are no embedded training points, fixed starting centers, saved learned centers, or target class labels in the app. A separate algorithm module learns the centers from whichever dataset you select.

## Run

Node.js 18 or newer is sufficient. No packages need to be installed.

```powershell
cd D:\machine_learning_2026\kmeans
npm start
```

Open **http://localhost:9123** (the address printed by the server). Keep the terminal open; Ctrl+C stops it. Use the server instead of double-clicking index.html because CSV loading and the training worker need an HTTP origin.

To choose a different port:

```powershell
$env:PORT = "8080"
npm start
```

## Use the interface

1. Choose one of the three datasets, or load your own CSV.
2. Enter **K**, the number of clusters you want.
3. Select **Learn cluster centers**. The app reads the rows and trains a fresh model.
4. Inspect the final centers, member counts, assignments and inertia.
5. Select **Replay learning**, or use the step slider, to watch initialization, assignments and mean updates.
6. Enter a new point and select **Find nearest learned center**, or click a dataset point to inspect it.
7. Compare its distance to every center. The smallest distance determines its cluster.

Use **Save SVG**, **Download centers CSV**, and **Download results CSV** to save the displayed graph and learned output. A downloaded results CSV includes every original row, its cluster number and its distance to the assigned center.

Changing a dataset invalidates the old model. Changing training parameters requires a new training run; until then, the status explicitly reports that the displayed result uses the previous K.

## The three datasets

All are synthetic educational data with two numeric features and an id. They contain **rows only**, with no cluster labels or stored centers.

| File | Rows | Features |
| --- | ---: | --- |
| [dist/data/dataset1.csv](dist/data/dataset1.csv) | 72 | annual_income_k, spending_score |
| [dist/data/dataset2.csv](dist/data/dataset2.csv) | 90 | study_hours_week, assessment_score |
| [dist/data/dataset3.csv](dist/data/dataset3.csv) | 84 | temperature_c, relative_humidity_pct |

Edit the CSV files to change the data. Refresh the page, select the dataset and retrain. You can also upload a CSV directly; it stays in your browser and is not sent to the server.

A custom CSV needs exactly two numeric feature columns, optionally preceded by a unique **id** column. Headers become the input and graph labels:

```csv
id,x,y
row1,1,2
row2,2,3
row3,10,12
row4,11,13
```

Values must be finite numbers. Negative and zero values are valid. Missing numbers are rejected rather than converted to zero. The interactive importer supports quoted fields, scientific notation, up to 1000 rows, and files smaller than 1 MB.

## Where the learning happens

**dist/kmeans.js** is the independent learning algorithm. **dist/train-worker.js** runs it in a Web Worker so the form stays responsive; **dist/app.js** controls the interface and graph. **dist/data.js** parses CSV files.

The algorithm performs Lloyd's K-means iterations:

1. Calculate scaling statistics from the selected CSV, if scaling is enabled.
2. Initialize K centers with **K-means++**, which samples rows from the dataset. These are starting guesses, not final means.
3. Assign each row to its nearest center.
4. Replace each center with the feature-wise mean of its assigned rows.
5. Repeat until center movement is within tolerance and assignments are stable, or the iteration cap is reached.
6. Repeat from several seeded initializations and keep the lowest-inertia run.

**This mean update is learning from the dataset.** For example, training on (0,0), (0,2), (10,10), (10,12) with K=2 and scaling off learns centers (0,1) and (10,11). Shifting every row shifts those learned centers too; the centers are not predefined.

Inertia is the sum of squared distances from each training point to its assigned center. It is a clustering objective, not classification accuracy. Cluster numbers are arbitrary group identifiers and may change between different seeds or datasets. K-means does not learn named categories and does not automatically choose K. Multiple initializations reduce sensitivity to starting points but do not guarantee a global optimum.

## Change parameters

All settings are available in the interface:

| Setting | Default | Effect |
| --- | ---: | --- |
| Number of clusters K | 3 | Any integer from 1 to the number of distinct feature pairs. More clusters split the data into smaller groups. |
| Scale both features equally | on | Learns each feature's mean and population standard deviation, then uses z=(x-mean)/scale. Constant features use scale 1. Turn off to use raw Euclidean distances. |
| Random seed | 42 | Makes initialization reproducible; changing it can produce a different local solution. Integer 0 through 4294967295. |
| Starting attempts | 5 | Try multiple initializations and retain the lowest final inertia. UI range 1-10. |
| Maximum iterations | 100 | Cap on assignment/update rounds per attempt. UI range 1-200. Increase if the solver reports an iteration limit. |
| Center movement tolerance | 0.0001 | Maximum allowed movement, measured in the training feature space. Smaller tolerances require more precise stabilization. Zero is valid. |

For an empty cluster, the algorithm moves the worst represented point from a cluster with at least two members into it, then recalculates the means. The replay reports this repair. Equal-distance assignments choose the lower cluster number. K cannot exceed the number of distinct feature pairs, even when the CSV contains duplicate rows.

A capped run is explicitly marked as approximate. Its final labels still use the nearest returned centers, but the centers may not yet equal the means of those final labels. Increase the iteration limit and retrain before treating it as converged.

The separate module also works directly in Node:

```javascript
const fs = require("node:fs");
const { parseCSV } = require("./dist/data.js");
const { fit, predict } = require("./dist/kmeans.js");

const dataset = parseCSV(fs.readFileSync("./dist/data/dataset1.csv", "utf8"));
const model = fit(dataset.records.map(row => row.values), {
  k: 4,
  seed: 7,
  nInit: 5,
  maxIterations: 100,
  tolerance: 0.0001,
  standardize: true
});
console.log(model.rawCenters); // centers in the original CSV units
console.log(predict([50, 60], model)); // nearest cluster and every distance
```

## Read the visualization and explanation

- Colored dots are training rows; crosses are centers. Hover a point or center to inspect its values.
- The faint background shows the nearest-center regions for the currently displayed centers.
- The gold diamond marks the selected point. At the final step, dashed lines compare it with every learned center; the winning line is emphasized.
- The graph uses standardized coordinates when scaling is on, with equal axis units. Center and assignment tables always retain original CSV units.
- During an update frame, colors retain the preceding assignment while centers move to their means. The next assignment frame recalculates membership.
- The decision table and equation use the **final trained model**. Selecting a point returns the graph to the final result.
- The mean explanation shows the selected cluster's feature sums divided by its member count, so you can see where its center came from.

## Files and checks

| File | Role |
| --- | --- |
| dist/kmeans.js | Standalone training, scaling, initialization, history and prediction |
| dist/data.js | CSV parsing, numeric validation and export escaping |
| dist/train-worker.js | Background training |
| dist/app.js | CSV loading, forms, replay, explanations, SVG drawing and downloads |
| dist/data/dataset1.csv through dataset3.csv | Independent training datasets |
| server.js | Local static server |
| tests/kmeans.test.js | Learning, means, different K values, scaling, convergence and CSV checks |
| tests/ui.test.js | CSV-to-worker-to-interface integration, replay, uploads, cancellation and exports |

```powershell
npm test
```

Tests exercise the actual worker script and interface events with a small DOM test harness. They do not substitute for visual browser layout testing.

The service worker caches the app and CSV files for later offline visits. Network-first loading picks up CSV and code edits, and the obsolete customer-example cache is retired. The existing vercel.json can still serve dist as a static app; no build step or backend ML service is needed.
