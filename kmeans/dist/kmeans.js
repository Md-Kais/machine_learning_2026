/* K-means learning and prediction. No dataset or UI is embedded here. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.KMeans = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const squaredDistance = (a, b) => a.reduce((sum, value, j) => sum + (value - b[j]) ** 2, 0);
  const copy = centers => centers.map(center => center.slice());
  const same = (a, b) => a.length === b.length && a.every((value, i) => value === b[i]);

  function randomGenerator(seed) {
    let state = seed >>> 0;
    return () => {
      state = (state + 0x6D2B79F5) >>> 0;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function validatePoints(points) {
    if (!Array.isArray(points) || !points.length) throw Error("Dataset must contain points.");
    const dimensions = points[0]?.length;
    if (!Number.isInteger(dimensions) || dimensions < 1) throw Error("Each point needs numeric features.");
    if (points.some(point => !Array.isArray(point) || point.length !== dimensions || point.some(v => !Number.isFinite(v)))) {
      throw Error("Every point must have the same number of finite numeric features.");
    }
    return dimensions;
  }

  function scaling(points, standardize) {
    const dimensions = points[0].length;
    const means = Array.from({ length: dimensions }, (_, j) => standardize ? points.reduce((sum, p) => sum + p[j], 0) / points.length : 0);
    const scales = means.map((mean, j) => standardize ? Math.sqrt(points.reduce((sum, p) => sum + (p[j] - mean) ** 2, 0) / points.length) || 1 : 1);
    if ([...means, ...scales].some(v => !Number.isFinite(v))) throw Error("Feature magnitudes are too large. Use smaller numeric units.");
    return { means, scales, standardize };
  }

  function transform(point, scaler) {
    return point.map((value, j) => (value - scaler.means[j]) / scaler.scales[j]);
  }

  function untransform(point, scaler) {
    return point.map((value, j) => value * scaler.scales[j] + scaler.means[j]);
  }

  function nearest(point, centers) {
    let cluster = 0, distance = Infinity;
    centers.forEach((center, index) => {
      const value = squaredDistance(point, center);
      if (value < distance) { distance = value; cluster = index; }
    });
    return cluster; // Equal distances consistently choose the lower cluster index.
  }

  function initialize(points, k, random) {
    // K-means++: sample one row, then sample proportional to squared distance
    // from the closest selected row. These are guesses, not learned means.
    const centers = [points[Math.floor(random() * points.length)].slice()];
    while (centers.length < k) {
      const weights = points.map(point => Math.min(...centers.map(center => squaredDistance(point, center))));
      const total = weights.reduce((sum, value) => sum + value, 0);
      let target = random() * total;
      let chosen = weights.findIndex(weight => { target -= weight; return weight > 0 && target < 0; });
      if (chosen < 0) chosen = weights.findIndex(weight => weight > 0);
      if (chosen < 0) throw Error("Not enough distinct points for this cluster count.");
      centers.push(points[chosen].slice());
    }
    return centers;
  }

  function countsFor(labels, k) {
    const counts = Array(k).fill(0);
    labels.forEach(label => counts[label]++);
    return counts;
  }

  function inertiaFor(points, labels, centers) {
    return points.reduce((sum, point, i) => sum + squaredDistance(point, centers[labels[i]]), 0);
  }

  function singleRun(points, options, seed, scaler) {
    let centers = initialize(points, options.k, randomGenerator(seed));
    let converged = false, iterations = 0;
    const history = [];
    function frame(phase, labels, extra = {}) {
      history.push({ phase, iteration: iterations, centers: copy(centers),
        rawCenters: centers.map(c => untransform(c, scaler)), labels: labels.slice(),
        counts: labels[0] === -1 ? Array(options.k).fill(0) : countsFor(labels, options.k),
        inertia: labels[0] === -1 ? null : inertiaFor(points, labels, centers), ...extra });
    }
    frame("initialize", Array(points.length).fill(-1), { note: "K-means++ chooses starting rows from the dataset." });
    for (let iteration = 1; iteration <= options.maxIterations; iteration++) {
      iterations = iteration;
      const labels = points.map(point => nearest(point, centers));
      frame("assign", labels, { note: "Every row joins its nearest current center." });
      const counts = countsFor(labels, options.k), repaired = [];
      for (let cluster = 0; cluster < options.k; cluster++) {
        if (counts[cluster]) continue;
        // Give an empty cluster the worst represented point from a cluster
        // with at least two members. The next mean update learns all centers.
        let candidate = -1, largest = -1;
        points.forEach((point, index) => {
          const distance = squaredDistance(point, centers[labels[index]]);
          if (counts[labels[index]] > 1 && distance > largest) { candidate = index; largest = distance; }
        });
        if (candidate < 0) throw Error("Unable to form nonempty clusters.");
        counts[labels[candidate]]--;
        labels[candidate] = cluster;
        counts[cluster]++;
        repaired.push({ cluster, pointIndex: candidate });
      }
      const next = Array.from({ length: options.k }, () => Array(points[0].length).fill(0));
      points.forEach((point, index) => point.forEach((value, j) => next[labels[index]][j] += value));
      next.forEach((center, index) => center.forEach((value, j) => center[j] = value / counts[index]));
      const shift = Math.sqrt(Math.max(...centers.map((center, i) => squaredDistance(center, next[i]))));
      centers = next;
      frame("update", labels, { shift, repaired, note: "Each center moves to the mean of its assigned rows." });
      const nextLabels = points.map(point => nearest(point, centers));
      if (shift <= options.tolerance && same(nextLabels, labels)) { converged = true; break; }
    }
    const labels = points.map(point => nearest(point, centers));
    frame("final", labels, { note: converged ? "Centers stopped moving within tolerance; assignments are stable." : "Iteration limit reached. Increase it for a closer fit." });
    return { centers: copy(centers), rawCenters: centers.map(center => untransform(center, scaler)),
      labels, counts: countsFor(labels, options.k), inertia: inertiaFor(points, labels, centers),
      converged, iterations, history, seed };
  }

  function fit(points, settings = {}) {
    validatePoints(points);
    const options = { k: 3, maxIterations: 100, tolerance: 0.0001, nInit: 5, seed: 42, standardize: true, ...settings };
    for (const key of ["k", "maxIterations", "nInit"]) {
      if (!Number.isInteger(options[key]) || options[key] < 1) throw Error(key + " must be a positive integer.");
    }
    if (!Number.isInteger(options.seed) || options.seed < 0 || options.seed > 4294967295) throw Error("Seed must be an integer from 0 to 4294967295.");
    if (!Number.isFinite(options.tolerance) || options.tolerance < 0) throw Error("Tolerance must be finite and nonnegative.");
    if (typeof options.standardize !== "boolean") throw Error("standardize must be true or false.");
    const distinctCount = new Set(points.map(point => JSON.stringify(point))).size;
    if (options.k > distinctCount) throw Error("Choose at most " + distinctCount + " clusters: that is the number of distinct rows.");
    const scaler = scaling(points, options.standardize);
    const transformed = points.map(point => transform(point, scaler));
    let best, bestRun = 0;
    const runs = [];
    for (let i = 0; i < options.nInit; i++) {
      const result = singleRun(transformed, options, (options.seed + i) >>> 0, scaler);
      runs.push({ run: i + 1, seed: result.seed, inertia: result.inertia, iterations: result.iterations, converged: result.converged });
      if (!best || result.inertia < best.inertia) { best = result; bestRun = i + 1; }
    }
    return { ...best, scaler, transformed, options, runs, bestRun, sampleCount: points.length, distinctCount };
  }

  function predict(point, model) {
    if (!model?.centers?.length || !model.scaler) throw Error("Train a model before predicting.");
    if (!Array.isArray(point) || point.length !== model.scaler.means.length || point.some(v => !Number.isFinite(v))) throw Error("Enter one finite number for each feature.");
    const z = transform(point, model.scaler);
    const distances = model.centers.map(center => Math.sqrt(squaredDistance(z, center)));
    const cluster = nearest(z, model.centers);
    return { cluster, distances, transformed: z, distance: distances[cluster] };
  }

  return { fit, predict, transform, squaredDistance };
});
