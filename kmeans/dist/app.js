/* Interface and SVG rendering. Learning lives in kmeans.js; CSV parsing in data.js. */
"use strict";
const $ = id => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const definitions = {
  "dataset1.csv": { name: "Customers", description: "72 synthetic customers: annual income in thousands and a spending score. No customer groups are provided." },
  "dataset2.csv": { name: "Study habits", description: "90 synthetic study records: weekly study hours and assessment score. There are no target class labels." },
  "dataset3.csv": { name: "Weather readings", description: "84 synthetic observations: temperature in Celsius and relative humidity percentage." }
};
let dataset = null, model = null, worker = null, selection = null;
let frameIndex = 0, replayTimer = null, loadVersion = 0;
const format = (value, digits = 3) => Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
const featureTitle = feature => feature.replaceAll("_", " ");
const color = index => "hsl(" + ((index * 137.508 + 155) % 360).toFixed(1) + " 58% 40%)";

function element(tag, attrs = {}, text) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function clearTable(id, headers) {
  const table = $(id);
  const head = table.tHead || table.createTHead();
  const body = table.tBodies[0] || table.createTBody();
  head.replaceChildren(); body.replaceChildren();
  const row = head.insertRow();
  for (const title of headers) {
    const cell = document.createElement("th"); cell.textContent = title; row.append(cell);
  }
  return body;
}

function cell(row, text) { const td = row.insertCell(); td.textContent = text; return td; }
function clusterCell(row, index) {
  const td = row.insertCell(), swatch = document.createElement("span");
  swatch.className = "swatch"; swatch.style.backgroundColor = color(index);
  td.append(swatch, document.createTextNode("Cluster " + (index + 1)));
  return td;
}

function stopReplay() {
  clearInterval(replayTimer); replayTimer = null; $("play").textContent = "Replay learning";
}

function setBusy(busy) {
  for (const id of ["dataset", "upload", "k", "seed", "restarts", "max-iterations", "tolerance", "standardize"]) $(id).disabled = busy;
  $("train").disabled = busy || !dataset;
  $("cancel").hidden = !busy;
  $("train").textContent = busy ? "Learning from CSV…" : "Learn cluster centers";
}

function invalidateModel() {
  stopReplay(); if (worker) worker.terminate(); worker = null;
  model = null; selection = null; frameIndex = 0;
  for (const id of ["predict", "play", "previous", "next", "final", "step", "download-svg", "download-centers", "download-labels"]) $(id).disabled = true;
  for (const id of ["cluster-count", "iteration-count", "inertia"]) $(id).textContent = "—";
  for (const id of ["centers", "assignments", "runs", "distances"]) {
    const table = $(id); if (table.tBodies[0]) table.tBodies[0].replaceChildren();
  }
  $("prediction-result").textContent = "Train a model to classify a point.";
  for (const id of ["distance-math", "mean-explanation", "fit-note", "step-description", "scaling-note"]) $(id).textContent = "";
  $("decision-rule").textContent = "Compare the distance to every learned center. The smallest distance wins.";
  $("selected-id").textContent = "Select a point after training";
  $("step").max = "0"; $("step").value = "0";
}

function acceptDataset(parsed, name, description) {
  invalidateModel(); dataset = { ...parsed, name };
  const unique = new Set(dataset.records.map(r => JSON.stringify(r.values))).size;
  $("k").max = String(unique);
  if (Number($("k").value) > unique) $("k").value = String(Math.min(3, unique));
  $("dataset-description").textContent = description;
  $("data-summary").textContent = dataset.records.length + " rows · 2 numeric features · " + unique + " distinct points";
  $("row-count").textContent = String(dataset.records.length);
  $("feature-x-label").textContent = featureTitle(dataset.features[0]);
  $("feature-y-label").textContent = featureTitle(dataset.features[1]);
  $("query-x").value = String(dataset.records[0].values[0]);
  $("query-y").value = String(dataset.records[0].values[1]);
  $("error").textContent = "";
  $("status").textContent = "CSV loaded. Choose K and learn the centers.";
  setBusy(false); drawGraph();
}

async function loadBuiltIn() {
  const version = ++loadVersion, file = $("dataset").value;
  invalidateModel(); dataset = null; setBusy(true);
  $("cancel").hidden = true; $("status").textContent = "Loading " + file + "…";
  $("error").textContent = "";
  $("dataset-link").hidden = false; $("dataset-link").href = "data/" + file;
  $("upload").value = "";
  try {
    const response = await fetch("data/" + file, { cache: "no-cache" });
    if (!response.ok) throw Error("Cannot load CSV. Start the app with npm start.");
    const parsed = ClusterData.parseCSV(await response.text());
    if (version !== loadVersion) return;
    acceptDataset(parsed, definitions[file].name, definitions[file].description);
  } catch (error) {
    if (version !== loadVersion) return;
    $("error").textContent = error.message + " Run npm start instead of opening the HTML file directly.";
    $("status").textContent = "Dataset could not be loaded."; setBusy(false);
  }
}

function train() {
  if (!dataset) return;
  const options = {
    k: Number($("k").value), seed: Number($("seed").value), nInit: Number($("restarts").value),
    maxIterations: Number($("max-iterations").value), tolerance: Number($("tolerance").value),
    standardize: $("standardize").checked
  };
  if (options.k > Number($("k").max)) { $("error").textContent = "K cannot exceed the number of distinct points: " + $("k").max; return; }
  invalidateModel(); setBusy(true); $("error").textContent = "";
  $("status").textContent = "Learning K=" + options.k + " from " + dataset.records.length + " CSV rows. No centers are supplied.";
  try { worker = new Worker("train-worker.js"); }
  catch (error) { setBusy(false); $("error").textContent = "Training worker unavailable. Run npm start and open its local URL."; return; }
  worker.onmessage = event => {
    worker.terminate(); worker = null; setBusy(false);
    if (event.data.error) { $("error").textContent = event.data.error; $("status").textContent = "Training failed. Check your settings."; return; }
    model = event.data.model; frameIndex = model.history.length - 1;
    $("step").max = String(frameIndex); $("step").value = String(frameIndex);
    $("cluster-count").textContent = String(model.options.k);
    $("iteration-count").textContent = String(model.iterations);
    $("inertia").textContent = format(model.inertia);
    $("status").textContent = model.converged ? "Learned centers from the CSV. Assignments are stable." : "Iteration limit reached. Increase maximum iterations and retrain.";
    for (const id of ["predict", "play", "previous", "next", "final", "step", "download-svg", "download-centers", "download-labels"]) $(id).disabled = false;
    populateResults(); setFrame(frameIndex); inspectPoint(dataset.records[0].values, "Dataset row " + dataset.records[0].id);
  };
  worker.onerror = () => {
    if (worker) worker.terminate(); worker = null; setBusy(false);
    $("status").textContent = "Training could not run.";
    $("error").textContent = "Training worker failed to load. Run npm start and refresh the page.";
  };
  worker.postMessage({ points: dataset.records.map(row => row.values), options });
}

function populateResults() {
  const centerBody = clearTable("centers", ["Cluster", "Assigned rows", ...dataset.features.map(featureTitle), "Cluster inertia"]);
  model.rawCenters.forEach((center, index) => {
    const row = centerBody.insertRow(); clusterCell(row, index); cell(row, model.counts[index]);
    center.forEach(value => cell(row, format(value, 5)));
    const inertia = model.transformed.reduce((sum, point, i) => sum + (model.labels[i] === index ? KMeans.squaredDistance(point, model.centers[index]) : 0), 0);
    cell(row, format(inertia));
  });
  $("fit-note").textContent = "Trained with K=" + model.options.k + "; " + (model.options.standardize ? "standardized" : "original") + " feature distances. Attempt " + model.bestRun + " of " + model.runs.length + " has the lowest inertia. " + (model.converged ? "Centers equal the means of their final assigned rows." : "This approximate fit reached its iteration limit; centers may change with more iterations.");
  const runBody = clearTable("runs", ["Attempt", "Seed", "Inertia", "Selected"]);
  model.runs.forEach(run => { const row = runBody.insertRow(); cell(row, run.run); cell(row, run.seed); cell(row, format(run.inertia, 6)); cell(row, run.run === model.bestRun ? "Lowest inertia ✓" : ""); });
  const assignmentBody = clearTable("assignments", ["Row id", ...dataset.features.map(featureTitle), "Cluster"]);
  dataset.records.forEach((record, index) => {
    const row = assignmentBody.insertRow(), td = row.insertCell(), button = document.createElement("button");
    button.type = "button"; button.className = "row-button"; button.textContent = record.id;
    button.addEventListener("click", () => inspectPoint(record.values, "Dataset row " + record.id));
    td.append(button); record.values.forEach(value => cell(row, format(value))); clusterCell(row, model.labels[index]);
  });
}

function inspectPoint(values, source = "New input") {
  if (!model) return;
  try {
    const result = KMeans.predict(values, model);
    selection = { ...result, values: values.slice(), source };
    $("query-x").value = String(values[0]); $("query-y").value = String(values[1]);
    $("error").textContent = "";
    $("selected-id").textContent = source;
    $("prediction-result").replaceChildren(document.createTextNode(source));
    const answer = document.createElement("strong"); answer.textContent = "Cluster " + (result.cluster + 1);
    answer.style.color = color(result.cluster); $("prediction-result").append(answer);
    $("prediction-result").append(document.createTextNode("Nearest-center distance: " + format(result.distance, 5)));
    const body = clearTable("distances", ["Cluster", "Center, original units", "Distance", "Decision"]);
    result.distances.forEach((distance, index) => {
      const row = body.insertRow(); clusterCell(row, index);
      cell(row, "(" + model.rawCenters[index].map(v => format(v, 4)).join(", ") + ")");
      cell(row, format(distance, 6)); cell(row, index === result.cluster ? "Smallest distance ✓" : "");
      if (index === result.cluster) row.className = "winner";
    });
    const z = result.transformed, center = model.centers[result.cluster];
    $("decision-rule").textContent = "Cluster " + (result.cluster + 1) + " has the smallest Euclidean distance. Exact ties choose the lower cluster number.";
    $("distance-math").textContent = "d = sqrt((" + format(z[0], 5) + " − " + format(center[0], 5) + ")² + (" + format(z[1], 5) + " − " + format(center[1], 5) + ")²) = " + format(result.distance, 6) + (model.options.standardize ? " standardized units" : " original feature units");
    const members = dataset.records.filter((row, i) => model.labels[i] === result.cluster);
    const sums = [0, 1].map(j => members.reduce((sum, row) => sum + row.values[j], 0));
    $("mean-explanation").textContent = model.converged ? "How its center was learned: " + members.length + " assigned rows. " + dataset.features.map((feature, j) => featureTitle(feature) + " mean = " + format(sums[j], 4) + " / " + members.length + " = " + format(model.rawCenters[result.cluster][j], 5)).join("; ") + "." : "This run stopped at the iteration cap. Refit with more iterations before interpreting the centers as final cluster means.";
    stopReplay(); setFrame(model.history.length - 1);
  } catch (error) { $("error").textContent = error.message; }
}

function setFrame(index) {
  if (!model) return;
  frameIndex = Math.max(0, Math.min(model.history.length - 1, index)); $("step").value = String(frameIndex);
  const frame = model.history[frameIndex];
  const phase = { initialize: "Initialization", assign: "Assign rows", update: "Update means", final: "Final result" }[frame.phase];
  $("step-description").textContent = phase + (frame.iteration ? " · iteration " + frame.iteration : "") + ": " + frame.note + (frame.inertia === null ? "" : " Inertia: " + format(frame.inertia, 5) + ".") + (frame.shift === undefined ? "" : " Maximum center movement: " + format(frame.shift, 6) + ".") + (frame.repaired?.length ? " Empty clusters repaired: " + frame.repaired.length + "." : "") + (frame.phase === "update" ? " Points keep their last assignments until the next assignment step." : "");
  $("previous").disabled = frameIndex === 0; $("next").disabled = frameIndex === model.history.length - 1;
  $("scaling-note").textContent = model.options.standardize ? "Axes and distances use z = (value − mean) / standard deviation. Dataset scaling: " + dataset.features.map((f, j) => featureTitle(f) + " [mean " + format(model.scaler.means[j]) + ", scale " + format(model.scaler.scales[j]) + "]").join("; ") + ". Center tables show original units." : "Scaling is off. Axes and distances use the original feature units; features with larger numeric ranges can dominate.";
  if (frame.phase !== "final") $("scaling-note").textContent += " Point prediction below uses the final learned model; this graph shows a training step.";
  drawGraph();
}

function drawGraph() {
  const svg = $("cluster-graph"); svg.replaceChildren();
  svg.append(element("rect", { x: 0, y: 0, width: 720, height: 620, fill: "white" }));
  if (!dataset) return;
  const frame = model ? model.history[frameIndex] : { centers: [], labels: Array(dataset.records.length).fill(-1), phase: "loaded" };
  const points = model ? model.transformed : dataset.records.map(row => row.values);
  const showSelection = selection && frame.phase === "final";
  const all = [...points, ...frame.centers, ...(showSelection ? [selection.transformed] : [])];
  let lo = Math.min(...all.flat()), hi = Math.max(...all.flat());
  const pad = Math.max((hi - lo) * 0.08, 0.5); lo -= pad; hi += pad;
  const left = 90, top = 35, size = 500, span = hi - lo;
  const sx = x => left + (x - lo) / span * size, sy = y => top + size - (y - lo) / span * size;
  svg.append(element("title", {}, "K-means " + frame.phase + ": " + dataset.name));
  svg.append(element("desc", {}, "Dataset rows are colored by assignment. Crosses show centers learned from data. Select a point to inspect distances."));
  if (frame.centers.length) {
    const resolution = 24, width = span / resolution;
    for (let ix = 0; ix < resolution; ix++) for (let iy = 0; iy < resolution; iy++) {
      const p = [lo + (ix + .5) * width, lo + (iy + .5) * width];
      let winner = 0, distance = Infinity;
      frame.centers.forEach((center, index) => { const d = KMeans.squaredDistance(p, center); if (d < distance) { winner = index; distance = d; } });
      svg.append(element("rect", { x: sx(lo + ix * width), y: sy(lo + (iy + 1) * width), width: size / resolution + .4, height: size / resolution + .4, fill: color(winner), opacity: .075 }));
    }
  }
  for (let i = 0; i <= 5; i++) {
    const value = lo + span * i / 5;
    svg.append(element("line", { x1: sx(value), y1: top, x2: sx(value), y2: top + size, stroke: "#dce6e0", "stroke-width": 1 }));
    svg.append(element("line", { x1: left, y1: sy(value), x2: left + size, y2: sy(value), stroke: "#dce6e0", "stroke-width": 1 }));
    svg.append(element("text", { x: sx(value), y: top + size + 24, "text-anchor": "middle", fill: "#61726c" }, format(value, 2)));
    svg.append(element("text", { x: left - 12, y: sy(value) + 4, "text-anchor": "end", fill: "#61726c" }, format(value, 2)));
  }
  if (frame.phase === "update" && frameIndex > 0) {
    const previous = model.history[frameIndex - 1];
    frame.centers.forEach((center, index) => {
      const before = previous.centers[index];
      svg.append(element("line", { x1: sx(before[0]), y1: sy(before[1]), x2: sx(center[0]), y2: sy(center[1]), stroke: color(index), "stroke-width": 2, "stroke-dasharray": "3 3" }));
      svg.append(element("circle", { cx: sx(before[0]), cy: sy(before[1]), r: 7, fill: "none", stroke: color(index), opacity: .5 }));
    });
  }
  if (showSelection) model.centers.forEach((center, index) => {
    svg.append(element("line", { x1: sx(selection.transformed[0]), y1: sy(selection.transformed[1]), x2: sx(center[0]), y2: sy(center[1]), stroke: index === selection.cluster ? color(index) : "#a6b8af", "stroke-width": index === selection.cluster ? 2.8 : 1, "stroke-dasharray": "5 4", opacity: index === selection.cluster ? 1 : .55 }));
  });
  points.forEach((point, index) => {
    const cluster = frame.labels[index], record = dataset.records[index];
    const dot = element("circle", { cx: sx(point[0]), cy: sy(point[1]), r: 4.5, fill: cluster < 0 ? "#93aaa0" : color(cluster), stroke: "white", "stroke-width": .8,
      ...(model ? { role: "button", tabindex: 0, "aria-label": "Inspect row " + record.id + ", cluster " + (model.labels[index] + 1) } : {}) });
    dot.append(element("title", {}, "Row " + record.id + ": " + record.values.join(", ") + (cluster < 0 ? " (unassigned)" : " · cluster " + (cluster + 1))));
    if (model) {
      dot.addEventListener("click", () => inspectPoint(record.values, "Dataset row " + record.id));
      dot.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); inspectPoint(record.values, "Dataset row " + record.id); } });
    }
    svg.append(dot);
  });
  frame.centers.forEach((center, index) => {
    const x = sx(center[0]), y = sy(center[1]);
    svg.append(element("circle", { cx: x, cy: y, r: 10, fill: "white", stroke: color(index), "stroke-width": 1.5 }));
    const cross = element("path", { d: "M " + (x - 6) + " " + y + " H " + (x + 6) + " M " + x + " " + (y - 6) + " V " + (y + 6), stroke: color(index), "stroke-width": 3, fill: "none" });
    cross.append(element("title", {}, "Cluster " + (index + 1) + " center: " + frame.rawCenters[index].map(v => format(v, 5)).join(", ")));
    svg.append(cross);
    if (frame.centers.length <= 12) svg.append(element("text", { x: Math.min(x + 14, left + size - 40), y: y - 12, fill: color(index), "font-weight": 700 }, "C" + (index + 1)));
  });
  if (showSelection) {
    const x = sx(selection.transformed[0]), y = sy(selection.transformed[1]);
    svg.append(element("path", { d: "M " + x + " " + (y - 9) + " L " + (x + 9) + " " + y + " L " + x + " " + (y + 9) + " L " + (x - 9) + " " + y + " Z", fill: "#f7c356", stroke: "#7b5b16", "stroke-width": 2 }));
  }
  svg.append(element("rect", { x: left, y: top, width: size, height: size, stroke: "#b8ccc0", fill: "none" }));
  const prefix = model?.options.standardize ? "Standardized " : "";
  svg.append(element("text", { x: left + size / 2, y: 592, "text-anchor": "middle", fill: "#183439" }, prefix + featureTitle(dataset.features[0])));
  svg.append(element("text", { transform: "translate(22 285) rotate(-90)", "text-anchor": "middle", fill: "#183439" }, prefix + featureTitle(dataset.features[1])));
  if (!model) $("step-description").textContent = "Untrained CSV points. Press Learn cluster centers to calculate their assignments and means.";
}

function download(filename, content, mime) {
  const url = URL.createObjectURL(new Blob([content], { type: mime })), link = document.createElement("a");
  link.href = url; link.download = filename; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function csv(rows) { return rows.map(row => row.map(ClusterData.escapeCSV).join(",")).join("\r\n") + "\r\n"; }

$("dataset").addEventListener("change", loadBuiltIn);
$("upload").addEventListener("change", async () => {
  const file = $("upload").files[0]; if (!file) return;
  const version = ++loadVersion;
  try {
    if (file.size > 1000000) throw Error("Use a CSV smaller than 1 MB.");
    const parsed = ClusterData.parseCSV(await file.text()); if (version !== loadVersion) return;
    acceptDataset(parsed, file.name, "Uploaded CSV. Centers will be learned from these rows, using the two numeric feature columns.");
    $("dataset-link").hidden = true;
  } catch (error) { if (version === loadVersion) $("error").textContent = error.message; }
});
$("train-form").addEventListener("submit", event => { event.preventDefault(); train(); });
$("train-form").addEventListener("change", () => {
  if (model) $("status").textContent = "Settings changed. Retrain to apply them; the current result still uses K=" + model.options.k + ".";
});
$("cancel").addEventListener("click", () => {
  if (worker) worker.terminate(); worker = null; setBusy(false);
  $("status").textContent = "Training cancelled. Change settings and train again."; drawGraph();
});
$("predict-form").addEventListener("submit", event => {
  event.preventDefault();
  if (!$("query-x").value.trim() || !$("query-y").value.trim()) { $("error").textContent = "Enter both numeric features."; return; }
  inspectPoint([Number($("query-x").value), Number($("query-y").value)]);
});
$("step").addEventListener("input", () => { stopReplay(); setFrame(Number($("step").value)); });
$("previous").addEventListener("click", () => { stopReplay(); setFrame(frameIndex - 1); });
$("next").addEventListener("click", () => { stopReplay(); setFrame(frameIndex + 1); });
$("final").addEventListener("click", () => { stopReplay(); setFrame(model.history.length - 1); });
$("play").addEventListener("click", () => {
  if (replayTimer) { stopReplay(); return; }
  setFrame(0); $("play").textContent = "Pause replay";
  replayTimer = setInterval(() => {
    if (frameIndex >= model.history.length - 1) { stopReplay(); return; }
    setFrame(frameIndex + 1);
  }, 900);
});
$("download-labels").addEventListener("click", () => {
  const rows = [["id", ...dataset.features, "cluster", "distance_to_center"], ...dataset.records.map((record, i) => [record.id, ...record.values, model.labels[i] + 1, KMeans.predict(record.values, model).distance])];
  download("kmeans-assignments.csv", csv(rows), "text/csv");
});
$("download-centers").addEventListener("click", () => {
  download("kmeans-centers.csv", csv([["cluster", "assigned_rows", ...dataset.features], ...model.rawCenters.map((center, i) => [i + 1, model.counts[i], ...center])]), "text/csv");
});
$("download-svg").addEventListener("click", () => {
  const graph = $("cluster-graph").cloneNode(true); graph.setAttribute("xmlns", NS);
  graph.prepend(element("style", {}, "text{font-family:Arial,sans-serif;font-size:12px}"));
  download("kmeans-" + model.history[frameIndex].phase + ".svg", new XMLSerializer().serializeToString(graph), "image/svg+xml");
});

if ("serviceWorker" in navigator && ["http:", "https:"].includes(location.protocol)) {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}
loadBuiltIn();
