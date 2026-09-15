"use strict";

/* Run with: node tests/test_browser_tree.js */

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { collectRegions, collectSplits, predict } = require("../dist/tree.js");

const projectDir = path.join(__dirname, "..");
const model = JSON.parse(
  fs.readFileSync(path.join(projectDir, "dist", "model.json"), "utf8")
);
const pageHtml = fs.readFileSync(path.join(projectDir, "dist", "index.html"), "utf8");
const pageCss = fs.readFileSync(path.join(projectDir, "dist", "styles.css"), "utf8");
const appJs = fs.readFileSync(path.join(projectDir, "dist", "app.js"), "utf8");

assert.equal(model.model_type, "decision_tree_classifier");
assert.equal(model.training_examples.length, 48);
assert.equal(model.tree.feature, "systolic_bp_mm_hg");
assert.equal(model.tree.threshold, 140);
assert.equal(model.tree.right.feature, "max_heart_rate_bpm");
assert.equal(model.tree.right.threshold, 115);

const lowLeft = predict(130, 150, model);
const lowRight = predict(155, 100, model);
const highRight = predict(155, 130, model);
assert.equal(lowLeft.label, "low_risk_discharge");
assert.equal(lowRight.label, "low_risk_discharge");
assert.equal(highRight.label, "high_risk_icu");
assert.equal(lowLeft.path.length, 1);
assert.equal(highRight.path.length, 2);
assert.throws(() => predict(0, 100, model), /greater than zero/);

const bounds = { xMin: 80, xMax: 190, yMin: 50, yMax: 170 };
const regions = collectRegions(model.tree, bounds);
const splits = collectSplits(model.tree, bounds);
assert.equal(regions.length, 3);
assert.equal(splits.length, 2);
assert.equal(splits[0].feature, "systolic_bp_mm_hg");
assert.equal(splits[1].feature, "max_heart_rate_bpm");

assert.match(pageHtml, /id="decision-chart"/);
assert.match(pageHtml, /id="decision-path"/);
assert.match(pageHtml, /Synthetic lesson · never clinical use/);
assert.match(appJs, /leaf-region/);
assert.match(appJs, /tree-split-vertical/);
assert.match(appJs, /tree-split-horizontal/);
assert.match(appJs, /name: "evaluate_synthetic_cardiac_tree"/);
assert.match(pageCss, /@keyframes userLand/);
assert.match(pageCss, /@media \(max-width: 620px\)/);

console.log(
  "Browser tests passed: learned questions, three rectangular leaves, two split directions, decision paths, safety copy, responsive chart, and errors."
);
