"use strict";

/* Run with: node tests/test_browser_model.js */

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { decisionScore, glucoseAtScore, predict } = require("../dist/svm.js");

const projectDir = path.join(__dirname, "..");
const model = JSON.parse(
  fs.readFileSync(path.join(projectDir, "dist", "model.json"), "utf8")
);
const pageHtml = fs.readFileSync(path.join(projectDir, "dist", "index.html"), "utf8");
const pageCss = fs.readFileSync(path.join(projectDir, "dist", "styles.css"), "utf8");
const appJs = fs.readFileSync(path.join(projectDir, "dist", "app.js"), "utf8");

assert.equal(model.model_type, "linear_svm");
assert.equal(model.training_examples.length, 40);
assert.equal(model.weights.length, 2);
assert.ok(model.weights[0] > model.weights[1]);
assert.equal(
  model.training_examples.filter((example) => example.is_support_example).length,
  6
);

assert.equal(predict(85, 24, model).zone, model.zones.lower);
assert.equal(predict(112, 29, model).zone, model.zones.margin);
assert.equal(predict(150, 33, model).zone, model.zones.higher);
assert.ok(decisionScore(85, 24, model) < decisionScore(150, 33, model));
assert.ok(glucoseAtScore(29, -1, model) < glucoseAtScore(29, 1, model));
assert.throws(() => predict(0, 20, model), /greater than zero/);

assert.match(pageHtml, /id="svm-chart"/);
assert.match(pageHtml, /id="glucose"/);
assert.match(pageHtml, /id="bmi"/);
assert.match(appJs, /zone-area-margin/);
assert.match(appJs, /support-ring/);
assert.match(appJs, /clinical-reference-line/);
assert.match(appJs, /name: "screen_diabetes_triage"/);
assert.match(pageCss, /@keyframes inputLand/);
assert.match(pageCss, /@media \(max-width: 620px\)/);

console.log(
  "Browser tests passed: three zones, model math, support points, chart structure, inputs, and errors."
);
