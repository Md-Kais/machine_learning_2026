"use strict";

/* Run with: node tests/test_browser_model.js */

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { calculateBmi, predictWithKnn } = require("../dist/knn.js");

const modelPath = path.join(__dirname, "..", "dist", "model.json");
const model = JSON.parse(fs.readFileSync(modelPath, "utf8"));
const pageHtml = fs.readFileSync(path.join(__dirname, "..", "dist", "index.html"), "utf8");
const pageCss = fs.readFileSync(path.join(__dirname, "..", "dist", "styles.css"), "utf8");
const appJs = fs.readFileSync(path.join(__dirname, "..", "dist", "app.js"), "utf8");

assert.equal(calculateBmi(170, 65).toFixed(2), "22.49");
assert.ok(
  model.training_examples.every(
    (example) => Number.isFinite(example.height_cm) && Number.isFinite(example.weight_kg)
  )
);

const cases = [
  { height: 170, weight: 50, expected: "Underweight" },
  { height: 170, weight: 65, expected: "Normal" },
  { height: 170, weight: 80, expected: "Overweight" },
  { height: 170, weight: 100, expected: "Obese" },
];

for (const testCase of cases) {
  const bmi = calculateBmi(testCase.height, testCase.weight);
  const result = predictWithKnn(bmi, model);
  assert.equal(result.category, testCase.expected);
  assert.equal(result.neighbors.length, model.k);
  assert.ok(result.neighbors.every((neighbor) => Number.isFinite(neighbor.height_cm)));
  assert.ok(result.neighbors.every((neighbor) => Number.isFinite(neighbor.weight_kg)));
}

assert.throws(() => calculateBmi(0, 65), /greater than zero/);
assert.throws(
  () => predictWithKnn(22, { k: 5, training_examples: [] }),
  /enough training examples/
);

assert.match(pageHtml, /id="neighbor-chart"/);
assert.match(pageHtml, /id="neighbor-details"/);
assert.match(appJs, /Height \(cm\) →/);
assert.match(appJs, /Weight \(kg\) →/);
assert.match(appJs, /class: "neighbor-halo"/);
assert.match(appJs, /class: "neighbor-coordinate"/);
assert.match(pageCss, /@keyframes neighborPop/);
assert.match(pageCss, /@keyframes neighborLabelIn/);
assert.match(pageCss, /@media \(max-width: 640px\)/);

console.log(
  "Browser tests passed: BMI maths, four categories, five labeled graph neighbors, animation, and errors."
);
