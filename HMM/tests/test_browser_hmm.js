"use strict";

const assert = require("node:assert/strict");
const path = require("node:path");
const hmm = require(path.join(__dirname, "..", "dist", "hmm.js"));

function closeTo(actual, expected, tolerance = 1e-12) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    "Expected " + actual + " to be within " + tolerance + " of " + expected,
  );
}

// This expected route and probability came from the fully tested Python engine.
// Matching it protects the offline JavaScript translation from accidental drift.
const defaultResult = hmm.analyse("H H H H H T H T T");
assert.deepEqual(defaultResult.path, [
  "Biased", "Biased", "Biased", "Biased", "Biased",
  "Fair", "Fair", "Fair", "Fair",
]);
closeTo(defaultResult.path_probability, 0.0005815734271999997);
closeTo(defaultResult.sequence_probability, 0.003710145020643719);
closeTo(defaultResult.posterior[0].Fair, 0.25248737540449806);
closeTo(defaultResult.posterior.at(-1).Biased, 0.13998976519953008);

// A different set of probability sliders checks that options really reach all
// three model tables instead of the browser silently using only its defaults.
const customResult = hmm.analyse("HTHT", {
  fair_heads: 0.45,
  biased_heads: 0.8,
  fair_stay: 0.65,
  biased_stay: 0.75,
  initial_fair: 0.4,
});
assert.deepEqual(customResult.path, ["Biased", "Fair", "Fair", "Fair"]);
closeTo(customResult.path_probability, 0.0069015374999999955);
closeTo(customResult.posterior[1].Fair, 0.5894337451505547);

// Every posterior is a complete two-slice pie.
for (const step of customResult.posterior) {
  closeTo(step.Fair + step.Biased, 1);
}

// Friendly separators are accepted, but impossible lesson inputs are rejected.
assert.deepEqual(hmm.cleanObservations("h-h, t > H"), ["H", "H", "T", "H"]);
assert.throws(() => hmm.analyse("HX"), /H \(heads\) or T \(tails\)/);
assert.throws(() => hmm.makeModel({ fair_heads: 1 }), /less than 1/);

console.log("HMM browser-model tests passed.");
