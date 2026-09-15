/*
 * The browser's small linear-SVM calculator.
 *
 * Imagine two teams of players with a wide straight road between them. The
 * learned weights tilt the road, the bias slides it, and scores -1 and +1 mark
 * its two edges. A new glucose/BMI dot gets a signed position on this map.
 */
(function (root, factory) {
  // Create the public calculator functions once.
  const api = factory();
  // Give Node.js tests the same functions used by the browser.
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  // Publish them to browser code as window.SimpleSvm.
  root.SimpleSvm = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  // Check that model.json contains two weights, two means, and two scales.
  function validateModel(model) {
    if (
      !model ||
      !Array.isArray(model.weights) ||
      model.weights.length !== 2 ||
      !Array.isArray(model.means) ||
      !Array.isArray(model.scales)
    ) {
      throw new Error("The SVM model is missing or incomplete.");
    }
  }

  // Calculate where a new dot sits relative to the learned center line.
  function decisionScore(fastingGlucose, bmi, model) {
    // Do not use a broken or incomplete model package.
    validateModel(model);
    // Reject NaN and Infinity; both inputs must be real, finite numbers.
    if (!Number.isFinite(fastingGlucose) || !Number.isFinite(bmi)) {
      throw new Error("Glucose and BMI must be numbers.");
    }
    // Physical measurements cannot be zero or negative.
    if (fastingGlucose <= 0 || bmi <= 0) {
      throw new Error("Glucose and BMI must be greater than zero.");
    }

    // Standardizing is like changing two different rulers into equal-sized
    // map steps. Subtract the team's middle (mean), then divide by its usual
    // spread (scale). A value at the mean becomes 0.
    const glucoseScaled = (fastingGlucose - model.means[0]) / model.scales[0];
    const bmiScaled = (bmi - model.means[1]) / model.scales[1];
    // Each scaled step is multiplied by its learned steering strength. The bias
    // slides the road without changing its tilt. The output is a signed score,
    // not a probability: negative is one side and positive is the other.
    return (
      model.weights[0] * glucoseScaled +
      model.weights[1] * bmiScaled +
      model.bias
    );
  }

  // Turn one signed road position into a plain-English teaching zone.
  function zoneFromScore(score, model) {
    // At or beyond the -1 road edge belongs to the lower side.
    if (score <= model.margin.lower) {
      return model.zones.lower;
    }
    // At or beyond the +1 road edge belongs to the higher side.
    if (score >= model.margin.upper) {
      return model.zones.higher;
    }
    // Anything between -1 and +1 stands inside the wide road (margin).
    return model.zones.margin;
  }

  // Give callers both the exact score and its friendly category.
  function predict(fastingGlucose, bmi, model) {
    const score = decisionScore(fastingGlucose, bmi, model);
    return { score, zone: zoneFromScore(score, model) };
  }

  // Find the glucose x-coordinate where a chosen SVM line crosses one BMI row.
  // app.js uses this to draw the -1 edge, center 0 line, and +1 edge.
  function glucoseAtScore(bmi, targetScore, model) {
    validateModel(model);
    // Put BMI onto the same training ruler.
    const bmiScaled = (bmi - model.means[1]) / model.scales[1];
    // Rearrange score = wg*x + wb*y + bias to solve for scaled glucose x.
    const glucoseScaled =
      (targetScore - model.bias - model.weights[1] * bmiScaled) /
      model.weights[0];
    // Undo scaling so the graph receives a familiar mg/dL number.
    return model.means[0] + glucoseScaled * model.scales[0];
  }

  // These are the only math helpers the UI and tests need.
  return { decisionScore, zoneFromScore, predict, glucoseAtScore };
});
