/*
 * The browser's Decision Tree brain.
 *
 * A Decision Tree is a choose-your-path story. Each question box compares one
 * input number with a fence number called a threshold. "Yes" walks left and
 * "No" walks right until a leaf (final answer card) is reached.
 */
(function (root, factory) {
  // Build the public functions once.
  const api = factory();
  // Node.js tests collect the functions through module.exports.
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  // Browsers collect the same functions from window.DecisionTree.
  root.DecisionTree = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  // Refuse to walk a tree if its root question or label dictionary is missing.
  function validateModel(model) {
    if (!model || !model.tree || !model.labels) {
      throw new Error("The Decision Tree model is missing or incomplete.");
    }
  }

  // Follow the yes/no path for two numeric measurements.
  function predict(systolicBp, maxHeartRate, model) {
    // Confirm model.json has the pieces we need.
    validateModel(model);
    // Number.isFinite rejects blanks converted to NaN and endless Infinity.
    if (!Number.isFinite(systolicBp) || !Number.isFinite(maxHeartRate)) {
      throw new Error("Blood pressure and heart rate must be numbers.");
    }
    // A body measurement cannot be zero or negative.
    if (systolicBp <= 0 || maxHeartRate <= 0) {
      throw new Error("Blood pressure and heart rate must be greater than zero.");
    }

    // Put both inputs in drawers named exactly like the exported feature names.
    const values = {
      systolic_bp_mm_hg: systolicBp,
      max_heart_rate_bpm: maxHeartRate,
    };
    // path is an empty breadcrumb trail that will remember every answer.
    const path = [];
    // Start at the tree's top question, called the root.
    let node = model.tree;

    // A leaf has no more question; keep walking while this is not a leaf.
    while (!node.is_leaf) {
      // Read the measurement requested by this question box.
      const value = values[node.feature];
      // The threshold is a fence: values on or below it walk left.
      const goLeft = value <= node.threshold;
      // Save an inspectable breadcrumb for the result panel.
      path.push({
        node_id: node.id,
        feature: node.feature,
        feature_title: node.feature_title,
        unit: node.unit,
        value,
        threshold: node.threshold,
        answer: goLeft ? "Yes" : "No",
        branch: goLeft ? "left" : "right",
      });
      // Move to exactly one child question or final leaf.
      node = goLeft ? node.left : node.right;
    }

    // The leaf supplies the toy label; the path explains how it was reached.
    return {
      label: node.prediction,
      label_title: model.labels[node.prediction],
      leaf_id: node.id,
      path,
    };
  }

  // Turn every final leaf into a colored rectangle for the x/y decision map.
  function collectRegions(node, bounds, regions = []) {
    // Reaching a leaf means the current rectangle belongs to one final label.
    if (node.is_leaf) {
      regions.push({ ...bounds, label: node.prediction, leaf_id: node.id });
      return regions;
    }

    // A blood-pressure question cuts the map vertically, like placing a fence
    // from the top to the bottom. Left values end at the threshold.
    if (node.feature === "systolic_bp_mm_hg") {
      collectRegions(node.left, { ...bounds, xMax: node.threshold }, regions);
      collectRegions(node.right, { ...bounds, xMin: node.threshold }, regions);
    } else {
      // A heart-rate question cuts horizontally: below walks left, above right.
      collectRegions(node.left, { ...bounds, yMax: node.threshold }, regions);
      collectRegions(node.right, { ...bounds, yMin: node.threshold }, regions);
    }
    // Return every rectangle collected during the recursive walk.
    return regions;
  }

  // Collect the fence lines themselves so app.js can draw them.
  function collectSplits(node, bounds, splits = [], depth = 0) {
    // Leaves draw regions but add no new dividing line.
    if (node.is_leaf) {
      return splits;
    }

    if (node.feature === "systolic_bp_mm_hg") {
      // A vertical line has one x value and stretches through y from/to.
      splits.push({
        node_id: node.id,
        feature: node.feature,
        value: node.threshold,
        from: bounds.yMin,
        to: bounds.yMax,
        depth,
      });
      // Search for any smaller fences inside the two new rectangles.
      collectSplits(node.left, { ...bounds, xMax: node.threshold }, splits, depth + 1);
      collectSplits(node.right, { ...bounds, xMin: node.threshold }, splits, depth + 1);
    } else {
      // A horizontal line has one y value and stretches through x from/to.
      splits.push({
        node_id: node.id,
        feature: node.feature,
        value: node.threshold,
        from: bounds.xMin,
        to: bounds.xMax,
        depth,
      });
      collectSplits(node.left, { ...bounds, yMax: node.threshold }, splits, depth + 1);
      collectSplits(node.right, { ...bounds, yMin: node.threshold }, splits, depth + 1);
    }
    return splits;
  }

  // Only these three helpers are part of the public browser/Node API.
  return { predict, collectRegions, collectSplits };
});
