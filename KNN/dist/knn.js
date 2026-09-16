"use strict";

/*
 * The browser's tiny KNN brain.
 *
 * Picture every saved BMI as a labeled dot on a strip of graph paper. A new
 * person's BMI is another dot. We measure the ruler gap to every saved dot,
 * invite the nearest five to vote, and return the winning label.
 *
 * Inputs: height in centimetres, weight in kilograms, and model.json.
 * Output: a BMI category plus the five examples that explain the answer.
 */
(function createBmiKnn(globalObject) {
  // Convert two body measurements into the one BMI coordinate used by KNN.
  function calculateBmi(heightCm, weightKg) {
    // A real measurement cannot be zero or negative; zero height would also
    // cause division by zero in the formula below.
    if (heightCm <= 0 || weightKg <= 0) {
      throw new Error("Height and weight must both be greater than zero.");
    }

    // BMI uses metres, so 170 centimetres becomes 1.70 metres.
    const heightMetres = heightCm / 100;
    // Example: 65 / (1.70 × 1.70) = about 22.49.
    // Squaring height is like making a floor tile with height on both sides.
    return weightKg / (heightMetres * heightMetres);
  }

  // Check the JSON notebook before trusting it.
  function checkModel(model) {
    // k is the number of voters, so it must be a positive whole number.
    if (!Number.isInteger(model?.k) || model.k <= 0) {
      throw new Error("The model needs a positive whole-number k value.");
    }
    // There must be at least k saved cards to choose k neighbors.
    if (!Array.isArray(model.training_examples) || model.training_examples.length < model.k) {
      throw new Error("The model does not contain enough training examples.");
    }
  }

  // Predict a category for one already-calculated BMI number.
  function predictWithKnn(bmi, model) {
    // Stop early with a friendly error if model.json is broken.
    checkModel(model);

    // STEP 1 — MEASURE: make a ruler-gap copy of every saved example.
    const sortedExamples = model.training_examples
      .map((example) => ({
        // Copy height, weight, BMI, and category from the saved card.
        ...example,
        // abs means the gap is positive: |22 - 25| and |25 - 22| are both 3.
        distance: Math.abs(example.bmi - bmi),
      }))
      // Smallest gaps come first, like lining up children nearest to a cone.
      .sort((first, second) => first.distance - second.distance);

    // STEP 2 — INVITE: slice keeps positions 0 up to, but not including, k.
    const neighbors = sortedExamples.slice(0, model.k);
    // This empty object will become the vote-and-distance scoreboard.
    const votes = {};

    // STEP 3 — VOTE: visit each selected neighbor once.
    for (const neighbor of neighbors) {
      // The first voter for a category needs a new scoreboard row.
      if (!votes[neighbor.category]) {
        votes[neighbor.category] = { count: 0, totalDistance: 0 };
      }
      // Add one ballot to that category.
      votes[neighbor.category].count += 1;
      // Also add its ruler gap so a tied election can favor the closer team.
      votes[neighbor.category].totalDistance += neighbor.distance;
    }

    // Sort category names so the strongest category moves to position zero.
    const category = Object.keys(votes).sort((first, second) => {
      // A positive difference puts the category with more votes first.
      const voteDifference = votes[second].count - votes[first].count;
      // If counts tie (difference 0), the smaller combined ruler gap wins.
      return voteDifference || votes[first].totalDistance - votes[second].totalDistance;
    })[0];

    // Return both the answer word and the evidence used to reach it.
    return { category, neighbors };
  }

  // Bundle the two public functions under one clear name.
  const BmiKnn = { calculateBmi, predictWithKnn };

  // A browser reads window.BmiKnn; the Node test reads module.exports.
  globalObject.BmiKnn = BmiKnn;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = BmiKnn;
  }
// Pick globalThis in Node and window in a normal browser.
})(typeof window === "undefined" ? globalThis : window);
