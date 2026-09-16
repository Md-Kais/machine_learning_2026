/*
 * A browser-friendly Hidden Markov Model for the hidden two-coin game.
 *
 * STORY FIRST
 * -----------
 * A dealer stands behind a curtain with two coins. We hear Heads or Tails, but
 * we cannot see which coin the dealer used. Think of the possible answers as
 * two rows of stepping stones: a Fair row and a Biased row. At every toss, the
 * Viterbi algorithm keeps the strongest route reaching each stone. Little
 * back-pointers remember where those two winning routes came from.
 *
 * This file mirrors hmm.py so the lesson works with no internet connection.
 * It contains no server calls, downloads, libraries, or secret model service.
 */

(function exposeHiddenCoinHMM(root, factory) {
  // A browser receives window.HiddenCoinHMM. Node receives module.exports so
  // the same mathematical brain can be checked by an automatic test.
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.HiddenCoinHMM = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function buildHiddenCoinHMM() {
  "use strict";

  // These are the only two hidden coins and the only two sounds we can hear.
  const STATES = Object.freeze(["Fair", "Biased"]);
  const ALLOWED_OBSERVATIONS = new Set(["H", "T"]);

  function checkProbability(name, value) {
    // Turn a slider value into an ordinary decimal number such as 0.85.
    const number = Number(value);

    // A probability is like a slice of one whole pie. It must be bigger than
    // an empty slice (0) and smaller than the entire pie (1) in this lesson.
    if (!Number.isFinite(number) || number <= 0 || number >= 1) {
      throw new Error(name + " must be greater than 0 and less than 1.");
    }
    return number;
  }

  function cleanObservations(sequence) {
    let observations;

    if (typeof sequence === "string") {
      // Spaces, commas, arrows, and dashes are just gaps between letter tiles.
      // Remove those gaps, make letters uppercase, and keep the H/T tiles.
      const compact = sequence.toUpperCase().replace(/[\s,\-–—>]+/g, "");
      observations = [...compact];
    } else if (sequence && typeof sequence[Symbol.iterator] === "function") {
      // The UI normally sends an array like ["H", "T"]. Make a fresh copy so
      // the algorithm never changes the caller's own box of observations.
      observations = [...sequence].map((item) => String(item).toUpperCase());
    } else {
      throw new Error("The observation sequence must contain at least one toss.");
    }

    if (observations.length === 0) {
      throw new Error("The observation sequence must contain at least one toss.");
    }
    if (observations.length > 60) {
      throw new Error("The observation sequence may contain at most 60 tosses.");
    }
    if (observations.some((item) => !ALLOWED_OBSERVATIONS.has(item))) {
      throw new Error("Every observation must be H (heads) or T (tails).");
    }
    return observations;
  }

  function makeModel(options = {}) {
    // Read each supplied chance, or use the classroom defaults. A value such
    // as 0.85 means 85 pieces out of a 100-piece probability pie.
    const fairHeads = checkProbability("fair_heads", options.fair_heads ?? 0.5);
    const biasedHeads = checkProbability("biased_heads", options.biased_heads ?? 0.85);
    const fairStay = checkProbability("fair_stay", options.fair_stay ?? 0.8);
    const biasedStay = checkProbability("biased_stay", options.biased_stay ?? 0.8);
    const initialFair = checkProbability("initial_fair", options.initial_fair ?? 0.5);

    // INITIAL: which coin is likely to be used first? The two slices total 1.
    const initial = {
      Fair: initialFair,
      Biased: 1 - initialFair,
    };

    // TRANSITIONS: road signs between yesterday's coin and today's coin.
    // If Fair stays with chance 0.8, its only other road—switching—gets 0.2.
    const transitions = {
      Fair: { Fair: fairStay, Biased: 1 - fairStay },
      Biased: { Fair: 1 - biasedStay, Biased: biasedStay },
    };

    // EMISSIONS: the chance each hidden coin makes the sound H or T.
    // A coin cannot make anything else, so its H and T slices total 1.
    const emissions = {
      Fair: { H: fairHeads, T: 1 - fairHeads },
      Biased: { H: biasedHeads, T: 1 - biasedHeads },
    };

    return { initial, transitions, emissions };
  }

  function viterbi(observations, model) {
    // delta[time][state] is the score of the strongest route that reaches one
    // state at one time. Logs turn many multiplications into safer additions.
    const delta = [];

    // psi[time][state] is a chalk arrow pointing to the previous winning stone.
    const psi = [];

    // The trellis keeps friendly ingredients that the explanation panel shows.
    const trellis = [];

    // VITERBI 1: INITIALIZATION
    // Formula: starting chance × chance of making the first heard result.
    const firstObservation = observations[0];
    const firstScores = {};
    const firstPointers = {};
    const firstDetails = {};

    for (const state of STATES) {
      const initialProbability = model.initial[state];
      const emissionProbability = model.emissions[state][firstObservation];

      // log(a × b) equals log(a) + log(b). This stops a long route from
      // becoming a number too tiny for the computer to store accurately.
      const logScore = Math.log(initialProbability) + Math.log(emissionProbability);
      firstScores[state] = logScore;
      firstPointers[state] = null;
      firstDetails[state] = {
        best_previous: null,
        initial_probability: initialProbability,
        transition_probability: null,
        emission_probability: emissionProbability,
        path_probability: Math.exp(logScore),
      };
    }

    delta.push(firstScores);
    psi.push(firstPointers);
    trellis.push(firstDetails);

    // VITERBI 2: RECURSION
    // For every later sound, compare the road arriving from Fair with the road
    // arriving from Biased. Keep one winner for each possible current coin.
    for (let time = 1; time < observations.length; time += 1) {
      const observation = observations[time];
      const scoresNow = {};
      const pointersNow = {};
      const detailsNow = {};

      for (const currentState of STATES) {
        // Begin with the road from Fair. If the road from Biased is strictly
        // stronger, replace it. A tie stays with Fair, matching Python's max.
        let bestPrevious = STATES[0];
        let bestIncomingScore =
          delta[time - 1][bestPrevious]
          + Math.log(model.transitions[bestPrevious][currentState]);

        for (const previousState of STATES.slice(1)) {
          const incomingScore =
            delta[time - 1][previousState]
            + Math.log(model.transitions[previousState][currentState]);
          if (incomingScore > bestIncomingScore) {
            bestPrevious = previousState;
            bestIncomingScore = incomingScore;
          }
        }

        // After taking the winning road, the chosen coin must also make the
        // sound heard today. Add that emission in log space.
        const emissionProbability = model.emissions[currentState][observation];
        const bestLogScore = bestIncomingScore + Math.log(emissionProbability);

        scoresNow[currentState] = bestLogScore;
        pointersNow[currentState] = bestPrevious;
        detailsNow[currentState] = {
          best_previous: bestPrevious,
          initial_probability: null,
          transition_probability: model.transitions[bestPrevious][currentState],
          emission_probability: emissionProbability,
          path_probability: Math.exp(bestLogScore),
        };
      }

      delta.push(scoresNow);
      psi.push(pointersNow);
      trellis.push(detailsNow);
    }

    // VITERBI 3: TERMINATION
    // The final column has one best route ending at each coin. Pick the larger.
    let finalState = STATES[0];
    if (delta.at(-1).Biased > delta.at(-1).Fair) finalState = STATES[1];
    const bestLogProbability = delta.at(-1)[finalState];

    // VITERBI 4: BACKTRACKING
    // Start at the winning final stone and follow its chalk arrows backward.
    const reversedPath = [finalState];
    for (let time = observations.length - 1; time > 0; time -= 1) {
      const currentState = reversedPath.at(-1);
      reversedPath.push(psi[time][currentState]);
    }

    // We walked finish-to-start, so reverse the list into the order heard.
    const path = reversedPath.reverse();
    return {
      path,
      path_probability: Math.exp(bestLogProbability),
      path_log_probability: bestLogProbability,
      trellis,
    };
  }

  function forwardBackward(observations, model) {
    // Forward is like passing all earlier clues from left to right. Each column
    // is divided by its total so Fair + Biased becomes one whole probability pie.
    const forward = [];
    const scales = [];
    const first = {};
    for (const state of STATES) {
      first[state] = model.initial[state] * model.emissions[state][observations[0]];
    }
    const firstScale = first.Fair + first.Biased;
    forward.push({ Fair: first.Fair / firstScale, Biased: first.Biased / firstScale });
    scales.push(firstScale);

    for (let time = 1; time < observations.length; time += 1) {
      const column = {};
      for (const stateNow of STATES) {
        // Both earlier roads can arrive at this stone, so add both road weights.
        let arrivingProbability = 0;
        for (const stateBefore of STATES) {
          arrivingProbability +=
            forward[time - 1][stateBefore]
            * model.transitions[stateBefore][stateNow];
        }
        column[stateNow] = arrivingProbability * model.emissions[stateNow][observations[time]];
      }
      const scale = column.Fair + column.Biased;
      forward.push({ Fair: column.Fair / scale, Biased: column.Biased / scale });
      scales.push(scale);
    }

    // Backward passes future clues from right to left. At the last toss there
    // are no future clues, so both coins start with a neutral helper value of 1.
    const backward = observations.map(() => ({ Fair: 0, Biased: 0 }));
    backward[backward.length - 1] = { Fair: 1, Biased: 1 };

    for (let time = observations.length - 2; time >= 0; time -= 1) {
      const nextObservation = observations[time + 1];
      for (const stateNow of STATES) {
        let futureEvidence = 0;
        for (const stateNext of STATES) {
          futureEvidence +=
            model.transitions[stateNow][stateNext]
            * model.emissions[stateNext][nextObservation]
            * backward[time + 1][stateNext];
        }
        backward[time][stateNow] = futureEvidence / scales[time + 1];
      }
    }

    // Join the left-looking clues and right-looking clues at every toss. Divide
    // by their total so the two final bars always add to 100%.
    const posterior = observations.map((unused, time) => {
      const fairWeight = forward[time].Fair * backward[time].Fair;
      const biasedWeight = forward[time].Biased * backward[time].Biased;
      const total = fairWeight + biasedWeight;
      return { Fair: fairWeight / total, Biased: biasedWeight / total };
    });

    // Multiplying the saved scale pieces gives the chance of hearing the whole
    // sequence across all possible hidden paths. Sum logs, then undo the log.
    const sequenceLogProbability = scales.reduce((sum, scale) => sum + Math.log(scale), 0);
    return {
      posterior,
      sequence_probability: Math.exp(sequenceLogProbability),
      sequence_log_probability: sequenceLogProbability,
    };
  }

  function analyse(sequence, options = {}) {
    // This is the front door: clean the sounds, build the three probability
    // tables, find the best complete route, and make per-step probability bars.
    const observations = cleanObservations(sequence);
    const model = makeModel(options);
    return {
      observations,
      model,
      ...viterbi(observations, model),
      ...forwardBackward(observations, model),
    };
  }

  function analyseJson(configJson) {
    // JSON is a labeled lunchbox of plain text and numbers. It gives app.js one
    // small doorway that is also easy to compare with hmm.py's analyse_json.
    const config = JSON.parse(configJson);
    return JSON.stringify(analyse(config.sequence, config));
  }

  return Object.freeze({
    STATES,
    cleanObservations,
    makeModel,
    viterbi,
    forwardBackward,
    analyse,
    analyseJson,
  });
});
