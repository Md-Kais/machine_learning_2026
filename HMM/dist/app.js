/*
 * The browser has two jobs in this project:
 *   1. Read what the learner changes in the controls.
 *   2. Ask the local HMM module to calculate the answer, then draw it.
 *
 * There is deliberately no HMM mathematics in this UI file. The browser-ready
 * algorithms live in hmm.js, while hmm.py remains the matching teaching and
 * Python-test version. Keeping the UI separate makes each job easy to find.
 */

const DEFAULTS = Object.freeze({
  sequence: "H H H H H T H T T",
  fairHeads: 0.5,
  biasedHeads: 0.85,
  fairStay: 0.8,
  biasedStay: 0.8,
  initialFair: 0.5,
});

// Keeping all page elements in one object makes the event code below easier to read.
const elements = {
  engineStatus: document.querySelector("#engine-status"),
  sequenceInput: document.querySelector("#sequence-input"),
  sequenceError: document.querySelector("#sequence-error"),
  tossCount: document.querySelector("#toss-count"),
  runButton: document.querySelector("#run-button"),
  resetButton: document.querySelector("#reset-button"),
  fairHeads: document.querySelector("#fair-heads"),
  biasedHeads: document.querySelector("#biased-heads"),
  fairStay: document.querySelector("#fair-stay"),
  biasedStay: document.querySelector("#biased-stay"),
  fairHeadsValue: document.querySelector("#fair-heads-value"),
  biasedHeadsValue: document.querySelector("#biased-heads-value"),
  fairStayValue: document.querySelector("#fair-stay-value"),
  biasedStayValue: document.querySelector("#biased-stay-value"),
  routeSentence: document.querySelector("#route-sentence"),
  resultBadge: document.querySelector("#result-badge"),
  pathProbability: document.querySelector("#path-probability"),
  timeline: document.querySelector("#timeline"),
  timelineTemplate: document.querySelector("#timeline-step-template"),
  inspectorStep: document.querySelector("#inspector-step"),
  inspectorTitle: document.querySelector("#inspector-title"),
  inspectorExplanation: document.querySelector("#inspector-explanation"),
  fairPosteriorLabel: document.querySelector("#fair-posterior-label"),
  biasedPosteriorLabel: document.querySelector("#biased-posterior-label"),
  fairBar: document.querySelector("#fair-bar"),
  biasedBar: document.querySelector("#biased-bar"),
  posteriorNote: document.querySelector("#posterior-note"),
  presetButtons: document.querySelectorAll("[data-observations]"),
};

let analyseJson = null;
let currentResult = null;
let selectedStep = 0;
let sliderTimer = null;
let latestRequestNumber = 0;

function setEngineStatus(kind, message) {
  elements.engineStatus.classList.toggle("is-ready", kind === "ready");
  elements.engineStatus.classList.toggle("is-error", kind === "error");
  elements.engineStatus.querySelector("span:last-child").textContent = message;
}

/*
 * People naturally type sequences in several ways: HHT, H H T, or H-H-T.
 * We remove harmless separators, but reject every other character so a typo can
 * never silently change the experiment.
 */
function parseSequence(rawValue) {
  const compact = rawValue.toUpperCase().replace(/[\s,\-–—>]+/g, "");

  if (compact.length === 0) {
    throw new Error("Add at least one toss using H or T.");
  }
  if (!/^[HT]+$/.test(compact)) {
    throw new Error("Use only H for heads and T for tails.");
  }
  if (compact.length > 60) {
    throw new Error("Keep the experiment to 60 tosses or fewer.");
  }

  return [...compact];
}

function readModelFromControls() {
  return {
    fair_heads: Number(elements.fairHeads.value) / 100,
    biased_heads: Number(elements.biasedHeads.value) / 100,
    fair_stay: Number(elements.fairStay.value) / 100,
    biased_stay: Number(elements.biasedStay.value) / 100,
    initial_fair: DEFAULTS.initialFair,
  };
}

function readExperimentFromControls() {
  return {
    sequence: parseSequence(elements.sequenceInput.value),
    ...readModelFromControls(),
  };
}

function showControlValues() {
  elements.fairHeadsValue.textContent = `${elements.fairHeads.value}%`;
  elements.biasedHeadsValue.textContent = `${elements.biasedHeads.value}%`;
  elements.fairStayValue.textContent = `${elements.fairStay.value}%`;
  elements.biasedStayValue.textContent = `${elements.biasedStay.value}%`;
}

function renderWaitingTimeline() {
  const observations = parseSequence(DEFAULTS.sequence);
  elements.timeline.replaceChildren();
  elements.timeline.style.gridTemplateColumns = `repeat(${observations.length}, minmax(3.8rem, 1fr))`;

  observations.forEach((observation, index) => {
    const fragment = elements.timelineTemplate.content.cloneNode(true);
    const button = fragment.querySelector("button");
    button.disabled = true;
    button.querySelector(".time-index").textContent = `t${index + 1}`;
    button.querySelector(".observation").textContent = observation;
    const stateCoin = button.querySelector(".state-coin");
    stateCoin.textContent = "…";
    stateCoin.classList.add("is-waiting");
    button.setAttribute("aria-label", `Step ${index + 1}: waiting for the model`);
    elements.timeline.append(fragment);
  });
}

/* Turn [Biased, Biased, Fair] into the compact phrase “Biased × 2 → Fair”. */
function describeRoute(path) {
  const runs = [];
  for (const state of path) {
    const previousRun = runs.at(-1);
    if (previousRun?.state === state) previousRun.count += 1;
    else runs.push({ state, count: 1 });
  }

  return runs
    .map(({ state, count }) => (count === 1 ? state : `${state} × ${count}`))
    .join(" → ");
}

function formatPathProbability(probability) {
  const percent = probability * 100;
  if (percent >= 0.01) return `${percent.toPrecision(3)}%`;
  if (probability === 0) return "≈ 0";
  return probability.toExponential(2);
}

function formatPercent(probability) {
  return `${(probability * 100).toFixed(1)}%`;
}

function renderTimeline(result) {
  elements.timeline.replaceChildren();
  elements.timeline.style.gridTemplateColumns = `repeat(${result.observations.length}, minmax(3.8rem, 1fr))`;

  result.observations.forEach((observation, index) => {
    const fragment = elements.timelineTemplate.content.cloneNode(true);
    const button = fragment.querySelector("button");
    const state = result.path[index];
    const stateCoin = button.querySelector(".state-coin");

    button.querySelector(".time-index").textContent = `t${index + 1}`;
    button.querySelector(".observation").textContent = observation;
    stateCoin.textContent = state.toUpperCase();
    stateCoin.classList.toggle("is-biased", state === "Biased");
    button.classList.toggle("is-selected", index === selectedStep);
    button.setAttribute("aria-pressed", String(index === selectedStep));
    button.setAttribute(
      "aria-label",
      `Step ${index + 1}: heard ${observation === "H" ? "heads" : "tails"}, predicted ${state} coin`,
    );
    button.addEventListener("click", () => {
      selectedStep = index;
      renderTimeline(currentResult);
      renderInspector(currentResult, selectedStep);
    });

    elements.timeline.append(fragment);
  });
}

function explainDecision(result, index) {
  const state = result.path[index];
  const otherState = state === "Fair" ? "Biased" : "Fair";
  const detail = result.trellis[index][state];
  const emission = result.model.emissions[state][result.observations[index]];
  const otherEmission = result.model.emissions[otherState][result.observations[index]];

  if (index === 0) {
    return `At the first toss, Viterbi combines the ${formatPercent(detail.initial_probability)} starting chance with the ${formatPercent(emission)} chance that the ${state.toLowerCase()} coin emits this result. That gives the stronger opening route.`;
  }

  const action = detail.best_previous === state ? "keeps" : "switches to";
  return `For this ${result.observations[index]}, the ${state.toLowerCase()} coin gives ${formatPercent(emission)} emission likelihood, compared with ${formatPercent(otherEmission)} for the ${otherState.toLowerCase()} coin. The strongest incoming route ${action} ${state.toLowerCase()} with a ${formatPercent(detail.transition_probability)} transition chance.`;
}

function renderInspector(result, index) {
  const observation = result.observations[index];
  const state = result.path[index];
  const posterior = result.posterior[index];

  elements.inspectorStep.textContent = `Step ${index + 1} · Heard ${observation}`;
  elements.inspectorTitle.textContent = `Viterbi chooses the ${state.toLowerCase()} coin`;
  elements.inspectorExplanation.textContent = explainDecision(result, index);
  elements.fairPosteriorLabel.textContent = formatPercent(posterior.Fair);
  elements.biasedPosteriorLabel.textContent = formatPercent(posterior.Biased);
  elements.fairBar.style.width = formatPercent(posterior.Fair);
  elements.biasedBar.style.width = formatPercent(posterior.Biased);
  elements.posteriorNote.textContent = `Posterior probability using all ${result.observations.length} observations`;
}

function renderResult(result) {
  currentResult = result;
  selectedStep = Math.min(selectedStep, result.observations.length - 1);
  elements.tossCount.textContent = `${result.observations.length} ${result.observations.length === 1 ? "toss" : "tosses"}`;
  elements.routeSentence.textContent = describeRoute(result.path);
  elements.pathProbability.textContent = formatPathProbability(result.path_probability);
  elements.resultBadge.textContent = `Decoded ${result.observations.length} ${result.observations.length === 1 ? "toss" : "tosses"}`;
  elements.resultBadge.dataset.kind = "done";
  renderTimeline(result);
  renderInspector(result, selectedStep);
}

async function runExperiment({ moveToFirstStep = false } = {}) {
  let request;
  try {
    request = readExperimentFromControls();
    elements.sequenceError.textContent = "";
  } catch (error) {
    elements.sequenceError.textContent = error.message;
    elements.resultBadge.textContent = "Check the tosses";
    elements.resultBadge.dataset.kind = "error";
    return null;
  }

  // Keep any changes made while the small local model is being prepared. The
  // calculation starts as soon as loadModelEngine connects it below.
  if (!analyseJson) return null;

  const thisRequestNumber = ++latestRequestNumber;
  elements.runButton.disabled = true;
  elements.runButton.textContent = "Working…";
  elements.resultBadge.textContent = "Decoding…";
  elements.resultBadge.dataset.kind = "working";

  try {
    // The local model accepts and returns plain JSON text. That makes the bridge
    // small, predictable, and easy to compare with Python's analyse_json.
    const result = JSON.parse(analyseJson(JSON.stringify(request)));
    if (thisRequestNumber !== latestRequestNumber) return null;
    if (moveToFirstStep) selectedStep = 0;
    renderResult(result);
    setEngineStatus("ready", "Offline model ready");
    return result;
  } catch (error) {
    console.error(error);
    elements.sequenceError.textContent = "The model calculation could not finish. Check the values and try again.";
    setEngineStatus("error", "Model calculation failed");
    elements.resultBadge.textContent = "Could not decode";
    elements.resultBadge.dataset.kind = "error";
    return null;
  } finally {
    if (thisRequestNumber === latestRequestNumber) {
      elements.runButton.disabled = false;
      elements.runButton.textContent = "Decode";
    }
  }
}

function resetControls() {
  elements.sequenceInput.value = DEFAULTS.sequence;
  elements.fairHeads.value = String(DEFAULTS.fairHeads * 100);
  elements.biasedHeads.value = String(DEFAULTS.biasedHeads * 100);
  elements.fairStay.value = String(DEFAULTS.fairStay * 100);
  elements.biasedStay.value = String(DEFAULTS.biasedStay * 100);
  elements.sequenceError.textContent = "";
  showControlValues();
  void runExperiment({ moveToFirstStep: true });
}

function scheduleExperiment() {
  clearTimeout(sliderTimer);
  sliderTimer = setTimeout(() => void runExperiment(), 120);
}

function wireVisibleControls() {
  elements.runButton.addEventListener("click", () => void runExperiment({ moveToFirstStep: true }));
  elements.resetButton.addEventListener("click", resetControls);
  elements.sequenceInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") void runExperiment({ moveToFirstStep: true });
  });

  [elements.fairHeads, elements.biasedHeads, elements.fairStay, elements.biasedStay].forEach((slider) => {
    slider.addEventListener("input", () => {
      showControlValues();
      scheduleExperiment();
    });
  });

  // Ready-made sequences let a first-time visitor learn by clicking instead of
  // deciding what to type. The same local calculation is used after selection.
  elements.presetButtons.forEach((button) => {
    button.addEventListener("click", () => {
      elements.sequenceInput.value = button.dataset.observations;
      elements.sequenceError.textContent = "";
      void runExperiment({ moveToFirstStep: true });
    });
  });
}

/*
 * WebMCP is an emerging browser feature that lets an AI use the same action as
 * the visible Decode button. Feature detection keeps ordinary browsers working.
 */
function registerWebMcpTool() {
  const context = document.modelContext;
  if (!context?.registerTool) return;

  const probabilitySchema = { type: "number", minimum: 0.05, maximum: 0.95 };
  const lifecycle = new AbortController();

  try {
    void Promise.resolve(
      context.registerTool(
        {
          name: "decode_hidden_coin_sequence",
          title: "Decode hidden coin sequence",
          description: "Set a heads/tails sequence and HMM probabilities, update the visible experiment, and return the most likely hidden coin route.",
          inputSchema: {
            type: "object",
            properties: {
              sequence: { type: "string", pattern: "^[HhTt\\s,\\-–—>]+$", minLength: 1 },
              fairHeads: probabilitySchema,
              biasedHeads: probabilitySchema,
              fairStay: probabilitySchema,
              biasedStay: probabilitySchema,
            },
            required: ["sequence"],
            additionalProperties: false,
          },
          annotations: { readOnlyHint: false, untrustedContentHint: false },
          async execute(input) {
            if (!input || typeof input.sequence !== "string") {
              throw new Error("sequence must be a string containing H and T.");
            }

            // Validate before changing the visible page, so a bad request leaves
            // the learner's current experiment exactly as it was.
            parseSequence(input.sequence);
            const optionalProbabilities = ["fairHeads", "biasedHeads", "fairStay", "biasedStay"];
            for (const name of optionalProbabilities) {
              if (input[name] !== undefined && (!Number.isFinite(input[name]) || input[name] < 0.05 || input[name] > 0.95)) {
                throw new Error(`${name} must be a number from 0.05 to 0.95.`);
              }
            }

            elements.sequenceInput.value = input.sequence;
            if (input.fairHeads !== undefined) elements.fairHeads.value = String(input.fairHeads * 100);
            if (input.biasedHeads !== undefined) elements.biasedHeads.value = String(input.biasedHeads * 100);
            if (input.fairStay !== undefined) elements.fairStay.value = String(input.fairStay * 100);
            if (input.biasedStay !== undefined) elements.biasedStay.value = String(input.biasedStay * 100);
            showControlValues();

            const result = await runExperiment({ moveToFirstStep: true });
            if (!result) throw new Error("The experiment could not be decoded.");
            return { observations: result.observations, path: result.path, pathProbability: result.path_probability };
          },
        },
        { signal: lifecycle.signal },
      ),
    ).catch((error) => console.warn("WebMCP tool registration failed", error));
  } catch (error) {
    console.warn("WebMCP is present but tool registration failed", error);
  }

  window.addEventListener("pagehide", () => lifecycle.abort(), { once: true });
}

async function loadModelEngine() {
  try {
    if (typeof window.HiddenCoinHMM?.analyseJson !== "function") {
      throw new Error("The local HMM module did not load.");
    }

    // The complete model was loaded from hmm.js with the page. No network,
    // WebAssembly runtime, package install, or server request is necessary.
    analyseJson = window.HiddenCoinHMM.analyseJson;
    setEngineStatus("ready", "Offline model ready");
    await runExperiment({ moveToFirstStep: true });
    registerWebMcpTool();
  } catch (error) {
    console.error(error);
    setEngineStatus("error", "Offline model unavailable");
    elements.sequenceError.textContent = "The local model could not load. Refresh the page and try again.";
    elements.runButton.disabled = true;
  }
}

// A service worker saves the local page files after the first hosted visit. It
// is skipped for file:// pages because browsers only allow workers on web origins.
if ("serviceWorker" in navigator && location.protocol !== "file:") {
  navigator.serviceWorker.register("./sw.js").catch((error) => {
    console.warn("Offline cache could not start", error);
  });
}

showControlValues();
renderWaitingTimeline();
wireVisibleControls();
elements.runButton.disabled = true;
elements.runButton.textContent = "Loading…";
void loadModelEngine();
