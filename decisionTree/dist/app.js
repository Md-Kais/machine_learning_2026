"use strict";

const form = document.querySelector("#triage-form");
const bpInput = document.querySelector("#systolic-bp");
const heartRateInput = document.querySelector("#max-heart-rate");
const evaluateButton = document.querySelector("#evaluate-button");
// Quick samples expose their two safe teaching values through data attributes.
const presetButtons = document.querySelectorAll("[data-bp][data-heart-rate]");
const modelStatus = document.querySelector("#model-status");
const resultPanel = document.querySelector("#result-panel");
const resultName = document.querySelector("#result-name");
const resultExplanation = document.querySelector("#result-explanation");
const pathSummary = document.querySelector("#path-summary");
const pathList = document.querySelector("#decision-path");
const graphSummary = document.querySelector("#graph-summary");
const plotWrap = document.querySelector("#plot-wrap");
const decisionChart = document.querySelector("#decision-chart");

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const LOW_LABEL = "low_risk_discharge";
const HIGH_LABEL = "high_risk_icu";

let model = null;
let currentCase = null;

function makeSvgElement(name, attributes = {}, text = "") {
  const element = document.createElementNS(SVG_NAMESPACE, name);
  for (const [attribute, value] of Object.entries(attributes)) {
    element.setAttribute(attribute, String(value));
  }
  if (text) {
    element.textContent = text;
  }
  return element;
}

function mapNumber(value, domainMinimum, domainMaximum, rangeMinimum, rangeMaximum) {
  const position = (value - domainMinimum) / (domainMaximum - domainMinimum);
  return rangeMinimum + position * (rangeMaximum - rangeMinimum);
}

function numberTicks(minimum, maximum, count) {
  return Array.from({ length: count }, (_, index) => {
    return minimum + ((maximum - minimum) * index) / (count - 1);
  });
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function validateInput(input, minimum, maximum, errorId, friendlyName) {
  const value = Number(input.value);
  const errorElement = document.querySelector(`#${errorId}`);
  let message = "";

  if (input.value.trim() === "") {
    message = `Please enter ${friendlyName}.`;
  } else if (!Number.isFinite(value) || value < minimum || value > maximum) {
    message = `${friendlyName} must be from ${minimum} to ${maximum}.`;
  }

  input.setAttribute("aria-invalid", String(Boolean(message)));
  errorElement.textContent = message;
  return message === "";
}

function renderPath(prediction) {
  pathList.replaceChildren();

  prediction.path.forEach((step, index) => {
    const item = document.createElement("li");
    const number = document.createElement("span");
    const content = document.createElement("div");
    const question = document.createElement("strong");
    const calculation = document.createElement("p");

    number.className = "path-number";
    number.textContent = String(index + 1);
    question.textContent = `Is ${step.feature_title.toLowerCase()} ≤ ${step.threshold.toFixed(0)} ${step.unit}?`;
    calculation.textContent = `${step.value.toFixed(0)} ≤ ${step.threshold.toFixed(0)} → ${step.answer}`;
    item.style.setProperty("--path-index", index);
    content.append(question, calculation);
    item.append(number, content);
    pathList.append(item);
  });

  const finalItem = document.createElement("li");
  const finalNumber = document.createElement("span");
  const finalContent = document.createElement("div");
  const finalHeading = document.createElement("strong");
  const finalText = document.createElement("p");
  finalItem.className = "path-leaf";
  finalItem.style.setProperty("--path-index", prediction.path.length);
  finalNumber.className = "path-number";
  finalNumber.textContent = "✓";
  finalHeading.textContent = "Reach a leaf (final answer box)";
  finalText.textContent = `Toy label: ${prediction.label_title}`;
  finalContent.append(finalHeading, finalText);
  finalItem.append(finalNumber, finalContent);
  pathList.append(finalItem);
}

function describePath(prediction) {
  return prediction.path
    .map((step) => {
      const shortFeature =
        step.feature === "systolic_bp_mm_hg" ? "BP" : "heart rate";
      const operator = step.answer === "Yes" ? "≤" : ">";
      return `${shortFeature} ${operator} ${step.threshold.toFixed(0)}`;
    })
    .join(" AND ");
}

function polygonPoints(points) {
  return points.map(([x, y]) => `${x},${y}`).join(" ");
}

function drawChart() {
  if (!model || !currentCase) {
    return;
  }

  const measuredWidth = Math.round(plotWrap.getBoundingClientRect().width);
  const width = Math.max(320, measuredWidth);
  const compact = width < 600;
  const height = compact ? 430 : 520;
  const margin = { top: 34, right: 24, bottom: 68, left: compact ? 62 : 76 };
  const plotLeft = margin.left;
  const plotRight = width - margin.right;
  const plotTop = margin.top;
  const plotBottom = height - margin.bottom;

  const bpValues = model.training_examples.map((example) => example.systolic_bp_mm_hg);
  const heartRateValues = model.training_examples.map(
    (example) => example.max_heart_rate_bpm
  );
  bpValues.push(currentCase.systolicBp);
  heartRateValues.push(currentCase.maxHeartRate);

  const xMinimum = Math.floor((Math.min(...bpValues) - 10) / 10) * 10;
  const xMaximum = Math.ceil((Math.max(...bpValues) + 10) / 10) * 10;
  const yMinimum = Math.floor((Math.min(...heartRateValues) - 10) / 10) * 10;
  const yMaximum = Math.ceil((Math.max(...heartRateValues) + 10) / 10) * 10;
  const dataBounds = { xMin: xMinimum, xMax: xMaximum, yMin: yMinimum, yMax: yMaximum };

  const xPosition = (value) =>
    mapNumber(value, xMinimum, xMaximum, plotLeft, plotRight);
  const yPosition = (value) =>
    mapNumber(value, yMinimum, yMaximum, plotBottom, plotTop);

  decisionChart.setAttribute("viewBox", `0 0 ${width} ${height}`);
  decisionChart.replaceChildren();
  decisionChart.append(
    makeSvgElement(
      "title",
      { id: "decision-chart-title" },
      "Decision Tree blood pressure and heart rate map"
    ),
    makeSvgElement(
      "desc",
      { id: "decision-chart-description" },
      `Synthetic example at systolic blood pressure ${currentCase.systolicBp} millimetres of mercury and maximum heart rate ${currentCase.maxHeartRate} beats per minute. The toy label is ${currentCase.prediction.label_title}. Vertical and horizontal lines show the learned Decision Tree questions.`
    ),
    makeSvgElement("rect", {
      class: "chart-frame",
      "data-chart-frame": "",
      x: plotLeft,
      y: plotTop,
      width: plotRight - plotLeft,
      height: plotBottom - plotTop,
      rx: 8,
    })
  );

  const regions = window.DecisionTree.collectRegions(model.tree, dataBounds);
  for (const region of regions) {
    const x = xPosition(region.xMin);
    const y = yPosition(region.yMax);
    const regionWidth = xPosition(region.xMax) - x;
    const regionHeight = yPosition(region.yMin) - y;
    const kind = region.label === LOW_LABEL ? "low" : "high";
    const isActive = region.leaf_id === currentCase.prediction.leaf_id;
    const rectangle = makeSvgElement("rect", {
      class: `leaf-region leaf-region-${kind}${isActive ? " is-active" : ""}`,
      x,
      y,
      width: regionWidth,
      height: regionHeight,
      "data-leaf-id": region.leaf_id,
    });
    rectangle.append(
      makeSvgElement(
        "title",
        {},
        `${model.labels[region.label]} toy leaf from ${region.xMin} to ${region.xMax} mmHg and ${region.yMin} to ${region.yMax} bpm${isActive ? ". Current input is here." : "."}`
      )
    );
    decisionChart.append(rectangle);

    if (regionWidth > (compact ? 95 : 135) && regionHeight > 72) {
      decisionChart.append(
        makeSvgElement(
          "text",
          {
            class: `leaf-label leaf-label-${kind}`,
            x: x + regionWidth / 2,
            y: y + regionHeight / 2,
            "text-anchor": "middle",
          },
          compact ? (kind === "low" ? "BLUE LEAF" : "RED LEAF") : model.labels[region.label].toUpperCase()
        )
      );
    }
  }

  const tickCount = compact ? 4 : 6;
  for (const tick of numberTicks(xMinimum, xMaximum, tickCount)) {
    const x = xPosition(tick);
    decisionChart.append(
      makeSvgElement("line", {
        class: "chart-grid-line",
        x1: x,
        y1: plotTop,
        x2: x,
        y2: plotBottom,
      }),
      makeSvgElement(
        "text",
        { class: "chart-tick", x, y: plotBottom + 25, "text-anchor": "middle" },
        Math.round(tick)
      )
    );
  }

  for (const tick of numberTicks(yMinimum, yMaximum, tickCount)) {
    const y = yPosition(tick);
    decisionChart.append(
      makeSvgElement("line", {
        class: "chart-grid-line",
        x1: plotLeft,
        y1: y,
        x2: plotRight,
        y2: y,
      }),
      makeSvgElement(
        "text",
        { class: "chart-tick", x: plotLeft - 12, y: y + 4, "text-anchor": "end" },
        Math.round(tick)
      )
    );
  }

  decisionChart.append(
    makeSvgElement(
      "text",
      {
        class: "axis-title",
        "data-axis": "x",
        x: (plotLeft + plotRight) / 2,
        y: height - 15,
        "text-anchor": "middle",
      },
      "Systolic blood pressure (mmHg) →"
    ),
    makeSvgElement(
      "text",
      {
        class: "axis-title",
        "data-axis": "y",
        x: 18,
        y: (plotTop + plotBottom) / 2,
        "text-anchor": "middle",
        transform: `rotate(-90 18 ${(plotTop + plotBottom) / 2})`,
      },
      "Maximum heart rate (bpm) →"
    )
  );

  const activeNodeIds = new Set(currentCase.prediction.path.map((step) => step.node_id));
  const splits = window.DecisionTree.collectSplits(model.tree, dataBounds);
  for (const split of splits) {
    const activeClass = activeNodeIds.has(split.node_id) ? " is-active" : "";
    if (split.feature === "systolic_bp_mm_hg") {
      const x = xPosition(split.value);
      decisionChart.append(
        makeSvgElement("line", {
          class: `tree-split tree-split-vertical${activeClass}`,
          x1: x,
          y1: yPosition(split.from),
          x2: x,
          y2: yPosition(split.to),
        }),
        makeSvgElement(
          "text",
          {
            class: "split-label",
            x: x + 8,
            y: plotTop + 17,
            "text-anchor": "start",
          },
          compact ? `1 · BP ${split.value}` : `SPLIT 1 · BP ≤ ${split.value.toFixed(0)}`
        )
      );
    } else {
      const y = yPosition(split.value);
      decisionChart.append(
        makeSvgElement("line", {
          class: `tree-split tree-split-horizontal${activeClass}`,
          x1: xPosition(split.from),
          y1: y,
          x2: xPosition(split.to),
          y2: y,
        }),
        makeSvgElement(
          "text",
          {
            class: "split-label",
            x: xPosition(split.from) + 9,
            y: y - 9,
            "text-anchor": "start",
          },
          compact ? `2 · HR ${split.value}` : `SPLIT 2 · HEART RATE ≤ ${split.value.toFixed(0)}`
        )
      );
    }
  }

  for (const example of model.training_examples) {
    const x = xPosition(example.systolic_bp_mm_hg);
    const y = yPosition(example.max_heart_rate_bpm);
    const title = `${model.labels[example.label]} synthetic example: ${example.systolic_bp_mm_hg.toFixed(0)} mmHg, ${example.max_heart_rate_bpm.toFixed(0)} bpm`;
    if (example.label === LOW_LABEL) {
      const point = makeSvgElement("circle", {
        class: "training-point training-point-low",
        cx: x,
        cy: y,
        r: 5,
      });
      point.append(makeSvgElement("title", {}, title));
      decisionChart.append(point);
    } else {
      const point = makeSvgElement("rect", {
        class: "training-point training-point-high",
        x: x - 4.5,
        y: y - 4.5,
        width: 9,
        height: 9,
        rx: 1,
      });
      point.append(makeSvgElement("title", {}, title));
      decisionChart.append(point);
    }
  }

  const userX = xPosition(currentCase.systolicBp);
  const userY = yPosition(currentCase.maxHeartRate);
  const userMark = makeSvgElement("g", { class: "user-mark" });
  userMark.append(
    makeSvgElement(
      "title",
      {},
      `Your synthetic case: ${currentCase.systolicBp.toFixed(0)} mmHg, ${currentCase.maxHeartRate.toFixed(0)} bpm. Toy label: ${currentCase.prediction.label_title}.`
    ),
    makeSvgElement("polygon", {
      class: "user-diamond",
      points: polygonPoints([
        [userX, userY - 15],
        [userX + 15, userY],
        [userX, userY + 15],
        [userX - 15, userY],
      ]),
    }),
    makeSvgElement(
      "text",
      {
        class: "user-label",
        x: clamp(userX, plotLeft + 54, plotRight - 54),
        y: userY < plotTop + 46 ? userY + 38 : userY - 25,
        "text-anchor": "middle",
      },
      `YOU · ${currentCase.systolicBp.toFixed(0)} / ${currentCase.maxHeartRate.toFixed(0)}`
    )
  );
  decisionChart.append(userMark);
}

function renderPrediction(systolicBp, maxHeartRate) {
  const prediction = window.DecisionTree.predict(systolicBp, maxHeartRate, model);
  const kind = prediction.label === LOW_LABEL ? "low" : "high";
  currentCase = { systolicBp, maxHeartRate, prediction };

  resultPanel.dataset.result = kind;
  resultName.textContent = prediction.label_title;
  resultExplanation.textContent =
    kind === "low"
      ? "The synthetic case reached a blue leaf. This is a programming result—not permission to discharge a person."
      : "The synthetic case reached a red leaf. This is a programming result—not an ICU admission decision.";
  pathSummary.textContent = describePath(prediction);
  graphSummary.textContent = `${systolicBp.toFixed(0)} mmHg · ${maxHeartRate.toFixed(0)} bpm · ${prediction.label_title} (toy)`;
  renderPath(prediction);
  drawChart();

  resultPanel.classList.remove("is-updating");
  void resultPanel.offsetWidth;
  resultPanel.classList.add("is-updating");
}

function registerWebMcpTool() {
  const context = document.modelContext;
  if (!context?.registerTool) {
    return;
  }

  try {
    Promise.resolve(
      context.registerTool({
        name: "evaluate_synthetic_cardiac_tree",
        title: "Evaluate a synthetic Decision Tree case",
        description:
          "Place teaching-only blood-pressure and heart-rate values on the page, follow the visible Decision Tree path, and return its synthetic target label. Never use this for real clinical triage.",
        inputSchema: {
          type: "object",
          properties: {
            systolic_bp_mm_hg: { type: "number", minimum: 90, maximum: 180 },
            max_heart_rate_bpm: { type: "number", minimum: 60, maximum: 160 },
          },
          required: ["systolic_bp_mm_hg", "max_heart_rate_bpm"],
          additionalProperties: false,
        },
        annotations: { readOnlyHint: false, untrustedContentHint: false },
        execute(input) {
          const systolicBp = Number(input?.systolic_bp_mm_hg);
          const maxHeartRate = Number(input?.max_heart_rate_bpm);
          if (
            !Number.isFinite(systolicBp) ||
            !Number.isFinite(maxHeartRate) ||
            systolicBp < 90 ||
            systolicBp > 180 ||
            maxHeartRate < 60 ||
            maxHeartRate > 160
          ) {
            throw new Error(
              "Systolic blood pressure must be 90–180 mmHg and maximum heart rate must be 60–160 bpm."
            );
          }

          bpInput.value = String(systolicBp);
          heartRateInput.value = String(maxHeartRate);
          const prediction = window.DecisionTree.predict(systolicBp, maxHeartRate, model);
          renderPrediction(systolicBp, maxHeartRate);
          return {
            toy_label: prediction.label_title,
            leaf_id: prediction.leaf_id,
            path: prediction.path.map((step) => ({
              question: `${step.feature_title} <= ${step.threshold} ${step.unit}`,
              answer: step.answer,
            })),
            warning: "Synthetic programming lesson only; never use for real patient care.",
          };
        },
      })
    ).catch((error) => console.error("Could not register the tree tool:", error));
  } catch (error) {
    console.error("Could not register the tree tool:", error);
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const bpIsValid = validateInput(
    bpInput,
    90,
    180,
    "systolic-bp-error",
    "systolic blood pressure"
  );
  const heartRateIsValid = validateInput(
    heartRateInput,
    60,
    160,
    "max-heart-rate-error",
    "maximum heart rate"
  );

  if (!bpIsValid || !heartRateIsValid || !model) {
    return;
  }
  renderPrediction(Number(bpInput.value), Number(heartRateInput.value));
});

// Presets fill the fields, then request a real form submission so they cannot
// bypass the same guardrails used for values typed by a person.
for (const button of presetButtons) {
  button.addEventListener("click", () => {
    bpInput.value = button.dataset.bp;
    heartRateInput.value = button.dataset.heartRate;
    form.requestSubmit();
  });
}

for (const input of [bpInput, heartRateInput]) {
  input.addEventListener("input", () => {
    input.removeAttribute("aria-invalid");
    document.querySelector(`#${input.id}-error`).textContent = "";
  });
}

async function loadModel() {
  try {
    const response = await fetch("model.json");
    if (!response.ok) {
      throw new Error(`Model request failed with status ${response.status}`);
    }
    model = await response.json();
    evaluateButton.disabled = false;
    modelStatus.textContent = `${model.training_examples.length} synthetic examples · depth ${model.max_depth} · Gini impurity`;
    renderPrediction(Number(bpInput.value), Number(heartRateInput.value));
    registerWebMcpTool();
  } catch (error) {
    console.error(error);
    modelStatus.textContent =
      "The model could not load. Run start_app.py instead of opening index.html directly.";
  }
}

loadModel();

if ("ResizeObserver" in window) {
  new ResizeObserver(drawChart).observe(plotWrap);
} else {
  window.addEventListener("resize", drawChart);
}
