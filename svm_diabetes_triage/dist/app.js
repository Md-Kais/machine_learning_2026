"use strict";

const form = document.querySelector("#screening-form");
const glucoseInput = document.querySelector("#glucose");
const bmiInput = document.querySelector("#bmi");
const screenButton = document.querySelector("#screen-button");
// Quick-sample buttons store safe glucose and BMI pairs in data attributes.
const presetButtons = document.querySelectorAll("[data-glucose][data-bmi]");
const modelStatus = document.querySelector("#model-status");
const resultPanel = document.querySelector("#result-panel");
const zoneName = document.querySelector("#zone-name");
const resultMessage = document.querySelector("#result-message");
const scoreValue = document.querySelector("#score-value");
const clinicalReference = document.querySelector("#clinical-reference");
const graphSummary = document.querySelector("#graph-summary");
const plotWrap = document.querySelector("#plot-wrap");
const svmChart = document.querySelector("#svm-chart");

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const ZONE_KIND = {
  "Lower screening signal": "lower",
  "Pre-diabetic / Further Diagnostic Needed": "margin",
  "Higher screening signal": "higher",
};

const ZONE_MESSAGES = {
  lower:
    "This point is beyond the lower SVM margin. It does not rule out diabetes or replace a laboratory test.",
  margin:
    "This point is inside the SVM margin, the model's uncertain zone. Discuss a laboratory result with a qualified health professional.",
  higher:
    "This point is beyond the higher SVM margin. A qualified health professional should interpret and, when needed, confirm the laboratory result.",
};

let model = null;
let currentPoint = null;

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

function ticks(minimum, maximum, count) {
  return Array.from({ length: count }, (_, index) => {
    return minimum + ((maximum - minimum) * index) / (count - 1);
  });
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function validateInput(input, minimum, maximum, errorId, label) {
  const errorElement = document.querySelector(`#${errorId}`);
  const value = Number(input.value);
  let message = "";

  if (input.value.trim() === "") {
    message = `Please enter ${label}.`;
  } else if (!Number.isFinite(value) || value < minimum || value > maximum) {
    message = `${label} must be from ${minimum} to ${maximum}.`;
  }

  input.setAttribute("aria-invalid", String(Boolean(message)));
  errorElement.textContent = message;
  return message === "";
}

function clinicalReferenceText(glucose) {
  if (glucose <= model.clinical_glucose_reference.normal_max) {
    return `${glucose.toFixed(0)} mg/dL is in the fasting-glucose normal reference range (99 or below).`;
  }
  if (glucose <= model.clinical_glucose_reference.prediabetes_max) {
    return `${glucose.toFixed(0)} mg/dL is in the fasting-glucose prediabetes reference range (100–125).`;
  }
  return `${glucose.toFixed(0)} mg/dL is at or above the fasting-glucose diabetes threshold (126); confirmation is usually needed.`;
}

function polygonPoints(points) {
  return points.map(([x, y]) => `${x},${y}`).join(" ");
}

function drawChart() {
  if (!model || !currentPoint) {
    return;
  }

  const measuredWidth = Math.round(plotWrap.getBoundingClientRect().width);
  const width = Math.max(320, measuredWidth);
  const compact = width < 620;
  const height = compact ? 430 : 520;
  const margin = { top: 46, right: 24, bottom: 68, left: compact ? 62 : 76 };
  const plotLeft = margin.left;
  const plotRight = width - margin.right;
  const plotTop = margin.top;
  const plotBottom = height - margin.bottom;

  const glucoseValues = model.training_examples.map(
    (example) => example.fasting_glucose_mg_dl
  );
  const bmiValues = model.training_examples.map((example) => example.bmi);
  glucoseValues.push(currentPoint.glucose, 100, 126);
  bmiValues.push(currentPoint.bmi);

  const xMinimum = Math.max(40, Math.floor((Math.min(...glucoseValues) - 10) / 20) * 20);
  const xMaximum = Math.min(320, Math.ceil((Math.max(...glucoseValues) + 10) / 20) * 20);
  const yMinimum = Math.max(10, Math.floor((Math.min(...bmiValues) - 3) / 5) * 5);
  const yMaximum = Math.min(65, Math.ceil((Math.max(...bmiValues) + 3) / 5) * 5);

  const xPosition = (value) =>
    mapNumber(value, xMinimum, xMaximum, plotLeft, plotRight);
  const yPosition = (value) =>
    mapNumber(value, yMinimum, yMaximum, plotBottom, plotTop);
  const boundaryX = (bmi, score) =>
    xPosition(
      clamp(window.SimpleSvm.glucoseAtScore(bmi, score, model), xMinimum, xMaximum)
    );

  const lowerBottom = boundaryX(yMinimum, model.margin.lower);
  const lowerTop = boundaryX(yMaximum, model.margin.lower);
  const centerBottom = boundaryX(yMinimum, model.margin.center);
  const centerTop = boundaryX(yMaximum, model.margin.center);
  const upperBottom = boundaryX(yMinimum, model.margin.upper);
  const upperTop = boundaryX(yMaximum, model.margin.upper);

  svmChart.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svmChart.replaceChildren();
  svmChart.append(
    makeSvgElement("title", { id: "svm-chart-title" }, "Glucose and BMI SVM margin graph"),
    makeSvgElement(
      "desc",
      { id: "svm-chart-description" },
      `Input at ${currentPoint.glucose} milligrams per decilitre fasting glucose and BMI ${currentPoint.bmi}. It is in the ${currentPoint.prediction.zone} zone. The graph shows training examples, support examples, the learned separator, and its margin.`
    )
  );

  svmChart.append(
    makeSvgElement("rect", {
      class: "chart-frame",
      x: plotLeft,
      y: plotTop,
      width: plotRight - plotLeft,
      height: plotBottom - plotTop,
      rx: 10,
    }),
    makeSvgElement("polygon", {
      class: "zone-area zone-area-lower",
      points: polygonPoints([
        [plotLeft, plotBottom],
        [lowerBottom, plotBottom],
        [lowerTop, plotTop],
        [plotLeft, plotTop],
      ]),
    }),
    makeSvgElement("polygon", {
      class: "zone-area zone-area-margin",
      points: polygonPoints([
        [lowerBottom, plotBottom],
        [upperBottom, plotBottom],
        [upperTop, plotTop],
        [lowerTop, plotTop],
      ]),
    }),
    makeSvgElement("polygon", {
      class: "zone-area zone-area-higher",
      points: polygonPoints([
        [upperBottom, plotBottom],
        [plotRight, plotBottom],
        [plotRight, plotTop],
        [upperTop, plotTop],
      ]),
    })
  );

  const tickCount = compact ? 4 : 6;
  for (const tick of ticks(xMinimum, xMaximum, tickCount)) {
    const x = xPosition(tick);
    svmChart.append(
      makeSvgElement("line", {
        class: "chart-grid-line",
        x1: x,
        y1: plotTop,
        x2: x,
        y2: plotBottom,
      }),
      makeSvgElement(
        "text",
        { class: "chart-tick", x, y: plotBottom + 26, "text-anchor": "middle" },
        Math.round(tick)
      )
    );
  }

  for (const tick of ticks(yMinimum, yMaximum, tickCount)) {
    const y = yPosition(tick);
    svmChart.append(
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
        tick.toFixed(0)
      )
    );
  }

  svmChart.append(
    makeSvgElement(
      "text",
      {
        class: "chart-axis-title",
        x: (plotLeft + plotRight) / 2,
        y: height - 16,
        "text-anchor": "middle",
      },
      "Fasting blood glucose (mg/dL) →"
    ),
    makeSvgElement(
      "text",
      {
        class: "chart-axis-title",
        x: 18,
        y: (plotTop + plotBottom) / 2,
        "text-anchor": "middle",
        transform: `rotate(-90 18 ${(plotTop + plotBottom) / 2})`,
      },
      "Body Mass Index (BMI) →"
    )
  );

  for (const reference of [
    { value: 100, label: compact ? "100" : "100 · prediabetes starts" },
    { value: 126, label: compact ? "126" : "126 · diabetes threshold" },
  ]) {
    if (reference.value >= xMinimum && reference.value <= xMaximum) {
      const x = xPosition(reference.value);
      svmChart.append(
        makeSvgElement("line", {
          class: "clinical-reference-line",
          x1: x,
          y1: plotTop,
          x2: x,
          y2: plotBottom,
        }),
        makeSvgElement(
          "text",
          {
            class: "clinical-reference-label",
            x,
            y: plotTop - 13,
            "text-anchor": reference.value === 100 ? "end" : "start",
          },
          reference.label
        )
      );
    }
  }

  svmChart.append(
    makeSvgElement("line", {
      class: "margin-line",
      x1: lowerBottom,
      y1: plotBottom,
      x2: lowerTop,
      y2: plotTop,
    }),
    makeSvgElement("line", {
      class: "separator-line",
      x1: centerBottom,
      y1: plotBottom,
      x2: centerTop,
      y2: plotTop,
    }),
    makeSvgElement("line", {
      class: "margin-line",
      x1: upperBottom,
      y1: plotBottom,
      x2: upperTop,
      y2: plotTop,
    })
  );

  if (!compact) {
    const labelY = plotBottom - 16;
    svmChart.append(
      makeSvgElement(
        "text",
        {
          class: "region-label",
          x: (plotLeft + lowerBottom) / 2,
          y: labelY,
          "text-anchor": "middle",
        },
        "LOWER SIDE"
      ),
      makeSvgElement(
        "text",
        {
          class: "region-label region-label-margin",
          x: (lowerBottom + upperBottom) / 2,
          y: labelY,
          "text-anchor": "middle",
        },
        "FOLLOW-UP MARGIN"
      ),
      makeSvgElement(
        "text",
        {
          class: "region-label",
          x: (upperBottom + plotRight) / 2,
          y: labelY,
          "text-anchor": "middle",
        },
        "HIGHER SIDE"
      )
    );
  }

  for (const example of model.training_examples) {
    const x = xPosition(example.fasting_glucose_mg_dl);
    const y = yPosition(example.bmi);
    const title = `${example.label === "lower_reference" ? "Lower" : "Higher"} training example: ${example.fasting_glucose_mg_dl.toFixed(0)} mg/dL, BMI ${example.bmi.toFixed(1)}`;

    if (example.is_support_example) {
      const support = makeSvgElement("circle", {
        class: "support-ring",
        cx: x,
        cy: y,
        r: 10,
      });
      support.append(makeSvgElement("title", {}, `Support example. ${title}`));
      svmChart.append(support);
    }

    if (example.label === "lower_reference") {
      const point = makeSvgElement("circle", {
        class: "training-point training-point-lower",
        cx: x,
        cy: y,
        r: 5,
      });
      point.append(makeSvgElement("title", {}, title));
      svmChart.append(point);
    } else {
      const point = makeSvgElement("rect", {
        class: "training-point training-point-higher",
        x: x - 4.5,
        y: y - 4.5,
        width: 9,
        height: 9,
        rx: 1,
      });
      point.append(makeSvgElement("title", {}, title));
      svmChart.append(point);
    }
  }

  const inputX = xPosition(currentPoint.glucose);
  const inputY = yPosition(currentPoint.bmi);
  const separatorX = boundaryX(currentPoint.bmi, model.margin.center);
  svmChart.append(
    makeSvgElement("line", {
      class: "input-distance-line",
      x1: inputX,
      y1: inputY,
      x2: separatorX,
      y2: inputY,
    })
  );

  const inputMark = makeSvgElement("g", { class: "input-mark" });
  inputMark.append(
    makeSvgElement(
      "title",
      {},
      `Your input: ${currentPoint.glucose.toFixed(0)} mg/dL, BMI ${currentPoint.bmi.toFixed(1)}, ${currentPoint.prediction.zone}`
    ),
    makeSvgElement("polygon", {
      class: "input-diamond",
      points: `${inputX},${inputY - 14} ${inputX + 14},${inputY} ${inputX},${inputY + 14} ${inputX - 14},${inputY}`,
    }),
    makeSvgElement(
      "text",
      {
        class: "input-label",
        x: clamp(inputX, plotLeft + 54, plotRight - 54),
        y: inputY < plotTop + 50 ? inputY + 38 : inputY - 25,
        "text-anchor": "middle",
      },
      `YOU · ${currentPoint.glucose.toFixed(0)} / ${currentPoint.bmi.toFixed(1)}`
    )
  );
  svmChart.append(inputMark);
}

function renderPrediction(glucose, bmi) {
  const prediction = window.SimpleSvm.predict(glucose, bmi, model);
  const kind = ZONE_KIND[prediction.zone];
  currentPoint = { glucose, bmi, prediction };

  resultPanel.dataset.zone = kind;
  zoneName.textContent = prediction.zone;
  resultMessage.textContent = ZONE_MESSAGES[kind];
  scoreValue.textContent = prediction.score.toFixed(2);
  clinicalReference.textContent = clinicalReferenceText(glucose);
  graphSummary.textContent = `${glucose.toFixed(0)} mg/dL · BMI ${bmi.toFixed(1)} · ${prediction.zone}`;

  resultPanel.classList.remove("is-updating");
  void resultPanel.offsetWidth;
  resultPanel.classList.add("is-updating");
  drawChart();
}

function registerWebMcpTool() {
  const context = document.modelContext;
  if (!context?.registerTool) {
    return;
  }

  try {
    Promise.resolve(
      context.registerTool({
        name: "screen_diabetes_triage",
        title: "Screen and plot a diabetes triage example",
        description:
          "Place fasting blood glucose and BMI on the page's educational SVM graph, update the visible result, and return the teaching zone. This is not a diagnosis.",
        inputSchema: {
          type: "object",
          properties: {
            fasting_glucose_mg_dl: { type: "number", minimum: 50, maximum: 300 },
            bmi: { type: "number", minimum: 12, maximum: 60 },
          },
          required: ["fasting_glucose_mg_dl", "bmi"],
          additionalProperties: false,
        },
        annotations: { readOnlyHint: false, untrustedContentHint: false },
        execute(input) {
          const glucose = Number(input?.fasting_glucose_mg_dl);
          const bmi = Number(input?.bmi);
          if (
            !Number.isFinite(glucose) ||
            !Number.isFinite(bmi) ||
            glucose < 50 ||
            glucose > 300 ||
            bmi < 12 ||
            bmi > 60
          ) {
            throw new Error("Fasting glucose must be 50–300 mg/dL and BMI must be 12–60.");
          }

          glucoseInput.value = String(glucose);
          bmiInput.value = String(bmi);
          const prediction = window.SimpleSvm.predict(glucose, bmi, model);
          renderPrediction(glucose, bmi);
          return {
            zone: prediction.zone,
            svm_position: Number(prediction.score.toFixed(2)),
            clinical_reference: clinicalReferenceText(glucose),
            warning: "Educational screening demo; not a medical diagnosis.",
          };
        },
      })
    ).catch((error) => console.error("Could not register the screening tool:", error));
  } catch (error) {
    console.error("Could not register the screening tool:", error);
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const glucoseIsValid = validateInput(
    glucoseInput,
    50,
    300,
    "glucose-error",
    "fasting glucose"
  );
  const bmiIsValid = validateInput(bmiInput, 12, 60, "bmi-error", "BMI");

  if (!glucoseIsValid || !bmiIsValid || !model) {
    return;
  }
  renderPrediction(Number(glucoseInput.value), Number(bmiInput.value));
});

// requestSubmit keeps presets on the same checked path as manually typed input.
for (const button of presetButtons) {
  button.addEventListener("click", () => {
    glucoseInput.value = button.dataset.glucose;
    bmiInput.value = button.dataset.bmi;
    form.requestSubmit();
  });
}

for (const input of [glucoseInput, bmiInput]) {
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
    screenButton.disabled = false;
    modelStatus.textContent = `${model.training_examples.length} examples loaded · 2 features · linear SVM`;
    renderPrediction(Number(glucoseInput.value), Number(bmiInput.value));
    registerWebMcpTool();
  } catch (error) {
    console.error(error);
    modelStatus.textContent =
      "The model could not load. Run the site through start_app.py instead of opening index.html directly.";
  }
}

loadModel();

if ("ResizeObserver" in window) {
  new ResizeObserver(drawChart).observe(plotWrap);
} else {
  window.addEventListener("resize", drawChart);
}
