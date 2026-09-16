"use strict";

const form = document.querySelector("#bmi-form");
const heightInput = document.querySelector("#height");
const weightInput = document.querySelector("#weight");
const predictButton = document.querySelector("#predict-button");
// Both preset buttons carry safe height and weight values in data attributes.
const presetButtons = document.querySelectorAll("[data-height][data-weight]");
const resetButton = document.querySelector("#reset-button");
const modelStatus = document.querySelector("#model-status");
const resultCard = document.querySelector("#result-card");
const resultEmpty = document.querySelector("#result-empty");
const resultContent = document.querySelector("#result-content");
const categoryName = document.querySelector("#category-name");
const bmiNumber = document.querySelector("#bmi-number");
const resultMessage = document.querySelector("#result-message");
const scaleMarker = document.querySelector("#scale-marker");
const chartSummary = document.querySelector("#chart-summary");
const plotWrap = document.querySelector("#plot-wrap");
const neighborChart = document.querySelector("#neighbor-chart");
const neighborDetails = document.querySelector("#neighbor-details");

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const categoryColors = {
  Underweight: "#3b8edb",
  Normal: "#36a852",
  Overweight: "#db8b12",
  Obese: "#d84952",
};

const categoryMessages = {
  Underweight:
    "This is below the typical adult BMI range. A qualified health professional can help interpret it with nutrition and health context.",
  Normal:
    "This is inside the typical adult BMI range. Remember that BMI is only one screening measurement, not a complete health picture.",
  Overweight:
    "This is above the typical adult BMI range. BMI cannot see muscle, body composition, or your personal medical history.",
  Obese:
    "This is inside the adult obesity screening range. A qualified health professional can give useful, personal guidance.",
};

let model = null;
let chartState = null;

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

function markerPosition(bmi) {
  const scaleMinimum = 14;
  const scaleMaximum = 42;
  const clampedBmi = Math.min(scaleMaximum, Math.max(scaleMinimum, bmi));
  return ((clampedBmi - scaleMinimum) / (scaleMaximum - scaleMinimum)) * 100;
}

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

function numberTicks(minimum, maximum, count) {
  return Array.from({ length: count }, (_, index) => {
    const position = index / (count - 1);
    return minimum + (maximum - minimum) * position;
  });
}

function mapNumber(value, domainMinimum, domainMaximum, rangeMinimum, rangeMaximum) {
  const position = (value - domainMinimum) / (domainMaximum - domainMinimum);
  return rangeMinimum + position * (rangeMaximum - rangeMinimum);
}

function updateNeighborDetails(prediction) {
  neighborDetails.replaceChildren();

  prediction.neighbors.forEach((neighbor, index) => {
    const item = document.createElement("li");
    const rank = document.createElement("span");
    const measurements = document.createElement("strong");
    const description = document.createElement("small");

    rank.className = "neighbor-rank";
    rank.textContent = String(index + 1);
    measurements.textContent = `${neighbor.height_cm.toFixed(0)} cm × ${neighbor.weight_kg.toFixed(1)} kg`;
    description.textContent = `BMI ${neighbor.bmi.toFixed(1)} · ${neighbor.category}`;

    item.append(rank, measurements, description);
    neighborDetails.append(item);
  });
}

function drawNeighborChart() {
  if (!model || !chartState) {
    return;
  }

  const measuredWidth = Math.round(plotWrap.getBoundingClientRect().width);
  const width = Math.max(320, measuredWidth);
  const height = width < 560 ? 390 : 430;
  const margin = { top: 28, right: 22, bottom: 64, left: 68 };
  const plotLeft = margin.left;
  const plotRight = width - margin.right;
  const plotTop = margin.top;
  const plotBottom = height - margin.bottom;

  const allHeights = model.training_examples.map((example) => example.height_cm);
  const allWeights = model.training_examples.map((example) => example.weight_kg);
  allHeights.push(chartState.height);
  allWeights.push(chartState.weight);

  const rawXMinimum = Math.min(...allHeights);
  const rawXMaximum = Math.max(...allHeights);
  const xMinimum = Math.floor((rawXMinimum - 5) / 10) * 10;
  const xMaximum = Math.ceil((rawXMaximum + 5) / 10) * 10;

  const rawYMinimum = Math.min(...allWeights);
  const rawYMaximum = Math.max(...allWeights);
  const yPadding = Math.max(8, (rawYMaximum - rawYMinimum) * 0.08);
  const yStep = rawYMaximum > 200 ? 50 : 20;
  const yMinimum = Math.max(0, Math.floor((rawYMinimum - yPadding) / yStep) * yStep);
  const yMaximum = Math.ceil((rawYMaximum + yPadding) / yStep) * yStep;

  const xPosition = (value) =>
    mapNumber(value, xMinimum, xMaximum, plotLeft, plotRight);
  const yPosition = (value) =>
    mapNumber(value, yMinimum, yMaximum, plotBottom, plotTop);

  neighborChart.setAttribute("viewBox", `0 0 ${width} ${height}`);
  neighborChart.replaceChildren();
  neighborChart.append(
    makeSvgElement("title", { id: "neighbor-chart-title" }, "Height and weight neighbor chart"),
    makeSvgElement(
      "desc",
      { id: "neighbor-chart-description" },
      `${chartState.isExample ? "Example" : "User"} point at ${chartState.height} centimetres and ${chartState.weight} kilograms. Five numbered dataset examples are connected as the nearest BMI neighbors.`
    )
  );

  neighborChart.append(
    makeSvgElement("rect", {
      class: "chart-frame",
      x: plotLeft,
      y: plotTop,
      width: plotRight - plotLeft,
      height: plotBottom - plotTop,
      rx: 12,
    })
  );

  const tickCount = width < 500 ? 4 : 6;
  for (const tick of numberTicks(xMinimum, xMaximum, tickCount)) {
    const x = xPosition(tick);
    neighborChart.append(
      makeSvgElement("line", {
        class: "chart-grid-line",
        x1: x,
        y1: plotTop,
        x2: x,
        y2: plotBottom,
      }),
      makeSvgElement(
        "text",
        { class: "chart-tick", x, y: plotBottom + 24, "text-anchor": "middle" },
        Math.round(tick)
      )
    );
  }

  for (const tick of numberTicks(yMinimum, yMaximum, tickCount)) {
    const y = yPosition(tick);
    neighborChart.append(
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

  neighborChart.append(
    makeSvgElement(
      "text",
      {
        class: "chart-axis-title",
        x: (plotLeft + plotRight) / 2,
        y: height - 14,
        "text-anchor": "middle",
      },
      "Height (cm) →"
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
      "Weight (kg) →"
    )
  );

  for (const example of model.training_examples) {
    neighborChart.append(
      makeSvgElement("circle", {
        class: "training-point",
        cx: xPosition(example.height_cm),
        cy: yPosition(example.weight_kg),
        r: 4.5,
        fill: categoryColors[example.category],
      })
    );
  }

  const userX = xPosition(chartState.height);
  const userY = yPosition(chartState.weight);
  const compactNeighborLabels = width < 560;

  chartState.prediction.neighbors.forEach((neighbor, index) => {
    const neighborX = xPosition(neighbor.height_cm);
    const neighborY = yPosition(neighbor.weight_kg);
    const labelOnRight = neighborX < (plotLeft + plotRight) / 2;
    const labelX = neighborX + (labelOnRight ? 23 : -23);
    const labelY = Math.min(
      plotBottom - (compactNeighborLabels ? 8 : 20),
      Math.max(plotTop + 16, neighborY + (index % 2 === 0 ? -22 : 30))
    );
    const labelAnchor = labelOnRight ? "start" : "end";

    neighborChart.append(
      makeSvgElement("line", {
        class: "neighbor-connection",
        x1: userX,
        y1: userY,
        x2: neighborX,
        y2: neighborY,
        style: `--neighbor-index: ${index}`,
      })
    );

    const mark = makeSvgElement("g", {
      class: "neighbor-mark",
      style: `--neighbor-index: ${index}`,
    });
    mark.append(
      makeSvgElement(
        "title",
        {},
        `Neighbor ${index + 1}: ${neighbor.height_cm.toFixed(0)} centimetres, ${neighbor.weight_kg.toFixed(1)} kilograms, BMI ${neighbor.bmi.toFixed(1)}, ${neighbor.category}`
      ),
      makeSvgElement("circle", {
        class: "neighbor-halo",
        cx: neighborX,
        cy: neighborY,
        r: 20,
      }),
      makeSvgElement("circle", {
        class: "neighbor-core",
        cx: neighborX,
        cy: neighborY,
        r: 13,
        fill: categoryColors[neighbor.category],
      }),
      makeSvgElement(
        "text",
        {
          class: "neighbor-number",
          x: neighborX,
          y: neighborY + 4,
          "text-anchor": "middle",
        },
        String(index + 1)
      )
    );
    neighborChart.append(mark);

    const coordinateLabel = makeSvgElement("text", {
      class: "neighbor-coordinate",
      x: labelX,
      y: labelY,
      "text-anchor": labelAnchor,
      style: `--neighbor-index: ${index}`,
    });

    if (compactNeighborLabels) {
      coordinateLabel.textContent = `#${index + 1} ${neighbor.height_cm.toFixed(0)}×${neighbor.weight_kg.toFixed(0)}`;
    } else {
      coordinateLabel.append(
        makeSvgElement(
          "tspan",
          { class: "neighbor-coordinate-rank", x: labelX, dy: 0 },
          `#${index + 1} · ${neighbor.height_cm.toFixed(0)} cm`
        ),
        makeSvgElement(
          "tspan",
          { x: labelX, dy: 15 },
          `${neighbor.weight_kg.toFixed(1)} kg · BMI ${neighbor.bmi.toFixed(1)}`
        )
      );
    }

    neighborChart.append(coordinateLabel);
  });

  const userMark = makeSvgElement("g", { class: "user-mark" });
  userMark.append(
    makeSvgElement("polygon", {
      class: "user-diamond",
      points: `${userX},${userY - 15} ${userX + 15},${userY} ${userX},${userY + 15} ${userX - 15},${userY}`,
    }),
    makeSvgElement(
      "text",
      {
        class: "user-label",
        x: Math.min(plotRight - 18, Math.max(plotLeft + 18, userX)),
        y: userY < plotTop + 42 ? userY + 36 : userY - 24,
        "text-anchor": "middle",
      },
      chartState.isExample ? "EXAMPLE" : "YOU"
    )
  );
  neighborChart.append(userMark);
}

function renderNeighborChart(height, weight, bmi, prediction, isExample = false) {
  chartState = { height, weight, bmi, prediction, isExample };
  const personLabel = isExample ? "Example person" : "Your point";
  chartSummary.textContent = `${personLabel} · ${height.toFixed(0)} cm · ${weight.toFixed(1)} kg · BMI ${bmi.toFixed(1)} · ${prediction.category}`;
  updateNeighborDetails(prediction);
  drawNeighborChart();
}

function renderExampleChart() {
  if (!model) {
    return;
  }
  const height = 170;
  const weight = 65;
  const bmi = window.BmiKnn.calculateBmi(height, weight);
  const prediction = window.BmiKnn.predictWithKnn(bmi, model);
  renderNeighborChart(height, weight, bmi, prediction, true);
}

function showResult(height, weight, bmi, prediction) {
  categoryName.textContent = prediction.category;
  bmiNumber.textContent = bmi.toFixed(1);
  resultMessage.textContent = categoryMessages[prediction.category];
  resultCard.dataset.category = prediction.category;
  scaleMarker.style.setProperty("--marker-position", `${markerPosition(bmi)}%`);

  renderNeighborChart(height, weight, bmi, prediction);

  resultEmpty.hidden = true;
  resultContent.hidden = false;
  resultContent.classList.remove("is-entering");
  void resultContent.offsetWidth;
  resultContent.classList.add("is-entering");

  if (window.matchMedia("(max-width: 900px)").matches) {
    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function resetResult() {
  form.reset();
  resultContent.hidden = true;
  resultEmpty.hidden = false;
  resultCard.removeAttribute("data-category");
  heightInput.removeAttribute("aria-invalid");
  weightInput.removeAttribute("aria-invalid");
  document.querySelector("#height-error").textContent = "";
  document.querySelector("#weight-error").textContent = "";
  renderExampleChart();
  heightInput.focus();
}

function registerWebMcpTool() {
  const context = document.modelContext;
  if (!context?.registerTool) {
    return;
  }

  try {
    Promise.resolve(
      context.registerTool({
        name: "predict_bmi_category",
        title: "Predict BMI category",
        description:
          "Use the page's KNN model to predict an adult BMI category from height in centimetres and weight in kilograms, then show the same result in the page.",
        inputSchema: {
          type: "object",
          properties: {
            height_cm: { type: "number", minimum: 100, maximum: 250 },
            weight_kg: { type: "number", minimum: 25, maximum: 350 },
          },
          required: ["height_cm", "weight_kg"],
          additionalProperties: false,
        },
        annotations: { readOnlyHint: false, untrustedContentHint: false },
        execute(input) {
          const height = Number(input?.height_cm);
          const weight = Number(input?.weight_kg);

          if (
            !Number.isFinite(height) ||
            !Number.isFinite(weight) ||
            height < 100 ||
            height > 250 ||
            weight < 25 ||
            weight > 350
          ) {
            throw new Error(
              "Height must be 100–250 cm and weight must be 25–350 kg."
            );
          }

          heightInput.value = String(height);
          weightInput.value = String(weight);
          const bmi = window.BmiKnn.calculateBmi(height, weight);
          const prediction = window.BmiKnn.predictWithKnn(bmi, model);
          showResult(height, weight, bmi, prediction);

          return {
            bmi: Number(bmi.toFixed(1)),
            category: prediction.category,
            neighbor_bmis: prediction.neighbors.map((neighbor) =>
              Number(neighbor.bmi.toFixed(1))
            ),
          };
        },
      })
    ).catch((error) => console.error("Could not register the BMI tool:", error));
  } catch (error) {
    console.error("Could not register the BMI tool:", error);
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();

  const heightIsValid = validateInput(heightInput, 100, 250, "height-error", "Height");
  const weightIsValid = validateInput(weightInput, 25, 350, "weight-error", "Weight");

  if (!heightIsValid || !weightIsValid || !model) {
    return;
  }

  const height = Number(heightInput.value);
  const weight = Number(weightInput.value);
  const bmi = window.BmiKnn.calculateBmi(height, weight);
  showResult(height, weight, bmi, window.BmiKnn.predictWithKnn(bmi, model));
});

// A preset is only a fast way to fill the form; requestSubmit deliberately
// sends it through the exact same validation and prediction path as typed data.
for (const button of presetButtons) {
  button.addEventListener("click", () => {
    heightInput.value = button.dataset.height;
    weightInput.value = button.dataset.weight;
    form.requestSubmit();
  });
}

resetButton.addEventListener("click", resetResult);

for (const input of [heightInput, weightInput]) {
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
    predictButton.disabled = false;
    modelStatus.textContent = `${model.training_examples.length} examples loaded · K = ${model.k}`;
    renderExampleChart();
    registerWebMcpTool();
  } catch (error) {
    console.error(error);
    modelStatus.textContent =
      "The model could not load. Run the site through a local web server instead of opening the HTML file directly.";
  }
}

loadModel();

// When the app is served from localhost or a host, this worker saves every
// required local file. Later visits can run without an internet connection.
if ("serviceWorker" in navigator && location.protocol !== "file:") {
  navigator.serviceWorker.register("./sw.js").catch((error) => {
    console.warn("Offline cache could not start", error);
  });
}

if ("ResizeObserver" in window) {
  new ResizeObserver(drawNeighborChart).observe(plotWrap);
} else {
  window.addEventListener("resize", drawNeighborChart);
}
