(() => {
  "use strict";

  const canvas = document.getElementById("perceptronCanvas");
  const context = canvas.getContext("2d");

  const elements = {
    classMode: document.getElementById("classMode"),
    modeLabel: document.getElementById("modeLabel"),
    modeHint: document.getElementById("modeHint"),
    pointCount: document.getElementById("pointCount"),
    presetSelect: document.getElementById("presetSelect"),
    presetButton: document.getElementById("presetButton"),
    trainButton: document.getElementById("trainButton"),
    stepButton: document.getElementById("stepButton"),
    resetButton: document.getElementById("resetButton"),
    speedSlider: document.getElementById("speedSlider"),
    speedValue: document.getElementById("speedValue"),
    errorMessage: document.getElementById("errorMessage"),
    modelBadge: document.getElementById("modelBadge"),
    coordinateReadout: document.getElementById("coordinateReadout"),
    weightsValue: document.getElementById("weightsValue"),
    biasValue: document.getElementById("biasValue"),
    pointValue: document.getElementById("pointValue"),
    netValue: document.getElementById("netValue"),
    errorValue: document.getElementById("errorValue"),
    outputValue: document.getElementById("outputValue"),
    accuracyValue: document.getElementById("accuracyValue"),
    accuracyTrack: document.getElementById("accuracyTrack"),
    accuracyBar: document.getElementById("accuracyBar"),
    boundaryEquation: document.getElementById("boundaryEquation"),
    stepCounter: document.getElementById("stepCounter"),
    narrationBadge: document.getElementById("narrationBadge"),
    narrationText: document.getElementById("narrationText"),
    epochValue: document.getElementById("epochValue"),
  };

  const COLORS = {
    ink: "#14243a",
    inkSoft: "#60748b",
    grid: "#dce5ee",
    axis: "#8194a9",
    blue: "#246bfd",
    pass: "#18a66a",
    passDark: "#08784a",
    passShade: "rgba(24, 166, 106, 0.11)",
    fail: "#e45353",
    failDark: "#b92f3c",
    white: "#ffffff",
  };

  const MAX_POINTS = 50;
  const ZERO_MODEL = Object.freeze({ W: Object.freeze([0, 0]), b: 0 });
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  const presets = {
    separable: [
      { p1: 1.0, p2: 2.0, target: 0 },
      { p1: 2.0, p2: 1.0, target: 0 },
      { p1: 3.0, p2: 3.0, target: 0 },
      { p1: 4.0, p2: 2.5, target: 0 },
      { p1: 2.5, p2: 4.0, target: 0 },
      { p1: 5.0, p2: 7.0, target: 1 },
      { p1: 6.0, p2: 6.0, target: 1 },
      { p1: 7.0, p2: 5.0, target: 1 },
      { p1: 8.0, p2: 8.0, target: 1 },
      { p1: 9.0, p2: 6.5, target: 1 },
    ],
    edge: [
      { p1: 1.0, p2: 4.0, target: 0 },
      { p1: 3.0, p2: 3.0, target: 0 },
      { p1: 4.0, p2: 2.0, target: 0 },
      { p1: 5.0, p2: 1.0, target: 0 },
      { p1: 2.0, p2: 5.0, target: 0 },
      { p1: 2.0, p2: 8.0, target: 1 },
      { p1: 4.0, p2: 6.0, target: 1 },
      { p1: 6.0, p2: 4.0, target: 1 },
      { p1: 7.0, p2: 3.0, target: 1 },
      { p1: 8.0, p2: 2.0, target: 1 },
    ],
    xor: [
      { p1: 2.0, p2: 2.0, target: 0 },
      { p1: 8.0, p2: 8.0, target: 0 },
      { p1: 2.0, p2: 8.0, target: 1 },
      { p1: 8.0, p2: 2.0, target: 1 },
    ],
  };

  let points = [];
  let model = cloneModel(ZERO_MODEL);
  let viewport = { width: 920, height: 680 };
  let pointerFeature = null;
  let keyboardCursor = { p1: 5, p2: 5, visible: false };
  let activeStep = null;
  let trainingResult = null;
  let trainingFingerprint = "";
  let stepCursor = 0;
  let animationToken = 0;
  let isBusy = false;

  function cloneModel(source) {
    return { W: [Number(source.W[0]), Number(source.W[1])], b: Number(source.b) };
  }

  function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function roundFeature(value) {
    return Math.round(clamp(value, 0, 10) * 100) / 100;
  }

  function plotBounds() {
    const compact = viewport.width < 560;
    const left = compact ? 55 : 72;
    const right = compact ? 18 : 28;
    const top = compact ? 37 : 43;
    const bottom = compact ? 58 : 66;
    return {
      left,
      top,
      right: viewport.width - right,
      bottom: viewport.height - bottom,
      width: viewport.width - left - right,
      height: viewport.height - top - bottom,
    };
  }

  function featureToPixel(p1, p2) {
    const plot = plotBounds();
    return {
      x: plot.left + (p1 / 10) * plot.width,
      y: plot.bottom - (p2 / 10) * plot.height,
    };
  }

  function pixelToFeature(x, y) {
    const plot = plotBounds();
    return {
      p1: roundFeature(((x - plot.left) / plot.width) * 10),
      p2: roundFeature(((plot.bottom - y) / plot.height) * 10),
    };
  }

  function isInsidePlot(x, y) {
    const plot = plotBounds();
    return x >= plot.left && x <= plot.right && y >= plot.top && y <= plot.bottom;
  }

  function scoreAt(point, parameters = model) {
    return parameters.W[0] * point.p1 + parameters.W[1] * point.p2 + parameters.b;
  }

  function hasBoundary(parameters = model) {
    return Math.hypot(parameters.W[0], parameters.W[1]) > 1e-10;
  }

  function positiveRegionPolygon() {
    if (!hasBoundary()) {
      return [];
    }

    const square = [
      { p1: 0, p2: 0 },
      { p1: 10, p2: 0 },
      { p1: 10, p2: 10 },
      { p1: 0, p2: 10 },
    ];
    const clipped = [];

    for (let index = 0; index < square.length; index += 1) {
      const current = square[index];
      const next = square[(index + 1) % square.length];
      const currentScore = scoreAt(current);
      const nextScore = scoreAt(next);
      const currentInside = currentScore >= -1e-10;
      const nextInside = nextScore >= -1e-10;

      if (currentInside) {
        clipped.push(current);
      }
      if (currentInside !== nextInside) {
        const ratio = currentScore / (currentScore - nextScore);
        clipped.push({
          p1: current.p1 + ratio * (next.p1 - current.p1),
          p2: current.p2 + ratio * (next.p2 - current.p2),
        });
      }
    }
    return clipped;
  }

  function boundarySegment() {
    if (!hasBoundary()) {
      return [];
    }

    const [w1, w2] = model.W;
    const candidates = [];
    if (Math.abs(w2) > 1e-10) {
      candidates.push({ p1: 0, p2: -model.b / w2 });
      candidates.push({ p1: 10, p2: -(10 * w1 + model.b) / w2 });
    }
    if (Math.abs(w1) > 1e-10) {
      candidates.push({ p1: -model.b / w1, p2: 0 });
      candidates.push({ p1: -(10 * w2 + model.b) / w1, p2: 10 });
    }

    const inside = candidates.filter(
      (point) =>
        point.p1 >= -1e-8 &&
        point.p1 <= 10 + 1e-8 &&
        point.p2 >= -1e-8 &&
        point.p2 <= 10 + 1e-8,
    );
    const unique = [];
    for (const point of inside) {
      const normalized = { p1: clamp(point.p1, 0, 10), p2: clamp(point.p2, 0, 10) };
      if (!unique.some((item) => Math.hypot(item.p1 - normalized.p1, item.p2 - normalized.p2) < 1e-6)) {
        unique.push(normalized);
      }
    }
    return unique.slice(0, 2);
  }

  function drawGrid() {
    const plot = plotBounds();
    const labelInterval = viewport.width < 520 ? 2 : 1;

    context.save();
    context.fillStyle = "#fbfdff";
    context.fillRect(plot.left, plot.top, plot.width, plot.height);

    context.lineWidth = 1;
    context.strokeStyle = COLORS.grid;
    context.font = "600 12px Inter, system-ui, sans-serif";
    context.fillStyle = COLORS.inkSoft;

    for (let tick = 0; tick <= 10; tick += 1) {
      const vertical = featureToPixel(tick, 0);
      const horizontal = featureToPixel(0, tick);

      context.beginPath();
      context.moveTo(vertical.x, plot.top);
      context.lineTo(vertical.x, plot.bottom);
      context.stroke();

      context.beginPath();
      context.moveTo(plot.left, horizontal.y);
      context.lineTo(plot.right, horizontal.y);
      context.stroke();

      if (tick % labelInterval === 0) {
        context.textAlign = "center";
        context.textBaseline = "top";
        context.fillText(String(tick), vertical.x, plot.bottom + 8);
        context.textAlign = "right";
        context.textBaseline = "middle";
        context.fillText(String(tick), plot.left - 10, horizontal.y);
      }
    }

    context.strokeStyle = COLORS.axis;
    context.lineWidth = 1.5;
    context.strokeRect(plot.left, plot.top, plot.width, plot.height);

    context.fillStyle = COLORS.ink;
    context.font = "700 13px Inter, system-ui, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "bottom";
    context.fillText("Study hours per week (p₁)", plot.left + plot.width / 2, viewport.height - 8);

    context.save();
    context.translate(viewport.width < 520 ? 13 : 16, plot.top + plot.height / 2);
    context.rotate(-Math.PI / 2);
    context.textAlign = "center";
    context.textBaseline = "top";
    context.fillText("Attendance / 10 (p₂)", 0, 0);
    context.restore();

    context.fillStyle = COLORS.inkSoft;
    context.font = "600 12px Inter, system-ui, sans-serif";
    context.textAlign = "left";
    context.textBaseline = "bottom";
    context.fillText("10 = 100% attendance", plot.left + 9, plot.top - 10);
    context.restore();
  }

  function drawPositiveRegion() {
    const polygon = positiveRegionPolygon();
    if (polygon.length < 3) {
      return;
    }

    context.save();
    context.beginPath();
    polygon.forEach((point, index) => {
      const pixel = featureToPixel(point.p1, point.p2);
      if (index === 0) {
        context.moveTo(pixel.x, pixel.y);
      } else {
        context.lineTo(pixel.x, pixel.y);
      }
    });
    context.closePath();
    context.fillStyle = COLORS.passShade;
    context.fill();
    context.restore();
  }

  function drawArrow(from, to) {
    const angle = Math.atan2(to.y - from.y, to.x - from.x);
    const headLength = 10;

    context.beginPath();
    context.moveTo(from.x, from.y);
    context.lineTo(to.x, to.y);
    context.stroke();

    context.beginPath();
    context.moveTo(to.x, to.y);
    context.lineTo(
      to.x - headLength * Math.cos(angle - Math.PI / 6),
      to.y - headLength * Math.sin(angle - Math.PI / 6),
    );
    context.lineTo(
      to.x - headLength * Math.cos(angle + Math.PI / 6),
      to.y - headLength * Math.sin(angle + Math.PI / 6),
    );
    context.closePath();
    context.fill();
  }

  function drawBoundaryAndWeight() {
    const segment = boundarySegment();
    if (segment.length < 2) {
      return;
    }

    const start = featureToPixel(segment[0].p1, segment[0].p2);
    const end = featureToPixel(segment[1].p1, segment[1].p2);
    const plot = plotBounds();

    context.save();
    context.beginPath();
    context.rect(plot.left, plot.top, plot.width, plot.height);
    context.clip();

    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.strokeStyle = "rgba(255, 255, 255, 0.92)";
    context.lineWidth = 7;
    context.stroke();

    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.strokeStyle = COLORS.blue;
    context.lineWidth = 3;
    context.stroke();

    const boundaryMidpoint = {
      p1: (segment[0].p1 + segment[1].p1) / 2,
      p2: (segment[0].p2 + segment[1].p2) / 2,
    };
    const magnitude = Math.hypot(model.W[0], model.W[1]);
    const arrowScale = viewport.width < 520 ? 1.0 : 1.35;
    const arrowTip = {
      p1: boundaryMidpoint.p1 + (model.W[0] / magnitude) * arrowScale,
      p2: boundaryMidpoint.p2 + (model.W[1] / magnitude) * arrowScale,
    };
    const arrowStartPixel = featureToPixel(boundaryMidpoint.p1, boundaryMidpoint.p2);
    const arrowEndPixel = featureToPixel(arrowTip.p1, arrowTip.p2);

    context.strokeStyle = COLORS.blue;
    context.fillStyle = COLORS.blue;
    context.lineWidth = 3;
    drawArrow(arrowStartPixel, arrowEndPixel);

    context.font = "800 13px Inter, system-ui, sans-serif";
    context.textAlign = "left";
    context.textBaseline = "bottom";
    context.fillText("w", arrowEndPixel.x + 6, arrowEndPixel.y - 5);
    context.restore();
  }

  function drawStudent(point, index) {
    const pixel = featureToPixel(point.p1, point.p2);
    const isPass = point.target === 1;
    const isActive = activeStep && activeStep.sample_index === index;

    context.save();
    context.shadowColor = "rgba(20, 36, 58, 0.22)";
    context.shadowBlur = 7;
    context.shadowOffsetY = 2;
    context.beginPath();
    context.arc(pixel.x, pixel.y, isActive ? 8.5 : 7, 0, Math.PI * 2);
    context.fillStyle = isPass ? COLORS.pass : COLORS.fail;
    context.fill();
    context.shadowColor = "transparent";
    context.strokeStyle = COLORS.white;
    context.lineWidth = 2.5;
    context.stroke();

    if (isActive) {
      context.beginPath();
      context.arc(pixel.x, pixel.y, 13, 0, Math.PI * 2);
      context.strokeStyle = isPass ? COLORS.passDark : COLORS.failDark;
      context.lineWidth = 2;
      context.setLineDash([4, 3]);
      context.stroke();
    }
    context.restore();
  }

  function drawKeyboardCursor() {
    if (!keyboardCursor.visible) {
      return;
    }
    const pixel = featureToPixel(keyboardCursor.p1, keyboardCursor.p2);
    context.save();
    context.strokeStyle = COLORS.ink;
    context.lineWidth = 1.5;
    context.setLineDash([3, 3]);
    context.beginPath();
    context.arc(pixel.x, pixel.y, 10, 0, Math.PI * 2);
    context.moveTo(pixel.x - 15, pixel.y);
    context.lineTo(pixel.x + 15, pixel.y);
    context.moveTo(pixel.x, pixel.y - 15);
    context.lineTo(pixel.x, pixel.y + 15);
    context.stroke();
    context.restore();
  }

  function drawEmptyHint() {
    if (points.length !== 0) {
      return;
    }
    const plot = plotBounds();
    context.save();
    context.fillStyle = "rgba(255, 255, 255, 0.92)";
    context.strokeStyle = "#cbd8e5";
    context.lineWidth = 1;
    const width = Math.min(285, plot.width - 30);
    const height = 76;
    const x = plot.left + (plot.width - width) / 2;
    const y = plot.top + (plot.height - height) / 2;
    context.beginPath();
    context.roundRect(x, y, width, height, 12);
    context.fill();
    context.stroke();
    context.fillStyle = COLORS.ink;
    context.font = "800 14px Inter, system-ui, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText("Click anywhere to place a student", x + width / 2, y + 27);
    context.fillStyle = COLORS.inkSoft;
    context.font = "600 12px Inter, system-ui, sans-serif";
    context.fillText("or load one of the example data sets", x + width / 2, y + 49);
    context.restore();
  }

  function draw() {
    context.clearRect(0, 0, viewport.width, viewport.height);
    context.fillStyle = "#f9fbfd";
    context.fillRect(0, 0, viewport.width, viewport.height);
    drawGrid();
    drawPositiveRegion();
    drawBoundaryAndWeight();
    points.forEach(drawStudent);
    drawKeyboardCursor();
    drawEmptyHint();
  }

  function resizeCanvas() {
    const bounds = canvas.getBoundingClientRect();
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, bounds.width);
    const height = Math.max(1, bounds.height);

    canvas.width = Math.round(width * pixelRatio);
    canvas.height = Math.round(height * pixelRatio);
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    viewport = { width, height };
    draw();
  }

  function updatePointCount() {
    const count = points.length;
    elements.pointCount.textContent = `${count} ${count === 1 ? "student" : "students"}`;
  }

  function updateModePresentation() {
    const failingMode = elements.classMode.checked;
    elements.modeLabel.textContent = failingMode
      ? "Adding failing students"
      : "Adding passing students";
    elements.modeHint.textContent = failingMode ? "Red · target 0" : "Green · target 1";
    elements.classMode.closest(".class-switch").classList.toggle("is-fail", failingMode);
  }

  function setBadge(kind, label) {
    elements.modelBadge.className = `status-badge status-badge--${kind}`;
    elements.modelBadge.innerHTML = '<span class="status-badge__dot" aria-hidden="true"></span>';
    elements.modelBadge.append(document.createTextNode(label));
  }

  function setNarrationBadge(label, kind = "") {
    elements.narrationBadge.className = kind ? `mini-badge mini-badge--${kind}` : "mini-badge";
    elements.narrationBadge.textContent = label;
  }

  function clearError() {
    elements.errorMessage.hidden = true;
    elements.errorMessage.textContent = "";
  }

  function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorMessage.hidden = false;
    setBadge("warning", "Needs attention");
    setNarrationBadge("Check the data", "warning");
    elements.narrationText.textContent = message;
  }

  function formatNumber(value, digits = 3) {
    if (!Number.isFinite(value)) {
      return "—";
    }
    const normalized = Math.abs(value) < 0.5 * 10 ** -digits ? 0 : value;
    return normalized.toFixed(digits);
  }

  function formatBoundary() {
    if (!hasBoundary()) {
      return "Boundary appears after the first correction.";
    }
    const [w1, w2] = model.W;
    const bSign = model.b >= 0 ? "+" : "−";
    const w2Sign = w2 >= 0 ? "+" : "−";
    return `${formatNumber(w1, 2)}p₁ ${w2Sign} ${formatNumber(Math.abs(w2), 2)}p₂ ${bSign} ${formatNumber(Math.abs(model.b), 2)} = 0`;
  }

  function updateTelemetry(step = null, accuracy = null) {
    elements.weightsValue.textContent = `[${formatNumber(model.W[0])}, ${formatNumber(model.W[1])}]`;
    elements.biasValue.textContent = formatNumber(model.b);
    elements.boundaryEquation.textContent = formatBoundary();

    if (step) {
      elements.pointValue.textContent = `[${formatNumber(step.p[0], 2)}, ${formatNumber(step.p[1], 2)}]`;
      elements.netValue.textContent = formatNumber(step.n);
      elements.errorValue.textContent = String(step.e);
      elements.outputValue.textContent = `${step.a} · ${step.a === 1 ? "Pass" : "Fail"}`;
      elements.stepCounter.textContent = `Step ${step.global_step}`;
      elements.epochValue.textContent = `Epoch ${step.epoch}`;
    } else {
      elements.pointValue.textContent = "[—, —]";
      elements.netValue.textContent = "—";
      elements.errorValue.textContent = "—";
      elements.outputValue.textContent = "—";
      elements.stepCounter.textContent = "Step 0";
      elements.epochValue.textContent = "Epoch 0";
    }

    const resolvedAccuracy = accuracy ?? (step ? step.accuracy : null);
    const percent = resolvedAccuracy === null ? 0 : Math.round(resolvedAccuracy * 100);
    elements.accuracyValue.textContent = resolvedAccuracy === null ? "—" : `${percent}%`;
    elements.accuracyBar.style.width = `${percent}%`;
    elements.accuracyTrack.setAttribute("aria-valuenow", String(percent));
  }

  function narrateStep(step) {
    const hours = formatNumber(step.p[0], 1);
    const attendance = Math.round(step.p[1] * 10);
    const profile = `${hours} study hours and ${attendance}% attendance`;

    if (step.e === 1) {
      setNarrationBadge("Correcting upward", "warning");
      elements.narrationText.textContent =
        `This student has ${profile} and should pass, but the model predicted Fail. ` +
        "It adds the student's feature vector, moving the passing side toward this point.";
    } else if (step.e === -1) {
      setNarrationBadge("Correcting downward", "warning");
      elements.narrationText.textContent =
        `This student has ${profile} and should fail, but the model predicted Pass. ` +
        "It subtracts the student's feature vector, moving the passing side away from this point.";
    } else {
      setNarrationBadge("Correct already", "success");
      elements.narrationText.textContent =
        `The model classified the student with ${profile} correctly. Error is zero, so the line stays put.`;
    }
  }

  function cancelPlayback() {
    animationToken += 1;
    isBusy = false;
    setButtonsDisabled(false);
  }

  function invalidateTraining(message) {
    cancelPlayback();
    trainingResult = null;
    trainingFingerprint = "";
    stepCursor = 0;
    activeStep = null;
    model = cloneModel(ZERO_MODEL);
    clearError();
    updateTelemetry();
    setBadge("ready", "Ready to train");
    setNarrationBadge("Data changed");
    elements.narrationText.textContent = message;
    draw();
  }

  function addPoint(point, message) {
    if (points.length >= MAX_POINTS) {
      showError(`The lab supports at most ${MAX_POINTS} students at a time.`);
      return;
    }
    points.push({
      p1: roundFeature(point.p1),
      p2: roundFeature(point.p2),
      target: point.target === 1 ? 1 : 0,
    });
    updatePointCount();
    invalidateTraining(message || "Student added. The model is ready to learn from the new data set.");
  }

  function canvasCoordinates(event) {
    const bounds = canvas.getBoundingClientRect();
    return {
      x: event.clientX - bounds.left,
      y: event.clientY - bounds.top,
    };
  }

  function handleCanvasClick(event) {
    if (event.button !== 0) {
      return;
    }
    const pixel = canvasCoordinates(event);
    if (!isInsidePlot(pixel.x, pixel.y)) {
      return;
    }
    const feature = pixelToFeature(pixel.x, pixel.y);
    addPoint(
      { ...feature, target: elements.classMode.checked ? 0 : 1 },
      `${elements.classMode.checked ? "Failing" : "Passing"} student added at ${formatNumber(feature.p1, 1)} hours and ${Math.round(feature.p2 * 10)}% attendance.`,
    );
  }

  function handleContextMenu(event) {
    event.preventDefault();
    const pixel = canvasCoordinates(event);
    if (!isInsidePlot(pixel.x, pixel.y)) {
      return;
    }
    const feature = pixelToFeature(pixel.x, pixel.y);
    addPoint(
      { ...feature, target: 0 },
      `Failing student added at ${formatNumber(feature.p1, 1)} hours and ${Math.round(feature.p2 * 10)}% attendance.`,
    );
  }

  function handlePointerMove(event) {
    const pixel = canvasCoordinates(event);
    if (!isInsidePlot(pixel.x, pixel.y)) {
      pointerFeature = null;
      elements.coordinateReadout.textContent = "Pointer: —";
      return;
    }
    pointerFeature = pixelToFeature(pixel.x, pixel.y);
    elements.coordinateReadout.textContent =
      `Pointer: ${formatNumber(pointerFeature.p1, 1)} h · ${Math.round(pointerFeature.p2 * 10)}%`;
  }

  function handleCanvasKeyboard(event) {
    const movements = {
      ArrowLeft: [-0.5, 0],
      ArrowRight: [0.5, 0],
      ArrowDown: [0, -0.5],
      ArrowUp: [0, 0.5],
    };

    if (movements[event.key]) {
      event.preventDefault();
      keyboardCursor.visible = true;
      keyboardCursor.p1 = roundFeature(keyboardCursor.p1 + movements[event.key][0]);
      keyboardCursor.p2 = roundFeature(keyboardCursor.p2 + movements[event.key][1]);
      elements.coordinateReadout.textContent =
        `Keyboard: ${formatNumber(keyboardCursor.p1, 1)} h · ${Math.round(keyboardCursor.p2 * 10)}%`;
      draw();
      return;
    }

    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      keyboardCursor.visible = true;
      addPoint(
        {
          p1: keyboardCursor.p1,
          p2: keyboardCursor.p2,
          target: elements.classMode.checked ? 0 : 1,
        },
        "Student placed with the keyboard. Use the arrow keys to move the cursor again.",
      );
    }
  }

  function loadPreset() {
    const selected = presets[elements.presetSelect.value];
    const availableSlots = MAX_POINTS - points.length;
    if (availableSlots < selected.length) {
      showError(`This example needs ${selected.length} open slots. Reset or remove data first.`);
      return;
    }
    points.push(...selected.map((point) => ({ ...point })));
    updatePointCount();
    const labels = {
      separable: "Clearly separable data loaded. A straight line can divide these two groups.",
      edge: "Boundary edge case loaded. The passing points sit very close to the correct line.",
      xor: "XOR-like data loaded. Try training to discover why one straight line is not enough.",
    };
    invalidateTraining(labels[elements.presetSelect.value]);
  }

  function resetLab() {
    cancelPlayback();
    points = [];
    model = cloneModel(ZERO_MODEL);
    pointerFeature = null;
    keyboardCursor = { p1: 5, p2: 5, visible: false };
    activeStep = null;
    trainingResult = null;
    trainingFingerprint = "";
    stepCursor = 0;
    elements.classMode.checked = false;
    updateModePresentation();
    updatePointCount();
    clearError();
    updateTelemetry();
    elements.coordinateReadout.textContent = "Pointer: —";
    setBadge("ready", "Ready to explore");
    setNarrationBadge("Waiting");
    elements.narrationText.textContent = "Add both green and red students, or load an example, to begin.";
    draw();
  }

  function setButtonsDisabled(disabled) {
    elements.trainButton.disabled = disabled;
    elements.stepButton.disabled = disabled;
    elements.presetButton.disabled = disabled;
    elements.classMode.disabled = disabled;
    canvas.setAttribute("aria-disabled", String(disabled));
  }

  function currentFingerprint() {
    return JSON.stringify(points);
  }

  async function requestTraining() {
    clearError();
    setButtonsDisabled(true);
    isBusy = true;
    setBadge("learning", "Preparing training");
    setNarrationBadge("Calculating");
    elements.narrationText.textContent = "The server is calculating each predict, compare, and correct step.";

    try {
      const response = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ points, max_epochs: 100 }),
      });
      const data = await response.json().catch(() => null);
      if (!response.ok || !data || !data.ok) {
        throw new Error(data?.error?.message || "Training could not start. Please check the data.");
      }

      trainingResult = data;
      trainingFingerprint = currentFingerprint();
      stepCursor = 0;
      activeStep = null;
      model = { W: [...data.initial_state.W], b: data.initial_state.b };
      updateTelemetry(null, data.initial_state.accuracy);
      setBadge("ready", "Training path ready");
      setNarrationBadge("Start state");
      elements.narrationText.textContent =
        "Weights and bias begin at zero. Step forward once, or play the complete animation.";
      draw();
      return data;
    } finally {
      isBusy = false;
      setButtonsDisabled(false);
    }
  }

  async function ensureTrainingResult() {
    if (trainingResult && trainingFingerprint === currentFingerprint()) {
      return trainingResult;
    }
    try {
      return await requestTraining();
    } catch (error) {
      showError(error instanceof Error ? error.message : "Training could not start.");
      return null;
    }
  }

  function applyStep(step) {
    model = { W: [...step.W], b: step.b };
    activeStep = step;
    updateTelemetry(step);
    narrateStep(step);
    draw();
  }

  function delay(milliseconds, token) {
    return new Promise((resolve) => {
      if (milliseconds <= 0 || token !== animationToken) {
        resolve();
        return;
      }
      window.setTimeout(resolve, milliseconds);
    });
  }

  function animateToStep(step, duration, token) {
    if (prefersReducedMotion.matches || duration <= 16) {
      applyStep(step);
      return Promise.resolve();
    }

    const startModel = cloneModel(model);
    const targetModel = { W: [...step.W], b: step.b };
    const startedAt = performance.now();
    activeStep = step;
    updateTelemetry(step);
    narrateStep(step);

    return new Promise((resolve) => {
      function frame(now) {
        if (token !== animationToken) {
          resolve();
          return;
        }
        const rawProgress = clamp((now - startedAt) / duration, 0, 1);
        const easedProgress = 1 - (1 - rawProgress) ** 3;
        model = {
          W: [
            startModel.W[0] + (targetModel.W[0] - startModel.W[0]) * easedProgress,
            startModel.W[1] + (targetModel.W[1] - startModel.W[1]) * easedProgress,
          ],
          b: startModel.b + (targetModel.b - startModel.b) * easedProgress,
        };
        elements.weightsValue.textContent = `[${formatNumber(model.W[0])}, ${formatNumber(model.W[1])}]`;
        elements.biasValue.textContent = formatNumber(model.b);
        elements.boundaryEquation.textContent = formatBoundary();
        draw();

        if (rawProgress < 1) {
          requestAnimationFrame(frame);
        } else {
          model = targetModel;
          draw();
          resolve();
        }
      }
      requestAnimationFrame(frame);
    });
  }

  function animationDuration(frameCount) {
    const speed = Number(elements.speedSlider.value);
    const baseDurations = { 1: 720, 2: 480, 3: 300, 4: 170, 5: 80 };
    const totalBudgets = { 1: 26000, 2: 20000, 3: 14000, 4: 9000, 5: 6000 };
    return Math.max(18, Math.min(baseDurations[speed], totalBudgets[speed] / Math.max(frameCount, 1)));
  }

  function finishTraining(result) {
    activeStep = null;
    model = { W: [...result.final.W], b: result.final.b };
    updateTelemetry(
      result.history.length ? result.history[result.history.length - 1] : null,
      result.final.accuracy,
    );
    draw();

    if (result.converged) {
      setBadge("success", "Converged · 100% correct");
      setNarrationBadge("Line found", "success");
      elements.narrationText.textContent =
        `Success: one straight boundary classifies every student after ${result.total_epochs} ` +
        `${result.total_epochs === 1 ? "epoch" : "epochs"} and ${result.total_updates} corrections.`;
    } else {
      setBadge("warning", "No single line fits");
      setNarrationBadge("Not separable", "warning");
      elements.narrationText.textContent =
        "The corrections keep pulling the line in conflicting directions. This pattern needs more than one straight boundary.";
    }
  }

  async function trainAnimated() {
    if (isBusy) {
      return;
    }
    const result = await ensureTrainingResult();
    if (!result) {
      return;
    }

    const token = ++animationToken;
    const steps = result.history;
    const perStepDuration = animationDuration(steps.length);
    isBusy = true;
    setButtonsDisabled(true);
    setBadge("learning", "Learning in progress");
    setNarrationBadge("Watching examples");

    model = { W: [...result.initial_state.W], b: result.initial_state.b };
    activeStep = null;
    stepCursor = 0;
    updateTelemetry(null, result.initial_state.accuracy);
    draw();

    for (const step of steps) {
      if (token !== animationToken) {
        return;
      }
      const duration = step.updated ? perStepDuration : perStepDuration * 0.28;
      await animateToStep(step, duration, token);
      stepCursor += 1;
      if (!step.updated) {
        await delay(Math.min(22, perStepDuration * 0.12), token);
      }
    }

    if (token === animationToken) {
      finishTraining(result);
      isBusy = false;
      setButtonsDisabled(false);
    }
  }

  async function stepOnce() {
    if (isBusy) {
      return;
    }
    const result = await ensureTrainingResult();
    if (!result) {
      return;
    }

    if (stepCursor >= result.history.length) {
      finishTraining(result);
      return;
    }

    const token = ++animationToken;
    isBusy = true;
    setButtonsDisabled(true);
    setBadge("learning", "One sample step");
    await animateToStep(result.history[stepCursor], prefersReducedMotion.matches ? 0 : 260, token);
    stepCursor += 1;
    isBusy = false;
    setButtonsDisabled(false);

    if (stepCursor >= result.history.length) {
      finishTraining(result);
    } else {
      setBadge("ready", "Paused between steps");
    }
  }

  function updateSpeedLabel() {
    const labels = { 1: "Very slow", 2: "Slow", 3: "Normal", 4: "Fast", 5: "Very fast" };
    elements.speedValue.textContent = labels[elements.speedSlider.value];
  }

  elements.classMode.addEventListener("change", updateModePresentation);
  elements.presetButton.addEventListener("click", loadPreset);
  elements.trainButton.addEventListener("click", trainAnimated);
  elements.stepButton.addEventListener("click", stepOnce);
  elements.resetButton.addEventListener("click", resetLab);
  elements.speedSlider.addEventListener("input", updateSpeedLabel);
  canvas.addEventListener("click", handleCanvasClick);
  canvas.addEventListener("contextmenu", handleContextMenu);
  canvas.addEventListener("pointermove", handlePointerMove);
  canvas.addEventListener("pointerleave", () => {
    pointerFeature = null;
    elements.coordinateReadout.textContent = "Pointer: —";
  });
  canvas.addEventListener("keydown", handleCanvasKeyboard);

  const resizeObserver = new ResizeObserver(resizeCanvas);
  resizeObserver.observe(canvas.parentElement);

  updateModePresentation();
  updatePointCount();
  updateSpeedLabel();
  updateTelemetry();
  resizeCanvas();
})();
