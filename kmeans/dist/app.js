/*
 * K-MEANS LEARNING ENGINE + LIGHTWEIGHT PAGE CONTROLLER
 *
 * Kid-sized picture: every customer is a marble on graph paper. Age chooses the
 * left/right spot and shopping score chooses the up/down spot. K=2 means there
 * are two bowls. Each marble joins the nearest bowl, then each bowl slides to
 * the balance point (average) of its marbles. The computer repeats this until
 * no marble wants to swap bowls.
 *
 * This Type-A variant trains and predicts directly in JavaScript. There is no
 * model.json because the small dataset and complete training recipe are here.
 */

// Pretend retail customers: each [age, shopping score] pair becomes a named dot.
const customers = [
  [19, 18], [22, 27], [25, 12], [28, 38], [31, 24], [34, 34], [37, 16],
  [41, 31], [44, 20], [48, 39], [52, 23], [56, 33], [61, 17], [67, 29],
  [20, 75], [24, 88], [27, 67], [30, 93], [34, 72], [38, 84], [42, 64],
  [46, 91], [50, 70], [54, 82], [58, 61], [63, 77], [68, 89]
// Destructuring changes [19, 18] into { age: 19, spend: 18 } for readability.
].map(([age, spend]) => ({ age, spend }));

const colors = {
  ink: "#14213d",
  muted: "#6c7890",
  grid: "#dce3f2",
  red: "#e8474f",
  blue: "#356ee8",
  yellow: "#ffd34e",
  white: "#ffffff"
};

const ageRange = document.querySelector("#age-range");
const ageNumber = document.querySelector("#age-number");
const spendRange = document.querySelector("#spend-range");
const spendNumber = document.querySelector("#spend-number");
const ageError = document.querySelector("#age-error");
const spendError = document.querySelector("#spend-error");
const form = document.querySelector("#customer-form");
const resultCard = document.querySelector("#result-card");
const resultTitle = document.querySelector("#result-title");
const resultCopy = document.querySelector("#result-copy");
const budgetCenterText = document.querySelector("#budget-center");
const premiumCenterText = document.querySelector("#premium-center");
const replayButton = document.querySelector("#replay-button");
const resetButton = document.querySelector("#reset-button");
const roundStatus = document.querySelector("#round-status");
const canvas = document.querySelector("#cluster-chart");
const context = canvas.getContext("2d");

let shownCustomer = null;
let winningCluster = null;
let replayTimer = null;

// DISTANCE: measure the straight ruler line between a dot and a team middle.
// Age spans 52 steps (18→70), while score spans 99 (1→100). Dividing by each
// span turns both into equal 0-to-1 map rulers, so the larger score ruler does
// not unfairly shout louder than age.
function normalizedDistance(point, center) {
  // Horizontal gap on the equal-sized age ruler.
  const ageGap = (point.age - center.age) / 52;
  // Vertical gap on the equal-sized shopping ruler.
  const spendGap = (point.spend - center.spend) / 99;
  // Pythagoras is a diagonal ruler: square both side gaps, add them, then take
  // the square root. For gaps 0.3 and 0.4, distance is sqrt(.09+.16)=0.5.
  return Math.sqrt((ageGap * ageGap) + (spendGap * spendGap));
}

// TRAINING: PUT each dot near a center, PUSH each center to its team's average,
// then repeat until no dot changes team.
function runKMeans(points) {
  // PICK: begin with two sensible pretend bowl positions. These are only
  // starting guesses; K-means is allowed to move them.
  let centers = [
    { age: 25, spend: 22 },
    { age: 56, spend: 76 }
  ];
  // -1 means a dot has not joined either team yet.
  let assignments = Array(points.length).fill(-1);
  // Save frames so the Replay button can show the learning process later.
  const history = [{ centers: centers.map(center => ({ ...center })), assignments: [...assignments] }];

  // Twelve rounds are more than enough for this tiny fixed dataset and also
  // guarantee the loop cannot run forever if something unusual happens.
  for (let round = 0; round < 12; round += 1) {
    // PUT: compare each dot with both centers and return team number 0 or 1.
    const nextAssignments = points.map(point => {
      const distanceToFirst = normalizedDistance(point, centers[0]);
      const distanceToSecond = normalizedDistance(point, centers[1]);
      // The smaller ruler length wins; an exact tie consistently chooses team 0.
      return distanceToFirst <= distanceToSecond ? 0 : 1;
    });

    // PUSH: calculate a new balance point for each team.
    const nextCenters = centers.map((oldCenter, clusterIndex) => {
      // Collect only the marbles assigned to this particular bowl.
      const team = points.filter((point, index) => nextAssignments[index] === clusterIndex);
      // An empty team has no average, so safely leave its bowl where it was.
      if (!team.length) return oldCenter;

      return {
        // Mean age = all team ages added together / number of team members.
        age: team.reduce((sum, point) => sum + point.age, 0) / team.length,
        // Mean score uses the same balance-point recipe.
        spend: team.reduce((sum, point) => sum + point.spend, 0) / team.length
      };
    });

    // Copy the round into history so later changes cannot rewrite old frames.
    history.push({
      centers: nextCenters.map(center => ({ ...center })),
      assignments: [...nextAssignments]
    });

    // Convergence means every marble picked the same team as last round.
    const nobodyMoved = nextAssignments.every((team, index) => team === assignments[index]);
    // Make this round the current truth.
    centers = nextCenters;
    assignments = nextAssignments;
    // Stable teams mean the next round would repeat exactly, so stop early.
    if (nobodyMoved) break;
  }

  // Return learned centers, each dot's team, and animation frames.
  return { centers, assignments, history };
}

// Training runs once when the script loads.
const model = runKMeans(customers);
let displayState = { centers: model.centers, assignments: model.assignments };

function getTeamRoles(centers) {
  const budgetIndex = centers[0].spend <= centers[1].spend ? 0 : 1;
  return { budgetIndex, premiumIndex: budgetIndex === 0 ? 1 : 0 };
}

function updateCenterLabels() {
  const roles = getTeamRoles(model.centers);
  const budget = model.centers[roles.budgetIndex];
  const premium = model.centers[roles.premiumIndex];
  budgetCenterText.textContent = `Age ${Math.round(budget.age)} · Score ${Math.round(budget.spend)}`;
  premiumCenterText.textContent = `Age ${Math.round(premium.age)} · Score ${Math.round(premium.spend)}`;
}

function clamp(value, min, max) {
  // Convert text to a number, replace non-numbers with min, round to a whole
  // value, and keep the answer between the two safe fence posts.
  const safeNumber = Number.isFinite(Number(value)) ? Number(value) : min;
  return Math.min(max, Math.max(min, Math.round(safeNumber)));
}

// Show a friendly warning instead of silently accepting an impossible value.
function validateNumberInput(numberInput, min, max, errorElement, label) {
  const value = Number(numberInput.value);
  let message = "";
  if (numberInput.value.trim() === "") {
    message = `Please enter ${label}.`;
  } else if (!Number.isFinite(value) || value < min || value > max) {
    message = `${label} must be from ${min} to ${max}.`;
  }
  numberInput.setAttribute("aria-invalid", String(Boolean(message)));
  errorElement.textContent = message;
  return message === "";
}

// Keep a safe range slider and its matching number box synchronized.
function linkInputs(rangeInput, numberInput, min, max, errorElement, label) {
  rangeInput.addEventListener("input", () => {
    numberInput.value = rangeInput.value;
    validateNumberInput(numberInput, min, max, errorElement, label);
  });

  numberInput.addEventListener("input", () => {
    if (validateNumberInput(numberInput, min, max, errorElement, label)) {
      rangeInput.value = clamp(numberInput.value, min, max);
    }
  });

  numberInput.addEventListener("change", () => {
    validateNumberInput(numberInput, min, max, errorElement, label);
  });
}

linkInputs(ageRange, ageNumber, 18, 70, ageError, "Age");
linkInputs(spendRange, spendNumber, 1, 100, spendError, "Shopping score");

function setInputs(age, spend) {
  ageRange.value = age;
  ageNumber.value = age;
  spendRange.value = spend;
  spendNumber.value = spend;
  // Programmatic presets are known-safe, so clear any earlier typed warning.
  validateNumberInput(ageNumber, 18, 70, ageError, "Age");
  validateNumberInput(spendNumber, 1, 100, spendError, "Shopping score");
}

function classifyCustomer() {
  // Guard the ML calculation until both typed values sit inside the teaching map.
  const ageIsValid = validateNumberInput(ageNumber, 18, 70, ageError, "Age");
  const spendIsValid = validateNumberInput(spendNumber, 1, 100, spendError, "Shopping score");
  if (!ageIsValid || !spendIsValid) return;

  // The new customer is one dot; both inputs are now known to be safe.
  const point = {
    age: clamp(ageNumber.value, 18, 70),
    spend: clamp(spendNumber.value, 1, 100)
  };
  setInputs(point.age, point.spend);

  // Measure the new dot to both learned bowl centers.
  const distances = model.centers.map(center => normalizedDistance(point, center));
  // The closest center supplies the cluster number.
  winningCluster = distances[0] <= distances[1] ? 0 : 1;
  shownCustomer = point;

  const roles = getTeamRoles(model.centers);
  const isBudget = winningCluster === roles.budgetIndex;
  const winnerName = isBudget ? "Budget Shopper" : "Premium Shopper";
  const winnerColor = isBudget ? "red" : "blue";
  const losingCluster = winningCluster === 0 ? 1 : 0;
  const nearSteps = Math.round(distances[winningCluster] * 100);
  const farSteps = Math.round(distances[losingCluster] * 100);

  resultCard.classList.remove("is-budget", "is-premium");
  resultCard.classList.add(isBudget ? "is-budget" : "is-premium");
  resultTitle.textContent = `${winnerColor === "red" ? "Red" : "Blue"} team: ${winnerName}!`;
  resultCopy.textContent = `Your dot is ${nearSteps} map-steps from this team’s middle, and ${farSteps} from the other one. Closer wins!`;
  drawChart();
}

form.addEventListener("submit", event => {
  event.preventDefault();
  classifyCustomer();
});

document.querySelectorAll(".preset").forEach(button => {
  button.addEventListener("click", () => {
    setInputs(button.dataset.age, button.dataset.spend);
    classifyCustomer();
  });
});

resetButton.addEventListener("click", () => {
  setInputs(34, 56);
  classifyCustomer();
});

function startReplay() {
  window.clearInterval(replayTimer);
  replayButton.disabled = true;
  shownCustomer = null;
  winningCluster = null;
  resultCard.classList.remove("is-budget", "is-premium");
  resultTitle.textContent = "Watch the middle stars move";
  resultCopy.textContent = "Each dot picks the nearest star. Then each star moves to its team’s average.";

  let step = 0;
  const frames = model.history;

  function showFrame() {
    displayState = frames[step];
    const isFirst = step === 0;
    roundStatus.innerHTML = isFirst
      ? "<strong>Starting:</strong> two middle stars are picked."
      : `<strong>Round ${step}:</strong> dots join a team, then the stars move.`;
    drawChart();
    step += 1;

    if (step >= frames.length) {
      window.clearInterval(replayTimer);
      window.setTimeout(() => {
        displayState = { centers: model.centers, assignments: model.assignments };
        replayButton.disabled = false;
        roundStatus.innerHTML = `<strong>Done in ${frames.length - 1} rounds:</strong> no dot needs to change teams.`;
        classifyCustomer();
      }, 520);
    }
  }

  showFrame();
  replayTimer = window.setInterval(showFrame, 720);
}

replayButton.addEventListener("click", startReplay);

function drawStar(ctx, x, y, radius, fill) {
  ctx.save();
  ctx.beginPath();
  for (let point = 0; point < 10; point += 1) {
    const angle = (-Math.PI / 2) + (point * Math.PI / 5);
    const length = point % 2 === 0 ? radius : radius * 0.46;
    const px = x + Math.cos(angle) * length;
    const py = y + Math.sin(angle) * length;
    if (point === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = colors.ink;
  ctx.stroke();
  ctx.restore();
}

function drawChart() {
  const bounds = canvas.getBoundingClientRect();
  const width = Math.max(320, bounds.width);
  const height = Math.max(320, bounds.height);
  const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(width * pixelRatio);
  canvas.height = Math.floor(height * pixelRatio);
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, width, height);

  const pad = { left: width < 460 ? 48 : 58, right: 22, top: 30, bottom: 52 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const xForAge = age => pad.left + ((age - 18) / 52) * plotWidth;
  const yForSpend = spend => pad.top + plotHeight - ((spend - 1) / 99) * plotHeight;

  const roles = getTeamRoles(displayState.centers);

  // Soft map squares show which center owns each part of the map.
  for (let x = pad.left; x < pad.left + plotWidth; x += 14) {
    for (let y = pad.top; y < pad.top + plotHeight; y += 14) {
      const mapPoint = {
        age: 18 + ((x - pad.left) / plotWidth) * 52,
        spend: 100 - ((y - pad.top) / plotHeight) * 99
      };
      const nearest = normalizedDistance(mapPoint, displayState.centers[0]) <= normalizedDistance(mapPoint, displayState.centers[1]) ? 0 : 1;
      context.fillStyle = nearest === roles.budgetIndex ? "rgba(232, 71, 79, 0.035)" : "rgba(53, 110, 232, 0.035)";
      context.fillRect(x, y, 14, 14);
    }
  }

  context.font = "700 12px Trebuchet MS, Segoe UI, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "top";

  [20, 30, 40, 50, 60, 70].forEach(age => {
    const x = xForAge(age);
    context.beginPath();
    context.moveTo(x, pad.top);
    context.lineTo(x, pad.top + plotHeight);
    context.strokeStyle = colors.grid;
    context.lineWidth = 1;
    context.stroke();
    context.fillStyle = colors.muted;
    context.fillText(age, x, pad.top + plotHeight + 9);
  });

  [20, 40, 60, 80, 100].forEach(spend => {
    const y = yForSpend(spend);
    context.beginPath();
    context.moveTo(pad.left, y);
    context.lineTo(pad.left + plotWidth, y);
    context.strokeStyle = colors.grid;
    context.stroke();
    context.textAlign = "right";
    context.textBaseline = "middle";
    context.fillStyle = colors.muted;
    context.fillText(spend, pad.left - 10, y);
  });

  context.beginPath();
  context.moveTo(pad.left, pad.top);
  context.lineTo(pad.left, pad.top + plotHeight);
  context.lineTo(pad.left + plotWidth, pad.top + plotHeight);
  context.strokeStyle = colors.ink;
  context.lineWidth = 2;
  context.stroke();

  context.fillStyle = colors.ink;
  context.textAlign = "center";
  context.textBaseline = "bottom";
  context.font = "800 13px Trebuchet MS, Segoe UI, sans-serif";
  context.fillText("CUSTOMER AGE →", pad.left + plotWidth / 2, height - 5);

  context.save();
  context.translate(14, pad.top + plotHeight / 2);
  context.rotate(-Math.PI / 2);
  context.fillText("SHOPPING SCORE →", 0, 0);
  context.restore();

  // A blank assignment means the replay has only picked its starting stars.
  customers.forEach((point, index) => {
    const team = displayState.assignments[index];
    const fill = team === -1 ? "#aab4c7" : team === roles.budgetIndex ? colors.red : colors.blue;
    const x = xForAge(point.age);
    const y = yForSpend(point.spend);
    context.beginPath();
    context.arc(x, y, 5.5, 0, Math.PI * 2);
    context.fillStyle = fill;
    context.fill();
    context.lineWidth = 1.5;
    context.strokeStyle = colors.white;
    context.stroke();
  });

  if (shownCustomer) {
    const userX = xForAge(shownCustomer.age);
    const userY = yForSpend(shownCustomer.spend);

    displayState.centers.forEach((center, clusterIndex) => {
      context.beginPath();
      context.moveTo(userX, userY);
      context.lineTo(xForAge(center.age), yForSpend(center.spend));
      context.setLineDash(clusterIndex === winningCluster ? [] : [5, 6]);
      context.strokeStyle = clusterIndex === winningCluster
        ? (clusterIndex === roles.budgetIndex ? colors.red : colors.blue)
        : "#a7b1c5";
      context.lineWidth = clusterIndex === winningCluster ? 3 : 1.5;
      context.stroke();
    });
    context.setLineDash([]);
  }

  displayState.centers.forEach((center, clusterIndex) => {
    const fill = clusterIndex === roles.budgetIndex ? colors.red : colors.blue;
    drawStar(context, xForAge(center.age), yForSpend(center.spend), 12, fill);
  });

  if (shownCustomer) {
    const x = xForAge(shownCustomer.age);
    const y = yForSpend(shownCustomer.spend);
    context.beginPath();
    context.arc(x, y, 9, 0, Math.PI * 2);
    context.fillStyle = colors.yellow;
    context.fill();
    context.lineWidth = 3;
    context.strokeStyle = colors.ink;
    context.stroke();

    const labelY = Math.max(pad.top + 3, y - 29);
    context.font = "900 11px Trebuchet MS, Segoe UI, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillStyle = colors.ink;
    context.fillText("YOU", x, labelY);
  }
}

const resizeObserver = new ResizeObserver(drawChart);
resizeObserver.observe(canvas);

updateCenterLabels();
classifyCustomer();

// Save the small local app after its first hosted visit so it can reopen when
// the device is offline. The algorithm itself already makes no network calls.
if ("serviceWorker" in navigator && location.protocol !== "file:") {
  navigator.serviceWorker.register("./sw.js").catch((error) => {
    console.warn("Offline cache could not start", error);
  });
}
