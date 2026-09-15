const state = {
  file: null,
  stream: null,
  resultUrl: null,
};

const el = Object.fromEntries(
  [
    "uploadTab", "cameraTab", "uploadPane", "cameraPane", "previewPane", "fileInput",
    "dropZone", "cameraVideo", "cameraCanvas", "cameraMessage", "captureButton",
    "stopCameraButton", "previewImage", "fileName", "fileSize", "confidenceInput",
    "confidenceValue", "detectButton", "resetButton", "emptyResult", "loadingResult",
    "resultImage", "downloadButton", "objectCount", "elapsedTime", "thresholdStat",
    "detectionList", "classCount", "resultBadge", "toast",
  ].map((id) => [id, document.getElementById(id)])
);

function showToast(message, kind = "error") {
  el.toast.textContent = message;
  el.toast.dataset.kind = kind;
  el.toast.hidden = false;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => { el.toast.hidden = true; }, 4500);
}

function formatBytes(bytes) {
  if (!bytes) return "Camera capture";
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function setTab(tab) {
  const camera = tab === "camera";
  el.uploadTab.classList.toggle("active", !camera);
  el.cameraTab.classList.toggle("active", camera);
  el.uploadTab.setAttribute("aria-selected", String(!camera));
  el.cameraTab.setAttribute("aria-selected", String(camera));
  el.uploadPane.hidden = camera;
  el.cameraPane.hidden = !camera;
  el.previewPane.hidden = true;
  if (camera) startCamera(); else stopCamera();
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    el.cameraMessage.textContent = "Camera access is not available in this browser.";
    showToast("This browser cannot open the camera. Upload a photo instead.");
    return;
  }
  try {
    stopCamera();
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 } },
      audio: false,
    });
    el.cameraVideo.srcObject = state.stream;
    await el.cameraVideo.play();
    el.cameraMessage.hidden = true;
    el.captureButton.disabled = false;
  } catch (_error) {
    el.cameraMessage.hidden = false;
    el.cameraMessage.textContent = "Camera access was blocked. Check your browser permission.";
    showToast("Camera permission is needed, or you can upload a photo.");
  }
}

function stopCamera() {
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  state.stream = null;
  el.cameraVideo.srcObject = null;
  el.captureButton.disabled = true;
}

function validateFile(file) {
  const allowed = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
  if (!allowed.includes(file.type)) throw new Error("Use a JPG, PNG, WEBP, or BMP image.");
  if (file.size > 12 * 1024 * 1024) throw new Error("The image is larger than the 12 MB limit.");
}

function chooseFile(file) {
  try {
    validateFile(file);
  } catch (error) {
    showToast(error.message);
    return;
  }
  stopCamera();
  state.file = file;
  const objectUrl = URL.createObjectURL(file);
  el.previewImage.onload = () => URL.revokeObjectURL(objectUrl);
  el.previewImage.src = objectUrl;
  el.fileName.textContent = file.name || "camera-capture.jpg";
  el.fileSize.textContent = formatBytes(file.size);
  el.uploadPane.hidden = true;
  el.cameraPane.hidden = true;
  el.previewPane.hidden = false;
  el.detectButton.disabled = false;
  el.resetButton.disabled = false;
}

function resetSource() {
  // Clear both the chosen source and any old downloadable result.
  stopCamera();
  state.file = null;
  state.resultUrl = null;
  el.fileInput.value = "";
  el.previewImage.removeAttribute("src");
  el.previewPane.hidden = true;
  el.uploadPane.hidden = false;
  el.cameraPane.hidden = true;
  el.uploadTab.classList.add("active");
  el.cameraTab.classList.remove("active");
  el.uploadTab.setAttribute("aria-selected", "true");
  el.cameraTab.setAttribute("aria-selected", "false");
  el.detectButton.disabled = true;
  el.resetButton.disabled = true;
  el.resultBadge.textContent = "Waiting for a photo";
  el.resultBadge.dataset.kind = "idle";
}

function captureFrame() {
  const { videoWidth: width, videoHeight: height } = el.cameraVideo;
  if (!width || !height) return showToast("The camera is still starting. Try again in a moment.");
  el.cameraCanvas.width = width;
  el.cameraCanvas.height = height;
  el.cameraCanvas.getContext("2d").drawImage(el.cameraVideo, 0, 0, width, height);
  el.cameraCanvas.toBlob((blob) => {
    if (!blob) return showToast("The camera frame could not be captured.");
    chooseFile(new File([blob], `camera-${Date.now()}.jpg`, { type: "image/jpeg" }));
  }, "image/jpeg", 0.92);
}

function setLoading(loading) {
  el.detectButton.disabled = loading || !state.file;
  el.resetButton.disabled = loading;
  el.loadingResult.hidden = !loading;
  if (loading) {
    el.resultBadge.textContent = "Analyzing image…";
    el.resultBadge.dataset.kind = "loading";
    el.emptyResult.hidden = true;
    el.resultImage.hidden = true;
    el.downloadButton.disabled = true;
  }
}

function renderDetections(data) {
  el.resultImage.src = data.image;
  el.resultImage.hidden = false;
  el.emptyResult.hidden = true;
  el.objectCount.textContent = data.count;
  el.elapsedTime.textContent = `${data.elapsed_ms} ms`;
  el.thresholdStat.textContent = `${Math.round(data.confidence * 100)}%`;
  const uniqueClasses = new Set(data.detections.map((item) => item.label)).size;
  el.classCount.textContent = `${uniqueClasses} ${uniqueClasses === 1 ? "class" : "classes"}`;
  el.downloadButton.disabled = false;
  state.resultUrl = data.image;

  // Use a plain-English status badge instead of making users interpret count 0.
  el.resultBadge.textContent = data.count
    ? `${data.count} ${data.count === 1 ? "object" : "objects"} found`
    : "No strong match";
  el.resultBadge.dataset.kind = data.count ? "found" : "empty";

  if (!data.detections.length) {
    el.detectionList.innerHTML = '<p class="list-empty">No object passed this confidence threshold. Try lowering it or use a clearer photo.</p>';
    return;
  }

  el.detectionList.innerHTML = data.detections.map((item, index) => {
    const percent = Math.round(item.confidence * 100);
    return `
      <article class="detection-item">
        <div class="detection-index" style="--item-color:${item.color}">${String(index + 1).padStart(2, "0")}</div>
        <div class="detection-detail">
          <div><strong>${escapeHtml(item.label)}</strong><span>${percent}%</span></div>
          <div class="confidence-bar"><i style="width:${percent}%;background:${item.color}"></i></div>
        </div>
      </article>`;
  }).join("");
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[character]));
}

async function runDetection() {
  if (!state.file) return showToast("Choose a photo first.");
  setLoading(true);
  const form = new FormData();
  form.append("image", state.file);
  form.append("confidence", String(Number(el.confidenceInput.value) / 100));

  try {
    const response = await fetch("/api/detect", { method: "POST", body: form });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || "Detection failed. Try again.");
    renderDetections(data);
  } catch (error) {
    el.emptyResult.hidden = false;
    el.resultBadge.textContent = "Could not analyze";
    el.resultBadge.dataset.kind = "error";
    showToast(error.message);
  } finally {
    setLoading(false);
  }
}

function downloadResult() {
  if (!state.resultUrl) return;
  const link = document.createElement("a");
  link.href = state.resultUrl;
  link.download = "sightline-detection.jpg";
  link.click();
}

el.uploadTab.addEventListener("click", () => setTab("upload"));
el.cameraTab.addEventListener("click", () => setTab("camera"));
el.dropZone.addEventListener("click", () => el.fileInput.click());
el.fileInput.addEventListener("change", () => el.fileInput.files[0] && chooseFile(el.fileInput.files[0]));
el.captureButton.addEventListener("click", captureFrame);
el.stopCameraButton.addEventListener("click", () => setTab("upload"));
el.resetButton.addEventListener("click", resetSource);
el.detectButton.addEventListener("click", runDetection);
el.downloadButton.addEventListener("click", downloadResult);

// Confidence presets change the slider, then fire its normal input handler so
// the displayed percentage and filled track stay synchronized.
document.querySelectorAll("[data-confidence]").forEach((button) => {
  button.addEventListener("click", () => {
    el.confidenceInput.value = button.dataset.confidence;
    el.confidenceInput.dispatchEvent(new Event("input"));
  });
});

["dragenter", "dragover"].forEach((eventName) => el.dropZone.addEventListener(eventName, (event) => {
  event.preventDefault();
  el.dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((eventName) => el.dropZone.addEventListener(eventName, (event) => {
  event.preventDefault();
  el.dropZone.classList.remove("dragging");
}));
el.dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) chooseFile(file);
});

el.confidenceInput.addEventListener("input", () => {
  const value = `${el.confidenceInput.value}%`;
  el.confidenceValue.textContent = value;
  el.thresholdStat.textContent = value;
  el.confidenceInput.style.setProperty("--range", `${(el.confidenceInput.value - 10) / 0.8}%`);
});

window.addEventListener("beforeunload", stopCamera);
el.confidenceInput.dispatchEvent(new Event("input"));
