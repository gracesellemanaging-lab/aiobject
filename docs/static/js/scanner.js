"use strict";

const overlayContext = els.overlay.getContext("2d");
let lastLiveErrorAt = 0;

function beginScan() {
  if (state.scanning) return null;
  state.scanToken += 1;
  state.scanning = true;
  els.stage.classList.add("is-scanning");
  els.scanSpinner.hidden = false;
  els.resultsStatus.textContent = "Scanning";
  els.resultsStatus.classList.add("active");
  els.scanNow.disabled = true;
  els.uploadTrigger.disabled = true;
  return state.scanToken;
}

function endScan(token) {
  if (state.scanToken !== token) return;
  state.scanning = false;
  els.stage.classList.remove("is-scanning");
  els.scanSpinner.hidden = true;
  els.resultsStatus.classList.remove("active");
  els.scanNow.disabled = false;
  els.uploadTrigger.disabled = false;
  if (!state.result) els.resultsStatus.textContent = "Waiting";
}

function abortScan() {
  state.scanToken += 1;
  state.scanning = false;
  state.epoch += 1;
  if (state.controller) {
    try {
      state.controller.abort();
    } catch {
      state.controller = null;
    }
    state.controller = null;
  }
  els.stage.classList.remove("is-scanning");
  els.scanSpinner.hidden = true;
  els.resultsStatus.classList.remove("active");
  els.scanNow.disabled = false;
  els.uploadTrigger.disabled = false;
  if (!state.result) els.resultsStatus.textContent = "Waiting";
}

async function startCamera() {
  if (state.cameraOn) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    toast("This browser does not expose a camera. Try uploading an image instead.", "error");
    return;
  }
  clearTimeout(state.loopTimer);
  const epoch = ++state.epoch;
  els.cameraStart.disabled = true;
  els.cameraState.innerHTML = '<span class="dot"></span>Requesting camera';
  els.cameraState.classList.remove("live");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1920 },
        height: { ideal: 1080 },
        frameRate: { ideal: 30 },
      },
    });
    await tuneCameraTrack(stream);
    if (epoch !== state.epoch) {
      stream.getTracks().forEach((track) => track.stop());
      return;
    }
    state.stream = stream;
    els.camera.srcObject = stream;
    await els.camera.play();
    state.cameraOn = true;
    state.sessionId = newSessionId();
    state.usingImage = false;
    state.imageBlob = null;
    if (state.uploadedUrl) {
      URL.revokeObjectURL(state.uploadedUrl);
      state.uploadedUrl = null;
    }
    els.uploadImage.removeAttribute("src");
    resetScanOutput();
    els.stage.dataset.kind = "camera";
    els.stageBadge.textContent = "LIVE";
    els.cameraStart.disabled = true;
    els.cameraStop.disabled = false;
    els.cameraState.innerHTML = `<span class="dot"></span>Live · ${(settings.interval / 1000).toFixed(1)}s scan`;
    els.cameraState.classList.add("live");
    updateOverlaySize();
    startLiveLoop(120);
  } catch (error) {
    state.cameraOn = false;
    els.cameraStart.disabled = false;
    els.cameraStop.disabled = true;
    els.cameraState.innerHTML = '<span class="dot"></span>Camera unavailable';
    els.cameraState.classList.remove("live");
    const denied = error && (error.name === "NotAllowedError" || error.name === "SecurityError");
    toast(denied
      ? "Camera permission was denied. Allow access in the browser or upload an image."
      : `Camera could not start: ${networkErrorText(error)}`, "error", 6000);
    if (!state.usingImage) {
      els.stage.dataset.kind = "empty";
      els.stageBadge.textContent = "STANDBY";
    }
  }
}

function stopCamera(options = {}) {
  const { keepFrame = true } = options;
  const hadCamera = state.cameraOn;
  const frozen = hadCamera && keepFrame && !state.usingImage ? captureFrameSync() : null;
  clearTimeout(state.loopTimer);
  state.loopTimer = null;
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  state.cameraOn = false;
  state.sessionId = null;
  abortScan();
  if (els.camera.srcObject) {
    try {
      els.camera.pause();
    } catch {
      els.camera.srcObject = null;
    }
    els.camera.srcObject = null;
  }
  els.cameraStart.disabled = false;
  els.cameraStop.disabled = true;
  els.cameraState.innerHTML = '<span class="dot"></span>Camera stopped';
  els.cameraState.classList.remove("live");

  if (frozen && !state.usingImage) {
    els.uploadImage.src = frozen;
    els.stage.dataset.kind = "image";
    els.stageBadge.textContent = "PAUSED";
    els.cameraState.innerHTML = '<span class="dot"></span>Preview paused';
    updateOverlaySize();
    return;
  }
  if (!state.usingImage) {
    els.stage.dataset.kind = "empty";
    els.stageBadge.textContent = "STANDBY";
    clearOverlay();
  }
}

function captureFrameSync() {
  const width = els.camera.videoWidth;
  const height = els.camera.videoHeight;
  if (!width || !height) return null;
  const canvas = document.createElement("canvas");
  const ratio = Math.min(1, 1280 / Math.max(width, height));
  canvas.width = Math.max(1, Math.round(width * ratio));
  canvas.height = Math.max(1, Math.round(height * ratio));
  canvas.getContext("2d").drawImage(els.camera, 0, 0, canvas.width, canvas.height);
  try {
    return canvas.toDataURL("image/jpeg", 0.92);
  } catch {
    return null;
  }
}

// Ask the camera for continuous autofocus/exposure where the device supports it.
async function tuneCameraTrack(stream) {
  const track = stream.getVideoTracks()[0];
  if (!track || !track.getCapabilities) return;
  try {
    const caps = track.getCapabilities();
    const advanced = [];
    if (caps.focusMode && caps.focusMode.includes("continuous")) advanced.push({ focusMode: "continuous" });
    if (caps.exposureMode && caps.exposureMode.includes("continuous")) advanced.push({ exposureMode: "continuous" });
    if (caps.whiteBalanceMode && caps.whiteBalanceMode.includes("continuous")) advanced.push({ whiteBalanceMode: "continuous" });
    if (advanced.length) await track.applyConstraints({ advanced });
  } catch {
    /* unsupported constraint: keep browser defaults */
  }
}

// Laplacian variance on a small grayscale copy: low values mean a blurry frame.
let sharpnessPeak = 0;
let skippedBlurry = 0;
function frameSharpness(video) {
  const size = 160;
  const canvas = document.createElement("canvas");
  const ratio = size / Math.max(video.videoWidth, video.videoHeight);
  const w = Math.max(8, Math.round(video.videoWidth * ratio));
  const h = Math.max(8, Math.round(video.videoHeight * ratio));
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(video, 0, 0, w, h);
  const { data } = ctx.getImageData(0, 0, w, h);
  const gray = new Float32Array(w * h);
  for (let i = 0; i < gray.length; i += 1) {
    gray[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
  }
  let sum = 0;
  let sumSq = 0;
  let count = 0;
  for (let y = 1; y < h - 1; y += 1) {
    for (let x = 1; x < w - 1; x += 1) {
      const i = y * w + x;
      const lap = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - w] - gray[i + w];
      sum += lap;
      sumSq += lap * lap;
      count += 1;
    }
  }
  const mean = sum / count;
  return sumSq / count - mean * mean;
}

function captureFrameBlob(options = {}) {
  const width = els.camera.videoWidth;
  const height = els.camera.videoHeight;
  if (!width || !height || els.camera.readyState < 2) return Promise.resolve(null);

  // Live scans skip motion-blurred frames (never more than 2 in a row); manual scans always send.
  if (!options.manual) {
    try {
      const sharpness = frameSharpness(els.camera);
      sharpnessPeak = Math.max(sharpness, sharpnessPeak * 0.97);
      if (sharpness < sharpnessPeak * 0.4 && skippedBlurry < 2) {
        skippedBlurry += 1;
        return Promise.resolve(null);
      }
      skippedBlurry = 0;
    } catch {
      /* sharpness check is best effort */
    }
  }

  const canvas = document.createElement("canvas");
  const longEdge = options.manual ? 1280 : 960;
  const ratio = Math.min(1, longEdge / Math.max(width, height));
  canvas.width = Math.max(1, Math.round(width * ratio));
  canvas.height = Math.max(1, Math.round(height * ratio));
  const context = canvas.getContext("2d");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(els.camera, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", options.manual ? 0.92 : 0.82);
  });
}

async function sendScan(blob, source, save) {
  if (!blob) throw new Error("The frame could not be captured.");
  const controller = new AbortController();
  state.controller = controller;
  const timeout = window.setTimeout(() => controller.abort(), 120000);
  const form = new FormData();
  form.append("image", blob, source === "camera" ? "frame.jpg" : (state.uploadName || "image.jpg"));
  form.append("mode", backendMode());
  form.append("confidence", String(settings.confidence / 100));
  form.append("source", source);
  form.append("save", save ? "true" : "false");
  if (source === "camera" && state.sessionId) form.append("session_id", state.sessionId);
  try {
    const response = await fetch("/api/scan", {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = data.detail ? ` ${data.detail}` : "";
      throw new Error(`${data.error || `Scan failed (${response.status}).`}${detail}`);
    }
    return data;
  } finally {
    clearTimeout(timeout);
    if (state.controller === controller) state.controller = null;
  }
}

async function runCameraScan(options = {}) {
  if (!state.cameraOn) return null;
  const token = beginScan();
  if (!token) return null;
  const epoch = state.epoch;
  const expectedMode = state.mode;
  try {
    const blob = await captureFrameBlob({ manual: Boolean(options.manual) });
    if (!blob || epoch !== state.epoch) return null;
    const shouldSave = Boolean(options.manual) || (settings.autoSave && Date.now() - state.lastSaveAt > 30000);
    const result = await sendScan(blob, "camera", shouldSave);
    if (epoch !== state.epoch || expectedMode !== state.mode) return null;
    applyScanResult(result);
    if (result.saved > 0) {
      state.lastSaveAt = Date.now();
      state.statsDirty = true;
      if (options.manual) toast(`${result.saved} detection${result.saved === 1 ? "" : "s"} saved to history.`);
    } else if (options.manual) {
      toast("Scan complete. Nothing new was saved to history.", "info");
    }
    return result;
  } catch (error) {
    if (epoch !== state.epoch || (error && error.name === "AbortError")) return null;
    els.resultsStatus.textContent = "Error";
    els.resultsStatus.classList.add("active");
    if (options.manual || Date.now() - lastLiveErrorAt > 12000) {
      lastLiveErrorAt = Date.now();
      toast(networkErrorText(error), "error", 6000);
    }
    return null;
  } finally {
    endScan(token);
  }
}

async function runUploadScan() {
  if (!state.imageBlob) {
    toast("Choose an image first.", "info");
    return null;
  }
  const token = beginScan();
  if (!token) return null;
  const epoch = state.epoch;
  const expectedMode = state.mode;
  try {
    const result = await sendScan(state.imageBlob, "upload", true);
    if (epoch !== state.epoch || expectedMode !== state.mode) return null;
    applyScanResult(result);
    if (result.saved > 0) {
      state.lastSaveAt = Date.now();
      state.statsDirty = true;
      toast(`${result.saved} detection${result.saved === 1 ? "" : "s"} saved to history.`);
    }
    return result;
  } catch (error) {
    if (epoch !== state.epoch || (error && error.name === "AbortError")) return null;
    els.resultsStatus.textContent = "Error";
    els.resultsStatus.classList.add("active");
    toast(networkErrorText(error), "error", 6500);
    return null;
  } finally {
    endScan(token);
  }
}

function startLiveLoop(delay = 160) {
  clearTimeout(state.loopTimer);
  const step = async () => {
    if (!state.cameraOn) return;
    await runCameraScan();
    if (state.cameraOn) state.loopTimer = window.setTimeout(step, settings.interval);
  };
  state.loopTimer = window.setTimeout(step, delay);
}

function resetScanOutput() {
  state.result = null;
  state.selectedId = null;
  els.resultCount.textContent = "0";
  els.metaInference.textContent = "—";
  els.metaCount.textContent = "0";
  els.metaImage.textContent = "—";
  els.metaEngine.textContent = MODES[state.mode].engine;
  els.scanNotes.hidden = true;
  els.scanNotesList.innerHTML = "";
  els.scanSummary.hidden = true;
  els.scanSummaryText.textContent = "";
  els.resultsStatus.textContent = "Waiting";
  els.resultsStatus.classList.remove("active");
  els.resultsList.innerHTML = `
    <div class="results-empty">
      <span class="radar"><i></i><i></i><i></i></span>
      <strong>No detections yet</strong>
      <p>Start a camera session or upload a photo. Results appear here with descriptions and visual references.</p>
    </div>`;
  clearOverlay();
}

function applyScanResult(result) {
  state.result = result;
  state.selectedId = null;
  els.metaEngine.textContent = result.meta && result.meta.engine ? result.meta.engine : MODES[state.mode].engine;
  const inference = result.meta && Number(result.meta.inference_ms);
  els.metaInference.textContent = inference ? `${Math.round(inference)} ms` : "—";
  const image = result.meta && result.meta.image;
  els.metaImage.textContent = image ? `${image.width} × ${image.height}` : "—";
  els.metaCount.textContent = String(result.detections ? result.detections.length : 0);
  if (result.meta && result.meta.capability_note) {
    els.capabilityNote.textContent = result.meta.capability_note;
  }
  if (result.description) {
    els.scanSummary.hidden = false;
    els.scanSummaryText.textContent = result.description;
  } else {
    els.scanSummary.hidden = true;
    els.scanSummaryText.textContent = "";
  }
  if (Array.isArray(result.notes) && result.notes.length) {
    els.scanNotes.hidden = false;
    els.scanNotesList.innerHTML = result.notes.map((note) => `<p>${esc(note)}</p>`).join("");
  } else {
    els.scanNotes.hidden = true;
    els.scanNotesList.innerHTML = "";
  }
  renderDetections(result.detections || []);
  els.resultsStatus.textContent = state.cameraOn ? "Live" : "Complete";
  els.resultsStatus.classList.toggle("active", state.cameraOn);
  if (state.cameraOn) els.stageBadge.textContent = "LIVE";
  else if (state.usingImage) els.stageBadge.textContent = "PHOTO";
  drawOverlay();
}

function renderDetections(detections) {
  els.resultCount.textContent = String(detections.length);
  if (!detections.length) {
    const message = state.result && state.result.notes && state.result.notes.length
      ? state.result.notes[0]
      : "No supported detection in this frame. Try a different angle or a dedicated scanner mode.";
    els.resultsList.innerHTML = `
      <div class="results-empty">
        <span class="radar"><i></i><i></i><i></i></span>
        <strong>No detections in this frame</strong>
        <p>${esc(message)}</p>
      </div>`;
    return;
  }
  els.resultsList.innerHTML = detections.map((detection) => detectionCardHTML(detection)).join("");
  els.resultsList.scrollTop = 0;
  detections.forEach((detection) => {
    if (detection.reference && detection.reference.status === "ready" && detection.reference.image) {
      state.refs.set(detection.reference_key, detection.reference);
    }
  });
  els.resultsList.querySelectorAll(".ref-strip[data-ref-key]").forEach((strip) => {
    hydrateReferenceStrip(strip);
  });
  bindImageFallbacks(els.resultsList);
}

function detectionCardHTML(detection) {
  const accent = validColor(detection.color, colorFor(detection.category));
  const confidence = Math.round((Number(detection.confidence) || 0) * 100);
  const quantity = Number(detection.quantity) || 1;
  const attributes = attributeChips(detection, false);
  const quantityChip = quantity > 1
    ? `<span class="attr-chip quantity-chip">Quantity <strong>×${quantity}</strong></span>`
    : (detection.category === "Fruits" || detection.category === "Vegetables")
      ? '<span class="attr-chip quantity-chip">Estimated quantity <strong>×1</strong></span>'
      : "";
  const note = detection.note && ["Expressions", "Gestures"].includes(detection.category)
    ? `<p class="detection-note">${esc(detection.note)}</p>`
    : "";
  return `
    <article class="detection-card" data-detection-id="${esc(detection.id)}" style="--accent:${esc(accent)}" tabindex="0">
      <div class="detection-head">
        <span class="detection-dot"></span>
        <span class="detection-label">${esc(detection.label)}</span>
        <span class="detection-category">${esc(detection.category)}</span>
        <span class="detection-confidence">${confidence}%</span>
      </div>
      <div class="confidence-track"><span style="width:${clamp(confidence, 2, 100)}%"></span></div>
      ${settings.descriptions ? `<p class="detection-desc">${esc(detection.description)}</p>` : ""}
      ${attributes.length || quantityChip ? `<div class="detection-attrs">${quantityChip}${attributes.join("")}</div>` : ""}
      ${note}
      ${referenceStripHTML(detection)}
      ${detection.source ? `<p class="detection-source">${esc(detection.source)}</p>` : ""}
    </article>`;
}

function attributeChips(detection, full) {
  const attributes = detection.attributes || {};
  const chips = [];
  const add = (label, value) => {
    if (value === undefined || value === null || value === "") return;
    chips.push(`<span class="attr-chip">${esc(label)} <strong>${esc(value)}</strong></span>`);
  };
  if (detection.category === "People") {
    if (attributes.gender) add("Gender (AI)", attributes.gender === "woman" ? "Woman" : "Man");
    if (attributes.age_range) add("Age prediction (AI)", attributes.age_range);
    if (typeof attributes.face_visible === "boolean") add("Face", attributes.face_visible ? "visible" : "not visible");
    if (attributes.age_confidence) add("Age confidence", formatConfidence(attributes.age_confidence));
    if (full && attributes.reason) add("Note", attributes.reason);
    if (full) add("Identity", "not recognized");
  }
  if (detection.details && detection.details.handedness) add("Hand", detection.details.handedness);
  if (detection.details && detection.details.sides) add("Sides", detection.details.sides.join(" + "));
  return chips;
}

function referenceStripHTML(detection) {
  if (!detection.reference_key) return "";
  const reference = detection.reference && detection.reference.status === "ready" ? detection.reference : null;
  const key = esc(detection.reference_key);
  const label = esc(detection.label);
  if (reference && reference.image) {
    return `
      <div class="ref-strip ready" role="button" tabindex="0" data-ref-key="${key}" data-ref-label="${label}" data-ref-ready="1">
        <span class="ref-thumb"><img loading="lazy" src="${esc(assetUrl(reference.image))}" alt="${esc(reference.title || "Visual reference")}"></span>
        <span class="ref-meta"><strong>${esc(reference.title || "Visual reference")}</strong><small>${esc(reference.source || "Wikimedia")} · cached locally</small></span>
        <span class="ref-meta" style="text-align:right"><small>Open</small></span>
      </div>`;
  }
  return `
    <div class="ref-strip" data-ref-key="${key}" data-ref-label="${label}">
      <span class="ref-thumb"><span class="ref-spinner"></span></span>
      <span class="ref-meta"><strong>Finding visual reference</strong><small>Wikimedia · cached locally</small></span>
    </div>`;
}

async function hydrateReferenceStrip(strip) {
  if (strip.dataset.refReady === "1") return;
  strip.dataset.refReady = "1";
  const key = strip.dataset.refKey;
  const label = strip.dataset.refLabel || key;
  let reference = state.refs.get(key);
  if (!reference) reference = await loadReference(key, label);
  if (!strip.isConnected) return;
  if (!reference || !reference.image) {
    strip.classList.remove("ready");
    strip.removeAttribute("role");
    strip.removeAttribute("tabindex");
    strip.innerHTML = `
      <span class="ref-thumb"><img src="${PLACEHOLDER_IMAGE}" alt=""></span>
      <span class="ref-meta"><strong>Reference unavailable</strong><small>Try again from a new scan</small></span>`;
    return;
  }
  state.refs.set(key, reference);
  strip.classList.add("ready");
  strip.setAttribute("role", "button");
  strip.setAttribute("tabindex", "0");
  strip.dataset.refLabel = label;
  strip.innerHTML = `
    <span class="ref-thumb"><img loading="lazy" src="${esc(assetUrl(reference.image))}" alt="${esc(reference.title || "Visual reference")}"></span>
    <span class="ref-meta"><strong>${esc(reference.title || "Visual reference")}</strong><small>${esc(reference.source || "Wikimedia")} · cached locally</small></span>
    <span class="ref-meta" style="text-align:right"><small>Open</small></span>`;
  bindImageFallbacks(strip);
}

function updateOverlaySize() {
  const rect = els.stage.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  els.overlay.width = Math.max(1, Math.round(rect.width * ratio));
  els.overlay.height = Math.max(1, Math.round(rect.height * ratio));
  overlayContext.setTransform(ratio, 0, 0, ratio, 0, 0);
  drawOverlay();
}

function clearOverlay() {
  const width = els.overlay.width;
  const height = els.overlay.height;
  overlayContext.setTransform(1, 0, 0, 1, 0, 0);
  overlayContext.clearRect(0, 0, width, height);
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  overlayContext.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function geometryForStage() {
  const kind = els.stage.dataset.kind;
  let naturalWidth = 0;
  let naturalHeight = 0;
  let fit = "contain";
  let mirror = false;
  if (kind === "camera" && els.camera.videoWidth) {
    naturalWidth = els.camera.videoWidth;
    naturalHeight = els.camera.videoHeight;
    fit = "cover";
    mirror = settings.mirror;
  } else if (kind === "image" && els.uploadImage.naturalWidth) {
    naturalWidth = els.uploadImage.naturalWidth;
    naturalHeight = els.uploadImage.naturalHeight;
  }
  if (!naturalWidth || !naturalHeight) return null;
  const width = els.overlay.clientWidth;
  const height = els.overlay.clientHeight;
  const scale = fit === "cover"
    ? Math.max(width / naturalWidth, height / naturalHeight)
    : Math.min(width / naturalWidth, height / naturalHeight);
  return {
    width,
    height,
    naturalWidth,
    naturalHeight,
    scale,
    offsetX: (width - naturalWidth * scale) / 2,
    offsetY: (height - naturalHeight * scale) / 2,
    mirror,
  };
}

function drawOverlay() {
  clearOverlay();
  if (!settings.boxes || !state.result || !Array.isArray(state.result.detections)) return;
  const geometry = geometryForStage();
  if (!geometry) return;
  const detections = state.result.detections.filter((detection) => detection.localized && Array.isArray(detection.bbox));
  if (!detections.length) return;
  const context = overlayContext;
  const hasSelection = detections.some((detection) => detection.id === state.selectedId);
  context.save();
  context.font = '600 11px "Space Grotesk", sans-serif';
  context.textBaseline = "middle";
  detections.slice(0, 18).forEach((detection) => {
    const [x, y, w, h] = detection.bbox;
    let x1 = geometry.offsetX + x * geometry.naturalWidth * geometry.scale;
    let x2 = geometry.offsetX + (x + w) * geometry.naturalWidth * geometry.scale;
    const y1 = geometry.offsetY + y * geometry.naturalHeight * geometry.scale;
    const y2 = geometry.offsetY + (y + h) * geometry.naturalHeight * geometry.scale;
    if (geometry.mirror) {
      const mirroredX1 = geometry.width - x2;
      const mirroredX2 = geometry.width - x1;
      x1 = mirroredX1;
      x2 = mirroredX2;
    }
    const boxWidth = Math.max(1, x2 - x1);
    const boxHeight = Math.max(1, y2 - y1);
    const selected = detection.id === state.selectedId;
    const accent = validColor(detection.color, colorFor(detection.category));
    context.globalAlpha = selected ? 1 : (hasSelection ? 0.5 : 1);
    context.strokeStyle = accent;
    context.lineWidth = selected ? 2.6 : 1.5;
    context.shadowColor = accent;
    context.shadowBlur = selected ? 14 : 5;
    roundedRect(context, x1, y1, boxWidth, boxHeight, 5);
    context.stroke();
    context.shadowBlur = 0;
    context.fillStyle = hexToRgba(accent, selected ? 0.2 : 0.1);
    context.fill();
    const text = `${detection.label} · ${Math.round((Number(detection.confidence) || 0) * 100)}%`;
    const textWidth = context.measureText(text).width;
    const labelWidth = textWidth + 14;
    const labelHeight = 19;
    let labelX = x1;
    let labelY = y1 - labelHeight - 5;
    if (labelY < 4) labelY = y1 + 5;
    if (labelX + labelWidth > geometry.width - 4) labelX = Math.max(4, geometry.width - labelWidth - 4);
    context.fillStyle = accent;
    context.globalAlpha = selected ? 1 : (hasSelection ? 0.55 : 0.94);
    roundedRect(context, labelX, labelY, labelWidth, labelHeight, 4);
    context.fill();
    context.fillStyle = "#03131c";
    context.globalAlpha = 1;
    context.fillText(text, labelX + 7, labelY + labelHeight / 2 + 0.5);
  });
  context.restore();
}

function roundedRect(context, x, y, width, height, radius) {
  const r = Math.min(radius, Math.abs(width) / 2, Math.abs(height) / 2);
  context.beginPath();
  context.moveTo(x + r, y);
  context.arcTo(x + width, y, x + width, y + height, r);
  context.arcTo(x + width, y + height, x, y + height, r);
  context.arcTo(x, y + height, x, y, r);
  context.arcTo(x, y, x + width, y, r);
  context.closePath();
}

function hexToRgba(hex, alpha) {
  const value = hex.replace("#", "");
  const normalized = value.length === 3
    ? value.split("").map((part) => part + part).join("")
    : value.slice(0, 6);
  const number = Number.parseInt(normalized, 16);
  if (Number.isNaN(number)) return `rgba(62, 233, 255, ${alpha})`;
  return `rgba(${(number >> 16) & 255}, ${(number >> 8) & 255}, ${number & 255}, ${alpha})`;
}

function selectDetection(id) {
  state.selectedId = state.selectedId === id ? null : id;
  els.resultsList.querySelectorAll(".detection-card").forEach((card) => {
    card.classList.toggle("selected", card.dataset.detectionId === state.selectedId);
  });
  drawOverlay();
}

function handleFile(file) {
  if (!file) return;
  if (!file.type || !file.type.startsWith("image/")) {
    toast("Please choose an image file.", "error");
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    toast("That image is larger than the 20 MB limit.", "error");
    return;
  }
  if (state.cameraOn) stopCamera({ keepFrame: false });
  abortScan();
  state.epoch += 1;
  state.usingImage = true;
  state.imageBlob = file;
  state.uploadName = file.name;
  if (state.uploadedUrl) URL.revokeObjectURL(state.uploadedUrl);
  state.uploadedUrl = URL.createObjectURL(file);
  els.uploadImage.onload = () => updateOverlaySize();
  els.uploadImage.src = state.uploadedUrl;
  els.stage.dataset.kind = "image";
  els.stageBadge.textContent = "PHOTO";
  els.cameraState.innerHTML = '<span class="dot"></span>Image ready';
  els.cameraState.classList.remove("live");
  resetScanOutput();
  updateOverlaySize();
  window.setTimeout(runUploadScan, 120);
}

function triggerUpload() {
  els.fileInput.click();
}

function applySettings(updateDescriptions = false) {
  els.stage.classList.toggle("mirror", settings.mirror);
  els.scanConfidence.value = String(settings.confidence);
  els.scanConfidenceValue.textContent = `${settings.confidence}%`;
  if (state.cameraOn) {
    els.cameraState.innerHTML = `<span class="dot"></span>Live · ${(settings.interval / 1000).toFixed(1)}s scan`;
  }
  if (updateDescriptions && state.result) renderDetections(state.result.detections || []);
  drawOverlay();
}
