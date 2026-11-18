"use strict";

const DEFAULT_SETTINGS = {
  confidence: Number(document.body.dataset.defaultConfidence) || 35,
  interval: 400,
  autoSave: true,
  mirror: true,
  boxes: true,
  descriptions: true,
};

const MODES = {
  smart: {
    title: "Smart Scan",
    eyebrow: "Smart Scan",
    description: "One pass for people, everyday objects and visible expressions.",
    capability: "Smart Scan combines person, object, expression and age-range detection. Visual estimates are approximate and are labeled accordingly.",
    engine: "YOLOv8n + FairFace (age) + MediaPipe Face",
  },
  person: {
    title: "Person Detection",
    eyebrow: "Person Detection",
    description: "Locate people, visible expressions and clearly-labeled approximate age ranges.",
    capability: "Man/woman and age ranges are uncertain AI estimates computed from visible facial features. They are not facts, exact ages or identity checks.",
    engine: "YOLOv8n + FairFace (age)",
  },
  object: {
    title: "Object Detection",
    eyebrow: "Object Detection",
    description: "Recognise the 80 everyday COCO object classes in a live camera or uploaded photo.",
    capability: "Universal scanner uses YOLOv8n trained on the 80 COCO object classes. Objects outside this vocabulary are not claimed as recognized.",
    engine: "YOLOv8n · 80 COCO classes",
  },
  fruit: {
    title: "Fruit Recognition",
    eyebrow: "Fruit Recognition",
    description: "Localise common fruit and vegetables, estimate quantity and report honest coverage limits.",
    capability: "Fruit scanner combines YOLO localization with an ImageNet food classifier. Fruits outside its vocabulary (for example mango, grapes, watermelon, papaya or dragon fruit) are marked as unsupported instead of guessed.",
    engine: "YOLOv8n + MobileNetV3",
  },
  gesture: {
    title: "Gesture Scanner",
    eyebrow: "Gesture Scanner",
    description: "Read posture, hand signals and basic visible expressions from body landmarks.",
    capability: "Gestures and expressions are estimated from body, hand and facial landmarks. Temporal gestures need several live frames. Expressions do not reveal true emotions or personality.",
    engine: "MediaPipe Pose · Hands · Face",
  },
  camera: {
    title: "Camera Detection",
    eyebrow: "Live camera",
    description: "Start your webcam or mobile camera and let the AI scan people, objects and expressions continuously.",
    capability: "Live camera frames are analysed on this computer. Bounding boxes, labels and confidence scores update automatically; nothing is uploaded to a cloud AI service.",
    engine: "YOLOv8n + FairFace (age) + MediaPipe Face",
  },
};

const CAMERA_SCAN_MODE = "smart";

const CATEGORY_COLORS = {
  People: "#3ee9ff",
  Electronics: "#4d9bff",
  Furniture: "#a78bfa",
  Vehicles: "#f7b955",
  Animals: "#5be7a9",
  Fruits: "#ffb347",
  Vegetables: "#8bd450",
  Food: "#ff9366",
  "Food & Drink": "#ff9366",
  Gestures: "#ffd166",
  Pose: "#b287ff",
  Expressions: "#ff7ad9",
  Sports: "#59d2fe",
  Accessories: "#f78fb3",
  Clothing: "#f78fb3",
  Home: "#74d3c6",
  Appliances: "#64b5f6",
  Stationery: "#c6a7ff",
  Urban: "#9db2cc",
  Toys: "#f9a8d4",
  "Personal Care": "#7dd3fc",
  Object: "#879bb8",
  Unknown: "#9db2cc",
};

const SCANNER_LABELS = {
  smart: "Smart Scan",
  person: "Person Detection",
  object: "Object Detection",
  fruit: "Fruit Recognition",
  gesture: "Gesture Scanner",
  camera: "Camera Detection",
};

const MODEL_SHORT = {
  yolo: "YOLO",
  faces: "Faces",
  fairface: "Age",
  mediapipe: "MediaPipe",
  classifier: "Food",
};

function backendMode(mode = state.mode) {
  return mode === "camera" ? CAMERA_SCAN_MODE : mode;
}

const PLACEHOLDER_IMAGE = "/static/img/reference-placeholder.svg";
const assetUrl = (url) => url;

const state = {
  view: "dashboard",
  mode: "smart",
  result: null,
  selectedId: null,
  scanning: false,
  scanToken: 0,
  epoch: 0,
  cameraOn: false,
  stream: null,
  sessionId: null,
  loopTimer: null,
  controller: null,
  usingImage: false,
  imageBlob: null,
  uploadName: null,
  uploadedUrl: null,
  pendingUploadMode: null,
  health: null,
  healthTimer: null,
  lastSaveAt: 0,
  statsDirty: false,
  historyById: new Map(),
  historyOffset: 0,
  historyLimit: 24,
  recentById: new Map(),
  refs: new Map(),
  refPromises: new Map(),
  modalToken: 0,
  themeSetting: "dark",
  theme: "dark",
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const els = {
  body: document.body,
  sidebar: $("#sidebar"),
  sidebarScrim: $("#sidebar-scrim"),
  menuToggle: $("#menu-toggle"),
  pageTitle: $("#page-title"),
  aiStatusPill: $("#ai-status-pill"),
  aiStatusText: $("#ai-status-text"),
  systemDot: $("#system-dot"),
  systemState: $("#system-state"),
  systemUptime: $("#system-uptime"),
  modelList: $("#model-list"),
  statScans: $("#stat-scans"),
  statDetections: $("#stat-detections"),
  statPeople: $("#stat-people"),
  stat24h: $("#stat-24h"),
  statObjects: $("#stat-objects"),
  statFruits: $("#stat-fruits"),
  statExpressions: $("#stat-expressions"),
  statConfidence: $("#stat-confidence"),
  expressionList: $("#expression-list"),
  expressionTotal: $("#expression-total"),
  themeToggle: $("#theme-toggle"),
  setTheme: $("#set-theme"),
  weekTotal: $("#week-total"),
  activityChart: $("#activity-chart"),
  recentList: $("#recent-list"),
  scanEyebrow: $("#scan-eyebrow"),
  scanTitle: $("#scan-title"),
  scanDesc: $("#scan-desc"),
  scanConfidence: $("#scan-confidence"),
  scanConfidenceValue: $("#scan-confidence-value"),
  stage: $("#stage"),
  camera: $("#camera"),
  uploadImage: $("#upload-image"),
  overlay: $("#overlay"),
  stagePlaceholder: $("#stage-placeholder"),
  stageBadge: $("#stage-badge"),
  scanSpinner: $("#scan-spinner"),
  scanline: $("#scanline"),
  cameraStart: $("#camera-start"),
  cameraStop: $("#camera-stop"),
  scanNow: $("#scan-now"),
  uploadTrigger: $("#upload-trigger"),
  placeholderStart: $("#placeholder-start"),
  placeholderUpload: $("#placeholder-upload"),
  cameraState: $("#camera-state"),
  metaEngine: $("#meta-engine"),
  metaInference: $("#meta-inference"),
  metaCount: $("#meta-count"),
  metaImage: $("#meta-image"),
  scanNotes: $("#scan-notes"),
  scanNotesList: $("#scan-notes-list"),
  capabilityNote: $("#capability-note"),
  resultCount: $("#result-count"),
  resultsStatus: $("#results-status"),
  resultsList: $("#results-list"),
  scanSummary: $("#scan-summary"),
  scanSummaryText: $("#scan-summary-text"),
  detailModal: $("#detail-modal"),
  modalImage: $("#modal-image"),
  modalCategory: $("#modal-category"),
  modalScanner: $("#modal-scanner"),
  modalTitle: $("#modal-title"),
  modalConfidence: $("#modal-confidence"),
  modalMeta: $("#modal-meta"),
  modalDesc: $("#modal-desc"),
  modalAttrs: $("#modal-attrs"),
  modalNote: $("#modal-note"),
  modalReference: $("#modal-reference"),
  modalReferenceLink: $("#modal-reference-link"),
  modalReferenceImage: $("#modal-reference-image"),
  modalReferenceTitle: $("#modal-reference-title"),
  fileInput: $("#file-input"),
  toastStack: $("#toast-stack"),
  historyList: $("#history-list"),
  historyEmpty: $("#history-empty"),
  historyTotal: $("#history-total"),
  historyFilter: $("#history-filter"),
  historyRefresh: $("#history-refresh"),
  historyClear: $("#history-clear"),
  historyLoadMore: $("#history-load-more"),
  settingsForm: $("#settings-form"),
  setConfidence: $("#set-confidence"),
  setConfidenceValue: $("#set-confidence-value"),
  setInterval: $("#set-interval"),
  setAutoSave: $("#set-autosave"),
  setMirror: $("#set-mirror"),
  setBoxes: $("#set-boxes"),
  setDescriptions: $("#set-descriptions"),
  resetSettings: $("#reset-settings"),
  settingsClearHistory: $("#settings-clear-history"),
  settingsUnsupported: $("#settings-unsupported"),
  heroStart: $("#hero-start"),
  heroUpload: $("#hero-upload"),
};

let settings = loadSettings();

function loadSettings() {
  let stored = {};
  try {
    stored = JSON.parse(localStorage.getItem("visionscan.settings") || "{}");
  } catch {
    stored = {};
  }
  const next = { ...DEFAULT_SETTINGS, ...stored };
  next.confidence = clamp(Number(next.confidence) || DEFAULT_SETTINGS.confidence, 15, 90);
  next.interval = clamp(Number(next.interval) || DEFAULT_SETTINGS.interval, 400, 6000);
  next.autoSave = Boolean(next.autoSave);
  next.mirror = Boolean(next.mirror);
  next.boxes = Boolean(next.boxes);
  next.descriptions = Boolean(next.descriptions);
  return next;
}

function persistSettings() {
  try {
    localStorage.setItem("visionscan.settings", JSON.stringify(settings));
  } catch {
    return;
  }
}

const THEME_KEY = "visionscan.theme";

function loadThemeSetting() {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    if (["dark", "light", "auto"].includes(stored)) return stored;
  } catch {
    /* storage can be blocked */
  }
  return "dark";
}

function systemTheme() {
  try {
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  } catch {
    return "dark";
  }
}

function applyTheme(setting, persist = false) {
  state.themeSetting = ["dark", "light", "auto"].includes(setting) ? setting : "dark";
  state.theme = state.themeSetting === "auto" ? systemTheme() : state.themeSetting;
  document.documentElement.dataset.theme = state.theme;
  const themeMeta = document.querySelector('meta[name="theme-color"]');
  if (themeMeta) themeMeta.setAttribute("content", state.theme === "light" ? "#eef4fb" : "#050b16");
  if (persist) {
    try {
      localStorage.setItem(THEME_KEY, state.themeSetting);
    } catch {
      /* storage can be blocked */
    }
  }
  if (els.setTheme) els.setTheme.value = state.themeSetting;
  if (els.themeToggle) {
    const label = state.theme === "light" ? "Switch to dark theme" : "Switch to light theme";
    els.themeToggle.setAttribute("aria-label", label);
    els.themeToggle.setAttribute("title", label);
  }
}

function toggleTheme() {
  applyTheme(state.theme === "light" ? "dark" : "light", true);
  toast(`${state.theme === "light" ? "Light" : "Dark"} theme enabled.`, "info", 2200);
}

function initTheme() {
  applyTheme(loadThemeSetting(), false);
  try {
    const media = window.matchMedia("(prefers-color-scheme: light)");
    const listener = () => {
      if (state.themeSetting === "auto") applyTheme("auto", false);
    };
    if (media.addEventListener) media.addEventListener("change", listener);
    else if (media.addListener) media.addListener(listener);
  } catch {
    /* matchMedia unavailable */
  }
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character]);
}

function colorFor(category) {
  return CATEGORY_COLORS[category] || "#3ee9ff";
}

function validColor(value, fallback) {
  return /^#[0-9a-f]{3,8}$/i.test(value || "") ? value : fallback;
}

function scannerLabel(mode) {
  return SCANNER_LABELS[mode] || "Scan";
}

function formatCount(value) {
  return new Intl.NumberFormat().format(Number(value) || 0);
}

function formatConfidence(value) {
  return `${Math.round((Number(value) || 0) * 100)}%`;
}

function formatDateTime(timestamp) {
  try {
    return new Date(Number(timestamp) * 1000).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "";
  }
}

function formatTimeAgo(timestamp) {
  const seconds = Math.max(0, (Date.now() - Number(timestamp) * 1000) / 1000);
  if (seconds < 45) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  return formatDateTime(timestamp);
}

function formatUptime(seconds) {
  const total = Math.round(Number(seconds) || 0);
  if (total < 60) return `${total}s uptime`;
  if (total < 3600) return `${Math.floor(total / 60)}m uptime`;
  return `${Math.floor(total / 3600)}h ${Math.floor((total % 3600) / 60)}m uptime`;
}

function toast(message, type = "success", duration = 4200) {
  const element = document.createElement("div");
  element.className = `toast ${type}`;
  element.textContent = message;
  els.toastStack.appendChild(element);
  window.setTimeout(() => {
    element.classList.add("hide");
    window.setTimeout(() => element.remove(), 260);
  }, duration);
}

function bindImageFallbacks(container) {
  container.querySelectorAll("img").forEach((image) => {
    if (image.dataset.fallbackBound) return;
    image.dataset.fallbackBound = "1";
    image.addEventListener("error", () => {
      image.src = PLACEHOLDER_IMAGE;
    }, { once: true });
  });
}

function loadReference(key, label) {
  if (!key) return Promise.resolve(null);
  if (state.refPromises.has(key)) return state.refPromises.get(key);
  const promise = (async () => {
    for (let attempt = 0; attempt < 4; attempt += 1) {
      try {
        const response = await fetch(`/api/reference?key=${encodeURIComponent(key)}&label=${encodeURIComponent(label || key)}`);
        if (response.status === 202) {
          await sleep(1800);
          continue;
        }
        if (!response.ok) return null;
        const data = await response.json();
        if (data && data.status === "ready") return data;
        return null;
      } catch {
        return null;
      }
    }
    return null;
  })().then((data) => {
    if (!data) state.refPromises.delete(key);
    return data;
  });
  state.refPromises.set(key, promise);
  return promise;
}

function networkErrorText(error) {
  if (error && error.name === "AbortError") return "Request cancelled.";
  if (error && error.name === "TypeError") return "Could not reach the AI server. Make sure the backend is running and the link is current.";
  return error && error.message ? error.message : "Something went wrong.";
}

function newSessionId() {
  if (window.crypto && window.crypto.randomUUID) return window.crypto.randomUUID();
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function setStatusState(kind, text) {
  els.aiStatusPill.classList.remove("ready", "warn", "bad");
  els.systemDot.classList.remove("warn", "bad", "good");
  if (kind === "ready") {
    els.aiStatusPill.classList.add("ready");
    els.systemDot.classList.add("good");
  } else if (kind === "warn") {
    els.aiStatusPill.classList.add("warn");
    els.systemDot.classList.add("warn");
  } else if (kind === "bad") {
    els.aiStatusPill.classList.add("bad");
    els.systemDot.classList.add("bad");
  }
  els.aiStatusText.textContent = text;
  els.systemState.textContent = text;
}

async function refreshHealth() {
  clearTimeout(state.healthTimer);
  let delay = 15000;
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error("Health request failed");
    const health = await response.json();
    state.health = health;
    const models = Array.isArray(health.models) ? health.models : [];
    const errors = models.filter((model) => model.state === "error");
    const readyCount = models.filter((model) => model.state === "ready").length;
    if (health.ready) {
      setStatusState("ready", "AI Online");
      delay = 30000;
    } else if (errors.length) {
      setStatusState("bad", "AI Degraded");
      delay = 8000;
    } else {
      setStatusState("warn", `Loading AI ${readyCount}/${models.length}`);
      delay = 4000;
    }
    els.modelList.innerHTML = models.map((model) => {
      const short = MODEL_SHORT[model.key] || model.name;
      const title = `${model.name}: ${model.state}${model.error ? ` (${model.error})` : ""}`;
      return `<span class="model-chip ${esc(model.state)}" title="${esc(title)}">${esc(short)}</span>`;
    }).join("");
    els.systemUptime.textContent = formatUptime(health.uptime_seconds);
    if (Array.isArray(health.unsupported_fruits) && health.unsupported_fruits.length) {
      els.settingsUnsupported.textContent = `${health.unsupported_fruits.join(", ")} are not in the supported fruit vocabulary and will not be guessed. Objects outside the COCO training set are also not claimed as recognised.`;
    }
  } catch {
    setStatusState("warn", "Reconnecting…");
    els.systemUptime.textContent = "—";
  } finally {
    state.healthTimer = window.setTimeout(refreshHealth, delay);
  }
}

function updateNavActive() {
  $$(".nav-item.active").forEach((item) => item.classList.remove("active"));
  if (state.view === "scanner") {
    if (state.mode === "camera") {
      const cameraItem = document.querySelector(".nav-item[data-camera-link]");
      if (cameraItem) cameraItem.classList.add("active");
      return;
    }
    const item = document.querySelector(`.nav-item[data-mode-link="${state.mode}"]`);
    if (item) item.classList.add("active");
    return;
  }
  const item = document.querySelector(`.nav-item[data-view-link="${state.view}"]`);
  if (item) item.classList.add("active");
}

function updateModeCopy() {
  const mode = MODES[state.mode] || MODES.smart;
  els.scanEyebrow.textContent = mode.eyebrow;
  els.scanTitle.textContent = mode.title;
  els.scanDesc.textContent = mode.description;
  els.capabilityNote.textContent = mode.capability;
  els.metaEngine.textContent = mode.engine;
  els.scanConfidence.value = String(settings.confidence);
  els.scanConfidenceValue.textContent = `${settings.confidence}%`;
  els.cameraState.innerHTML = state.cameraOn
    ? `<span class="dot"></span>Live · ${(settings.interval / 1000).toFixed(1)}s scan`
    : `<span class="dot"></span>Camera idle`;
  els.cameraState.classList.toggle("live", state.cameraOn);
  if (state.view === "scanner") {
    els.pageTitle.textContent = mode.title;
    document.title = `${mode.title} · AI Object Visualizer`;
  }
  if (!state.cameraOn && !state.usingImage) {
    els.stageBadge.textContent = "STANDBY";
  }
}

function showView(view) {
  if (view !== "scanner" && state.cameraOn) {
    stopCamera({ keepFrame: true });
  }
  state.view = view;
  $$(".view").forEach((section) => {
    section.hidden = section.id !== `view-${view}`;
  });
  updateNavActive();
  if (view === "scanner") {
    els.pageTitle.textContent = MODES[state.mode].title;
    document.title = `${MODES[state.mode].title} · AI Object Visualizer`;
    window.setTimeout(updateOverlaySize, 40);
  } else if (view === "dashboard") {
    els.pageTitle.textContent = "Dashboard";
    document.title = "Dashboard · AI Object Visualizer";
    refreshDashboard();
  } else if (view === "similar") {
    els.pageTitle.textContent = "Visual Similarity Search";
    document.title = "Visual Similarity Search · AI Object Visualizer";
    if (typeof window.activateSimilarView === "function") window.activateSimilarView();
  } else if (view === "history") {
    els.pageTitle.textContent = "Analysis History";
    document.title = "Analysis History · AI Object Visualizer";
    loadHistory(true);
  } else if (view === "settings") {
    els.pageTitle.textContent = "Settings";
    document.title = "Settings · AI Object Visualizer";
    syncSettingsUI();
  }
  closeSidebar();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function selectMode(mode) {
  if (!MODES[mode]) mode = "smart";
  const changed = state.mode !== mode;
  state.mode = mode;
  updateModeCopy();
  updateNavActive();
  if (state.view === "scanner") {
    els.pageTitle.textContent = MODES[mode].title;
    document.title = `${MODES[mode].title} · AI Object Visualizer`;
  }
  if (changed) {
    abortScan();
    resetScanOutput();
    if (state.cameraOn) {
      state.sessionId = newSessionId();
      startLiveLoop(160);
    }
  }
}

function openScanner(mode, options = {}) {
  const sameView = state.view === "scanner";
  selectMode(mode);
  if (!sameView) showView("scanner");
  updateNavActive();
  const autoStart = options.autoStart === undefined ? mode === "camera" : Boolean(options.autoStart);
  if (autoStart && !state.cameraOn) {
    startCamera();
  }
}

function openSidebar() {
  els.body.classList.add("sidebar-open");
}

function closeSidebar() {
  els.body.classList.remove("sidebar-open");
}
