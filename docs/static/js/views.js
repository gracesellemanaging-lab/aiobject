"use strict";

let historyRequestToken = 0;
let resizeTimer = null;

async function refreshDashboard() {
  try {
    const response = await fetch("/api/stats", { cache: "no-store" });
    if (!response.ok) throw new Error("Stats unavailable");
    const stats = await response.json();
    els.statScans.textContent = formatCount(stats.total_scans);
    els.statDetections.textContent = formatCount(stats.total_detections);
    els.statPeople.textContent = formatCount(stats.people);
    els.stat24h.textContent = formatCount(stats.detections_24h);
    if (els.statObjects) els.statObjects.textContent = formatCount(stats.objects);
    if (els.statFruits) els.statFruits.textContent = formatCount(stats.fruits);
    if (els.statExpressions) els.statExpressions.textContent = formatCount(stats.expressions);
    if (els.statConfidence) els.statConfidence.textContent = `${Number(stats.avg_confidence_percent || 0).toFixed(1)}%`;
    els.weekTotal.textContent = `${formatCount(stats.detections_7d)} detection${Number(stats.detections_7d) === 1 ? "" : "s"}`;
    renderChart(stats.daily || []);
    renderExpressions(stats.by_expression || [], stats.expressions || 0);
  } catch {
    els.weekTotal.textContent = "Activity unavailable";
  }
  try {
    const response = await fetch("/api/history?limit=6", { cache: "no-store" });
    if (!response.ok) throw new Error("History unavailable");
    const data = await response.json();
    state.recentById.clear();
    (data.items || []).forEach((item) => state.recentById.set(String(item.id), item));
    renderRecent(data.items || []);
  } catch {
    els.recentList.innerHTML = '<div class="mini-empty">Recent activity is unavailable.</div>';
  }
  state.statsDirty = false;
}

function renderChart(daily) {
  if (!daily.length) {
    els.activityChart.innerHTML = "";
    return;
  }
  const maximum = Math.max(1, ...daily.map((day) => Number(day.count) || 0));
  const today = new Date();
  const todayKey = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`;
  els.activityChart.innerHTML = daily.map((day) => {
    const count = Number(day.count) || 0;
    const height = count === 0 ? 3 : Math.max(8, Math.round((count / maximum) * 100));
    const date = new Date(`${day.date}T00:00:00`);
    const label = Number.isNaN(date.getTime()) ? day.date.slice(5) : date.toLocaleDateString(undefined, { weekday: "short" });
    const isToday = day.date === todayKey;
    return `
      <div class="chart-col${isToday ? " today" : ""}" title="${esc(day.date)} · ${count} detections">
        <div class="chart-bar-wrap">
          <div class="chart-bar" style="height:${height}%"><span>${count}</span></div>
        </div>
        <span class="chart-day">${esc(label)}</span>
      </div>`;
  }).join("");
}

function renderExpressions(items, total) {
  if (!els.expressionList) return;
  els.expressionTotal.textContent = `${formatCount(total)} saved`;
  if (!items.length) {
    els.expressionList.innerHTML = '<div class="mini-empty">Expression estimates appear after a Smart Scan or Person Detection scan.</div>';
    return;
  }
  const maximum = Math.max(1, ...items.map((item) => Number(item.count) || 0));
  els.expressionList.innerHTML = items.map((item) => {
    const count = Number(item.count) || 0;
    return `
      <div class="expression-row">
        <span class="expression-name">${esc(item.label)}</span>
        <span class="expression-track"><span style="width:${Math.max(6, Math.round((count / maximum) * 100))}%"></span></span>
        <span class="expression-count">${formatCount(count)} · ${Number(item.confidence_percent || 0).toFixed(0)}%</span>
      </div>`;
  }).join("");
}

function renderRecent(items) {
  if (!items.length) {
    els.recentList.innerHTML = '<div class="mini-empty">No saved scans yet. Start with Smart Scan.</div>';
    return;
  }
  els.recentList.innerHTML = items.map((item) => {
    const accent = colorFor(item.category);
    const thumb = item.snapshot_url
      ? `<span class="recent-thumb" style="--accent:${esc(accent)}"><img loading="lazy" src="${esc(assetUrl(item.snapshot_url))}" alt=""></span>`
      : `<span class="recent-thumb" style="--accent:${esc(accent)}">${esc(String(item.label || "?").slice(0, 2).toUpperCase())}</span>`;
    return `
      <button class="recent-item" data-recent-id="${esc(item.id)}">
        ${thumb}
        <span class="recent-info"><strong>${esc(item.label)}</strong><small>${esc(scannerLabel(item.scanner))} · ${esc(formatTimeAgo(item.timestamp))}</small></span>
        <span class="recent-confidence">${Math.round(Number(item.confidence || 0) * 100)}%</span>
      </button>`;
  }).join("");
  bindImageFallbacks(els.recentList);
}

async function loadHistory(reset = true) {
  historyRequestToken += 1;
  const token = historyRequestToken;
  if (reset) {
    state.historyOffset = 0;
    state.historyById.clear();
    els.historyList.innerHTML = "";
    els.historyLoadMore.hidden = true;
  }
  els.historyLoadMore.disabled = true;
  const params = new URLSearchParams({
    limit: String(state.historyLimit),
    offset: String(state.historyOffset),
  });
  const scanner = els.historyFilter.value;
  if (scanner) params.set("scanner", scanner);
  try {
    const response = await fetch(`/api/history?${params.toString()}`, { cache: "no-store" });
    if (!response.ok) throw new Error("History request failed");
    const data = await response.json();
    if (token !== historyRequestToken) return;
    const items = data.items || [];
    items.forEach((item) => state.historyById.set(String(item.id), item));
    state.historyOffset += items.length;
    els.historyTotal.textContent = `${formatCount(data.total)} saved detection${Number(data.total) === 1 ? "" : "s"}`;
    els.historyList.insertAdjacentHTML("beforeend", items.map(historyCardHTML).join(""));
    bindImageFallbacks(els.historyList);
    els.historyEmpty.hidden = state.historyOffset > 0;
    els.historyLoadMore.hidden = state.historyOffset >= Number(data.total);
    els.historyLoadMore.disabled = false;
  } catch {
    if (token === historyRequestToken) toast("Scan history could not be loaded.", "error");
  }
}

function historyCardHTML(item) {
  const accent = validColor(colorFor(item.category), "#3ee9ff");
  const image = item.snapshot_url
    ? `<img loading="lazy" src="${esc(assetUrl(item.snapshot_url))}" alt="${esc(item.label)}">`
    : `<div class="history-placeholder">
         <svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/><circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="1.6"/></svg>
         <span>${esc(item.category || "Detection")}</span>
       </div>`;
  return `
    <button class="history-card" data-history-id="${esc(item.id)}" style="--accent:${esc(accent)}">
      <span class="history-thumb">
        ${image}
        <span class="history-scanner">${esc(scannerLabel(item.scanner))}</span>
        <span class="history-confidence">${Math.round(Number(item.confidence || 0) * 100)}%</span>
      </span>
      <span class="history-body">
        <strong>${esc(item.label)}</strong>
        <small>${esc(formatDateTime(item.timestamp))} · ${esc(item.category || "Detection")}</small>
        ${item.note ? `<span class="history-note">${esc(item.note)}</span>` : ""}
      </span>
    </button>`;
}

function openModal() {
  els.detailModal.hidden = false;
  els.detailModal.setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
}

function closeModal() {
  state.modalToken += 1;
  els.detailModal.hidden = true;
  els.detailModal.setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
}

function setModalImage(source) {
  els.modalImage.onerror = () => {
    els.modalImage.onerror = null;
    els.modalImage.src = PLACEHOLDER_IMAGE;
  };
  els.modalImage.src = source || PLACEHOLDER_IMAGE;
}

function openDetail(item) {
  if (!item) return;
  const token = ++state.modalToken;
  setModalImage(item.snapshot_url || PLACEHOLDER_IMAGE);
  els.modalImage.alt = `${item.label || "Detection"} snapshot`;
  els.modalCategory.textContent = item.category || "Detection";
  els.modalCategory.style.color = colorFor(item.category);
  els.modalScanner.textContent = scannerLabel(item.scanner);
  els.modalTitle.textContent = item.label || "Detection";
  els.modalConfidence.hidden = false;
  els.modalConfidence.textContent = formatConfidence(item.confidence);
  els.modalMeta.innerHTML = [
    `<span>${esc(formatDateTime(item.timestamp))}</span>`,
    item.source ? `<span>${esc(item.source)}</span>` : "",
    item.raw_label ? `<span>${esc(item.raw_label)}</span>` : "",
  ].filter(Boolean).join("");
  els.modalDesc.textContent = item.description || "No description was stored for this detection.";
  const chips = attributeChips({
    category: item.category,
    attributes: item.attributes || {},
    details: item.details || {},
  }, true);
  els.modalAttrs.innerHTML = chips.join("");
  const note = item.note || (item.notes && item.notes[0]) || "";
  els.modalNote.hidden = !note;
  els.modalNote.textContent = note;
  els.modalReference.hidden = !item.reference_key;
  if (item.reference_key) {
    els.modalReferenceTitle.textContent = "Loading reference";
    els.modalReferenceImage.removeAttribute("src");
    els.modalReferenceLink.href = "#";
    const cached = state.refs.get(item.reference_key);
    const referencePromise = cached ? Promise.resolve(cached) : loadReference(item.reference_key, item.label);
    referencePromise.then((reference) => {
      if (token !== state.modalToken) return;
      if (!reference || !reference.image) {
        els.modalReference.hidden = true;
        return;
      }
      state.refs.set(item.reference_key, reference);
      els.modalReference.hidden = false;
      els.modalReferenceTitle.textContent = reference.title || "Visual reference";
      els.modalReferenceImage.src = reference.image;
      els.modalReferenceImage.alt = reference.title || "Visual reference";
      els.modalReferenceImage.onerror = () => {
        els.modalReferenceImage.onerror = null;
        els.modalReferenceImage.src = PLACEHOLDER_IMAGE;
      };
      els.modalReferenceLink.href = reference.page_url || reference.image;
    });
  }
  openModal();
}

function openReferenceModal(reference, label) {
  if (!reference) return;
  state.modalToken += 1;
  setModalImage(reference.image);
  els.modalImage.alt = reference.title || label || "Visual reference";
  els.modalCategory.textContent = "Online reference";
  els.modalCategory.style.color = "#3ee9ff";
  els.modalScanner.textContent = "Wikimedia reference";
  els.modalTitle.textContent = reference.title || label || "Visual reference";
  els.modalConfidence.hidden = true;
  els.modalMeta.innerHTML = [
    `<span>${esc(reference.source || "Wikimedia")}</span>`,
    `<span>Cached on this computer</span>`,
    `<span>Public educational image</span>`,
  ].join("");
  els.modalDesc.textContent = "This public reference image is shown to help you visually compare the detected item. It is not part of the detection result.";
  els.modalAttrs.innerHTML = "";
  els.modalNote.hidden = true;
  els.modalReference.hidden = false;
  els.modalReferenceTitle.textContent = reference.title || "Open source";
  els.modalReferenceImage.src = reference.image;
  els.modalReferenceImage.alt = reference.title || "Visual reference";
  els.modalReferenceImage.onerror = () => {
    els.modalReferenceImage.onerror = null;
    els.modalReferenceImage.src = PLACEHOLDER_IMAGE;
  };
  els.modalReferenceLink.href = reference.page_url || reference.image;
  openModal();
}

function openHistoryItem(id) {
  const key = String(id);
  const item = state.historyById.get(key) || state.recentById.get(key);
  if (item) openDetail(item);
}

async function clearHistory() {
  const confirmed = window.confirm("Delete every saved detection and thumbnail? This cannot be undone.");
  if (!confirmed) return;
  try {
    const response = await fetch("/api/history", { method: "DELETE" });
    if (!response.ok) throw new Error("Clear failed");
    state.historyById.clear();
    state.recentById.clear();
    historyRequestToken += 1;
    toast("Scan history cleared.");
    if (state.view === "history") {
      await loadHistory(true);
    }
    if (state.view === "dashboard") refreshDashboard();
  } catch {
    toast("History could not be cleared.", "error");
  }
}

function syncSettingsUI() {
  els.setConfidence.value = String(settings.confidence);
  els.setConfidenceValue.textContent = `${settings.confidence}%`;
  els.scanConfidence.value = String(settings.confidence);
  els.scanConfidenceValue.textContent = `${settings.confidence}%`;
  els.setInterval.value = String(settings.interval);
  els.setAutoSave.checked = settings.autoSave;
  els.setMirror.checked = settings.mirror;
  els.setBoxes.checked = settings.boxes;
  els.setDescriptions.checked = settings.descriptions;
  if (els.setTheme) els.setTheme.value = state.themeSetting;
}

function bindEvents() {
  $$("[data-view-link]").forEach((element) => {
    element.addEventListener("click", (event) => {
      event.preventDefault();
      showView(element.dataset.viewLink);
    });
  });

  $$("[data-mode-link]").forEach((element) => {
    element.addEventListener("click", () => {
      openScanner(element.dataset.modeLink);
    });
  });

  $$("[data-camera-link]").forEach((element) => {
    element.addEventListener("click", () => {
      openScanner("camera", { autoStart: true });
    });
  });

  $$("[data-open-mode]").forEach((element) => {
    element.addEventListener("click", () => {
      openScanner(element.dataset.openMode);
    });
  });

  if (els.themeToggle) els.themeToggle.addEventListener("click", toggleTheme);
  if (els.setTheme) {
    els.setTheme.addEventListener("change", () => {
      applyTheme(els.setTheme.value, true);
    });
  }

  els.menuToggle.addEventListener("click", openSidebar);
  els.sidebarScrim.addEventListener("click", closeSidebar);
  els.aiStatusPill.addEventListener("click", refreshHealth);

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeSidebar();
      if (!els.detailModal.hidden) closeModal();
    }
  });

  $$("[data-close-modal]").forEach((element) => {
    element.addEventListener("click", closeModal);
  });

  els.heroStart.addEventListener("click", () => {
    openScanner("smart", { autoStart: true });
  });
  els.heroUpload.addEventListener("click", () => {
    state.pendingUploadMode = "smart";
    triggerUpload();
  });
  els.placeholderStart.addEventListener("click", startCamera);
  els.placeholderUpload.addEventListener("click", () => {
    state.pendingUploadMode = null;
    triggerUpload();
  });
  els.cameraStart.addEventListener("click", startCamera);
  els.cameraStop.addEventListener("click", () => stopCamera({ keepFrame: true }));
  els.uploadTrigger.addEventListener("click", () => {
    state.pendingUploadMode = null;
    triggerUpload();
  });
  els.scanNow.addEventListener("click", () => {
    if (state.scanning) {
      toast("A scan is already in progress.", "info", 2200);
      return;
    }
    if (state.cameraOn) {
      runCameraScan({ manual: true });
    } else if (state.usingImage && state.imageBlob) {
      runUploadScan();
    } else {
      toast("Start the camera or upload an image first.", "info", 3000);
    }
  });

  els.fileInput.addEventListener("change", () => {
    const file = els.fileInput.files && els.fileInput.files[0];
    els.fileInput.value = "";
    if (!file) return;
    const pending = state.pendingUploadMode;
    state.pendingUploadMode = null;
    if (pending) openScanner(pending);
    handleFile(file);
  });

  els.scanConfidence.addEventListener("input", () => {
    settings.confidence = clamp(Number(els.scanConfidence.value) || 35, 15, 90);
    els.scanConfidenceValue.textContent = `${settings.confidence}%`;
    els.setConfidence.value = String(settings.confidence);
    els.setConfidenceValue.textContent = `${settings.confidence}%`;
    persistSettings();
  });

  els.resultsList.addEventListener("click", (event) => {
    const refStrip = event.target.closest(".ref-strip.ready");
    if (refStrip) {
      const reference = state.refs.get(refStrip.dataset.refKey);
      if (reference) openReferenceModal(reference, refStrip.dataset.refLabel);
      return;
    }
    const card = event.target.closest(".detection-card");
    if (card) selectDetection(card.dataset.detectionId);
  });
  els.resultsList.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const refStrip = event.target.closest(".ref-strip.ready");
    if (refStrip) {
      event.preventDefault();
      const reference = state.refs.get(refStrip.dataset.refKey);
      if (reference) openReferenceModal(reference, refStrip.dataset.refLabel);
      return;
    }
    const card = event.target.closest(".detection-card");
    if (card) {
      event.preventDefault();
      selectDetection(card.dataset.detectionId);
    }
  });

  els.recentList.addEventListener("click", (event) => {
    const item = event.target.closest("[data-recent-id]");
    if (item) openHistoryItem(item.dataset.recentId);
  });
  els.historyList.addEventListener("click", (event) => {
    const item = event.target.closest("[data-history-id]");
    if (item) openHistoryItem(item.dataset.historyId);
  });

  els.historyFilter.addEventListener("change", () => loadHistory(true));
  els.historyRefresh.addEventListener("click", () => loadHistory(true));
  els.historyClear.addEventListener("click", clearHistory);
  els.historyLoadMore.addEventListener("click", () => loadHistory(false));
  els.settingsClearHistory.addEventListener("click", clearHistory);

  els.settingsForm.addEventListener("submit", (event) => {
    event.preventDefault();
    persistSettings();
    applySettings(false);
    toast("Preferences saved locally.");
  });

  els.setConfidence.addEventListener("input", () => {
    settings.confidence = clamp(Number(els.setConfidence.value) || 35, 15, 90);
    els.setConfidenceValue.textContent = `${settings.confidence}%`;
    els.scanConfidence.value = String(settings.confidence);
    els.scanConfidenceValue.textContent = `${settings.confidence}%`;
    persistSettings();
  });
  els.setInterval.addEventListener("change", () => {
    settings.interval = clamp(Number(els.setInterval.value) || 1800, 1200, 6000);
    persistSettings();
    applySettings(false);
  });
  els.setAutoSave.addEventListener("change", () => {
    settings.autoSave = els.setAutoSave.checked;
    persistSettings();
  });
  els.setMirror.addEventListener("change", () => {
    settings.mirror = els.setMirror.checked;
    persistSettings();
    applySettings(false);
  });
  els.setBoxes.addEventListener("change", () => {
    settings.boxes = els.setBoxes.checked;
    persistSettings();
    applySettings(false);
  });
  els.setDescriptions.addEventListener("change", () => {
    settings.descriptions = els.setDescriptions.checked;
    persistSettings();
    applySettings(true);
  });
  els.resetSettings.addEventListener("click", () => {
    settings = { ...DEFAULT_SETTINGS };
    persistSettings();
    syncSettingsUI();
    applySettings(true);
    toast("Preferences reset to defaults.");
  });

  els.camera.addEventListener("loadedmetadata", updateOverlaySize);
  els.uploadImage.addEventListener("load", updateOverlaySize);
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(updateOverlaySize, 120);
  });
  if (window.ResizeObserver) {
    new ResizeObserver(updateOverlaySize).observe(els.stage);
  }
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      clearTimeout(state.loopTimer);
    } else if (state.cameraOn) {
      startLiveLoop(300);
    }
  });
  window.addEventListener("beforeunload", () => {
    if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  });
}

function init() {
  initTheme();
  syncSettingsUI();
  updateModeCopy();
  applySettings(false);
  bindEvents();
  showView("dashboard");
  refreshHealth();
  updateOverlaySize();
}

init();
