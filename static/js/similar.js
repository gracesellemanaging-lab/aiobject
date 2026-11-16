"use strict";

const SIMILAR_MAX_BYTES = 20 * 1024 * 1024;

const similarEls = {
  drop: document.getElementById("similar-drop"),
  file: document.getElementById("similar-file"),
  preview: document.getElementById("similar-preview"),
  dropCopy: document.getElementById("similar-drop-copy"),
  mode: document.getElementById("similar-mode"),
  query: document.getElementById("similar-query"),
  search: document.getElementById("similar-search"),
  reset: document.getElementById("similar-reset"),
  progress: document.getElementById("similar-progress"),
  progressText: document.getElementById("similar-progress-text"),
  error: document.getElementById("similar-error"),
  errorText: document.getElementById("similar-error-text"),
  descriptionPanel: document.getElementById("similar-description-panel"),
  description: document.getElementById("similar-description"),
  tags: document.getElementById("similar-tags"),
  queries: document.getElementById("similar-queries"),
  analysisBadge: document.getElementById("similar-analysis-badge"),
  matches: document.getElementById("similar-matches"),
  count: document.getElementById("similar-count"),
  status: document.getElementById("similar-status"),
  foot: document.getElementById("similar-foot"),
};

const similarState = {
  file: null,
  previewUrl: null,
  controller: null,
  busy: false,
  initialized: false,
  lastResult: null,
};

function initSimilarView() {
  if (similarState.initialized || !similarEls.drop) return;
  similarState.initialized = true;

  similarEls.drop.addEventListener("click", () => similarEls.file.click());
  similarEls.drop.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      similarEls.file.click();
    }
  });
  similarEls.file.addEventListener("change", () => {
    const file = similarEls.file.files && similarEls.file.files[0];
    similarEls.file.value = "";
    if (file) setSimilarFile(file);
  });

  ["dragenter", "dragover"].forEach((name) => {
    similarEls.drop.addEventListener(name, (event) => {
      event.preventDefault();
      similarEls.drop.classList.add("dragging");
    });
  });
  ["dragleave", "dragend"].forEach((name) => {
    similarEls.drop.addEventListener(name, () => similarEls.drop.classList.remove("dragging"));
  });
  similarEls.drop.addEventListener("drop", (event) => {
    event.preventDefault();
    similarEls.drop.classList.remove("dragging");
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) setSimilarFile(file);
  });

  similarEls.search.addEventListener("click", runSimilaritySearch);
  similarEls.reset.addEventListener("click", resetSimilarView);
  similarEls.query.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      runSimilaritySearch();
    }
  });

  window.activateSimilarView = () => {
    if (!similarState.file) similarEls.drop.focus();
  };
}

function setSimilarFile(file) {
  if (!file) return;
  if (!file.type || !file.type.startsWith("image/")) {
    showSimilarError("Please choose a JPG, PNG or WEBP image.");
    return;
  }
  if (file.size > SIMILAR_MAX_BYTES) {
    showSimilarError("That image is larger than the 20 MB limit.");
    return;
  }
  similarState.file = file;
  if (similarState.previewUrl) URL.revokeObjectURL(similarState.previewUrl);
  similarState.previewUrl = URL.createObjectURL(file);
  similarEls.preview.src = similarState.previewUrl;
  similarEls.preview.hidden = false;
  similarEls.dropCopy.hidden = true;
  similarEls.drop.classList.add("has-image");
  hideSimilarError();
  similarEls.status.textContent = "Ready";
  similarEls.status.classList.remove("active");
  resetSimilarResults();
  toast("Image ready. Run the visual similarity search.", "info", 2400);
}

function resetSimilarResults() {
  similarState.lastResult = null;
  similarEls.descriptionPanel.hidden = true;
  similarEls.description.textContent = "";
  similarEls.tags.innerHTML = "";
  similarEls.queries.textContent = "";
  similarEls.count.textContent = "0";
  similarEls.foot.hidden = true;
  similarEls.foot.textContent = "";
  similarEls.progress.hidden = true;
  similarEls.status.textContent = similarState.file ? "Ready" : "Waiting";
  similarEls.status.classList.remove("active");
  similarEls.matches.innerHTML = `
    <div class="results-empty">
      <span class="radar"><i></i><i></i><i></i></span>
      <strong>No matches yet</strong>
      <p>Upload a picture and run the search. Results include similarity scores, the matching image and a link to its original licensed source.</p>
    </div>`;
}

function resetSimilarView() {
  if (similarState.controller) {
    try {
      similarState.controller.abort();
    } catch {
      /* ignore */
    }
    similarState.controller = null;
  }
  if (similarState.previewUrl) URL.revokeObjectURL(similarState.previewUrl);
  similarState.previewUrl = null;
  similarState.file = null;
  similarEls.preview.removeAttribute("src");
  similarEls.preview.hidden = true;
  similarEls.dropCopy.hidden = false;
  similarEls.drop.classList.remove("has-image", "dragging");
  similarEls.search.disabled = false;
  similarEls.query.value = "";
  hideSimilarError();
  resetSimilarResults();
}

function showSimilarError(message) {
  similarEls.error.hidden = false;
  similarEls.errorText.textContent = message;
  similarEls.status.textContent = "Error";
  similarEls.status.classList.remove("active");
}

function hideSimilarError() {
  similarEls.error.hidden = true;
  similarEls.errorText.textContent = "";
}

function setSimilarBusy(busy, message) {
  similarState.busy = busy;
  similarEls.search.disabled = busy;
  similarEls.reset.disabled = busy;
  similarEls.progress.hidden = !busy;
  similarEls.status.classList.toggle("active", busy);
  if (busy) {
    similarEls.status.textContent = "Analysing";
    similarEls.progressText.textContent = message || "Running local AI models and comparing licensed images…";
  }
}

async function runSimilaritySearch() {
  if (similarState.busy) return;
  if (!similarState.file) {
    showSimilarError("Choose or drop an image first.");
    similarEls.drop.focus();
    return;
  }
  hideSimilarError();
  resetSimilarResults();
  setSimilarBusy(true, "Detecting people, expressions and objects…");

  const controller = new AbortController();
  similarState.controller = controller;
  const timeout = window.setTimeout(() => controller.abort(), 240000);
  const form = new FormData();
  form.append("image", similarState.file, similarState.file.name || "similarity.jpg");
  form.append("mode", similarEls.mode.value || "smart");
  form.append("query", similarEls.query.value.trim());
  form.append("confidence", String(settings.confidence / 100));
  form.append("limit", "9");

  window.setTimeout(() => {
    if (similarState.busy) {
      similarEls.progressText.textContent = "Comparing image embeddings with licensed Wikimedia images…";
    }
  }, 2500);

  try {
    const response = await fetch("/api/similar", {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      if (data.description) renderSimilarAnalysis(data);
      throw new Error(data.error || `Similarity search failed (${response.status}).`);
    }
    similarState.lastResult = data;
    renderSimilarAnalysis(data);
    renderSimilarMatches(data);
    similarEls.status.textContent = "Complete";
    similarEls.status.classList.remove("active");
  } catch (error) {
    if (error && error.name === "AbortError") {
      showSimilarError("The search was cancelled or timed out. On a slower CPU, try again with a smaller image.");
    } else {
      showSimilarError(networkErrorText(error));
    }
  } finally {
    clearTimeout(timeout);
    if (similarState.controller === controller) similarState.controller = null;
    setSimilarBusy(false);
    similarEls.progress.hidden = true;
    similarEls.reset.disabled = false;
  }
}

function renderSimilarAnalysis(data) {
  const description = data.description || "No AI description was returned for this image.";
  similarEls.descriptionPanel.hidden = false;
  similarEls.description.textContent = description;

  const detections = Array.isArray(data.detections) ? data.detections : [];
  const seen = new Set();
  const chips = [];
  detections.forEach((detection) => {
    const label = String(detection.label || "").trim();
    if (!label) return;
    const key = label.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    const confidence = Math.round((Number(detection.confidence) || 0) * 100);
    chips.push(`<span class="attr-chip">${esc(label)} <strong>${confidence}%</strong></span>`);
  });
  similarEls.tags.innerHTML = chips.length
    ? chips.join("")
    : '<span class="attr-chip">No supported detection</span>';

  const meta = data.analysis_meta || {};
  const counts = [
    meta.people ? `${meta.people} people` : "",
    meta.objects ? `${meta.objects} objects` : "",
    meta.fruits ? `${meta.fruits} fruit` : "",
    meta.expressions ? `${meta.expressions} expressions` : "",
  ].filter(Boolean);
  similarEls.analysisBadge.textContent = counts.length ? counts.join(" · ") : (meta.engine || "Local AI");

  const queries = Array.isArray(data.queries) ? data.queries : [];
  similarEls.queries.textContent = queries.length
    ? `Visual search queries: ${queries.join(" · ")}`
    : "";
}

function renderSimilarMatches(data) {
  const matches = Array.isArray(data.matches) ? data.matches : [];
  similarEls.count.textContent = String(matches.length);
  similarEls.status.textContent = matches.length ? "Complete" : "No matches";
  similarEls.status.classList.remove("active");

  if (!matches.length) {
    similarEls.matches.innerHTML = `
      <div class="results-empty">
        <span class="radar"><i></i><i></i><i></i></span>
        <strong>No licensed matches found</strong>
        <p>Try a more common subject, add a visual keyword, or use a brighter photo.</p>
      </div>`;
    similarEls.foot.hidden = true;
    return;
  }

  similarEls.matches.innerHTML = matches.map((match) => {
    const percent = Math.round(Number(match.similarity_percent) || 0);
    const title = match.title || "Untitled reference";
    const license = match.license || "See source";
    const artist = match.artist ? ` · ${match.artist}` : "";
    const link = match.page_url || match.preview_url;
    const description = match.description
      ? `<span class="similar-desc">${esc(match.description)}</span>`
      : "";
    return `
      <a class="similar-card" href="${esc(link)}" target="_blank" rel="noopener noreferrer" title="Open the original licensed image">
        <span class="similar-thumb">
          <img loading="lazy" src="${esc(match.preview_url)}" alt="${esc(title)}">
          <span class="similar-percent">${percent}% similar</span>
        </span>
        <span class="similar-body">
          <strong>${esc(title)}</strong>
          <small>${esc(match.source || "Wikimedia Commons")} · ${esc(license)}${esc(artist)}</small>
          ${description}
          <span class="similar-source">View original source ↗</span>
        </span>
      </a>`;
  }).join("");
  bindImageFallbacks(similarEls.matches);

  const meta = data.meta || {};
  const details = [];
  if (meta.method) details.push(meta.method);
  if (meta.candidate_count) details.push(`${meta.candidate_count} licensed candidates compared`);
  if (meta.failed_count) details.push(`${meta.failed_count} unavailable`);
  if (meta.total_ms) details.push(`${Math.round(Number(meta.total_ms))} ms total`);
  if (meta.source_note) details.push(meta.source_note);
  similarEls.foot.hidden = !details.length;
  similarEls.foot.textContent = details.join(" · ");
}

initSimilarView();
