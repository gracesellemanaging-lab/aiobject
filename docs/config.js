// Browser-only mode: detection runs inside the page (TensorFlow.js COCO-SSD).
// No Python server, tunnel or Hugging Face Space is needed.
// To use a Python backend instead, set BROWSER_AI = false and API_BASE = "https://your-backend-url".
window.BROWSER_AI = true;
window.API_BASE = "";

// Friendly notice when neither browser mode nor a backend is configured.
if (!window.BROWSER_AI && !window.API_BASE && /github\.io$/.test(location.hostname)) {
  document.addEventListener("DOMContentLoaded", () => {
    const bar = document.createElement("div");
    bar.textContent = "Backend not connected yet: the page works, but scanning needs the Python server. Set API_BASE in docs/config.js.";
    bar.style.cssText = "position:fixed;top:0;left:0;right:0;z-index:99999;padding:8px 12px;background:#ffcf5a;color:#201800;font:600 13px/1.3 system-ui,sans-serif;text-align:center";
    document.body.appendChild(bar);
  });
}
