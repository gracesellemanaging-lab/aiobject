// URL of the hosted Python backend (Hugging Face Space, Render, etc.).
// Fastest way to try a new tunnel without editing this file:
//   https://your-name.github.io/aiobject/?api=https://your-backend.example.com
// You can also hard-code it here: window.API_BASE = "https://your-name.hf.space";
const apiFromQuery = new URLSearchParams(window.location.search).get("api");
window.API_BASE = (apiFromQuery || "https://symmetrical-funicular-7v4xwqr7q4452rx7g.github.dev").trim().replace(/\/$/, "");

// Friendly notice while the backend URL is not set (scanning needs the Python server).
if (!window.API_BASE && /github\.io$/.test(location.hostname)) {
  document.addEventListener("DOMContentLoaded", () => {
    const bar = document.createElement("div");
    bar.textContent = "Backend not connected. Add ?api=https://your-backend-url to this page URL, or set API_BASE in docs/config.js.";
    bar.style.cssText = "position:fixed;top:0;left:0;right:0;z-index:99999;padding:8px 12px;background:#ffcf5a;color:#201800;font:600 13px/1.3 system-ui,sans-serif;text-align:center";
    document.body.appendChild(bar);
  });
}
