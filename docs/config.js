// URL of the hosted Python backend (Hugging Face Space, Render, etc.).
// Example: window.API_BASE = "https://your-name-aiobject.hf.space";
window.API_BASE = "https://vocal-nicole-heading-cadillac.trycloudflare.com";

// Friendly notice while the backend URL is not set (scanning needs the Python server).
if (!window.API_BASE && /github\.io$/.test(location.hostname)) {
  document.addEventListener("DOMContentLoaded", () => {
    const bar = document.createElement("div");
    bar.textContent = "Backend not connected yet: the page works, but scanning needs the Python server. Set API_BASE in docs/config.js.";
    bar.style.cssText = "position:fixed;top:0;left:0;right:0;z-index:99999;padding:8px 12px;background:#ffcf5a;color:#201800;font:600 13px/1.3 system-ui,sans-serif;text-align:center";
    document.body.appendChild(bar);
  });
}
