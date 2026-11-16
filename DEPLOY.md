# Deploying AI Object Visualizer

The app has two parts:

1. **Front-end** (`docs/`): static files published by GitHub Pages at
   https://gracesellemanaging-lab.github.io/aiobject/
2. **Backend** (Flask + YOLO/torch/mediapipe): needs a Python host. GitHub Pages cannot run it.

## Backend on Hugging Face Spaces (free)

1. Create a new Space at https://huggingface.co/new-space, SDK **Docker**.
2. Upload this repo's files to the Space (or push this repo to the Space's git remote).
   The Space's `README.md` needs this header:
   ```
   ---
   sdk: docker
   app_port: 7860
   ---
   ```
3. Wait for the build. Your backend URL is `https://<user>-<space>.hf.space`.

## Connect the front-end

Edit `docs/config.js`:

```js
window.API_BASE = "https://<user>-<space>.hf.space";
```

Commit and push. Pages updates in about a minute. The backend only accepts requests from
`https://gracesellemanaging-lab.github.io` by default; set the `ALLOWED_ORIGINS` env var
(comma-separated) to allow others.

## Backend on GitHub Codespaces (no other account needed)

1. On the repo page: **Code** → **Codespaces** → **Create codespace on main**.
2. Wait for setup (installs dependencies, then starts the server on port 5000).
3. Open the **Ports** tab, right-click port **5000** → **Port Visibility** → **Public**.
4. Copy the forwarded address (`https://<name>-5000.app.github.dev`) and set it in `docs/config.js`:
   ```js
   window.API_BASE = "https://<name>-5000.app.github.dev";
   ```
The Codespace stops after about 30 minutes idle; reopen it and the server starts again.
