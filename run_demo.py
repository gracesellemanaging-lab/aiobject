#!/usr/bin/env python3
"""One command to bring the demo back online.

    python run_demo.py            # start backend + tunnel, update docs/config.js, commit and push
    python run_demo.py --no-push  # same, but only edit docs/config.js locally

It starts app.py, opens a Cloudflare quick tunnel, writes the new tunnel URL into
docs/config.js and pushes it so GitHub Pages points at the live backend (Pages needs ~1 minute).
Keep this window open: closing it stops the backend and the tunnel.

Needs: Python deps from requirements.txt, `cloudflared` on PATH, and git push access.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "docs" / "config.js"
PORT = int(os.environ.get("PORT", "5000"))
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
API_RE = re.compile(r'window\.API_BASE\s*=\s*"[^"]*";')


def set_api_base(text: str, url: str) -> str:
    """Return config.js text with window.API_BASE pointing at ``url``."""
    return API_RE.sub(lambda _m: f'window.API_BASE = "{url}";', text, count=1)


def wait_for(url: str, seconds: int) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "run_demo/1.0"})
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status == 200:
                    return True
        except Exception:  # noqa: BLE001 - keep polling
            pass
        time.sleep(3)
    return False


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-push", action="store_true", help="only edit docs/config.js, do not commit/push")
    args = parser.parse_args()

    if not shutil.which("cloudflared"):
        print("cloudflared is not installed. Install it first:\n"
              "  Windows: winget install Cloudflare.cloudflared\n"
              "  macOS:   brew install cloudflared\n"
              "  Linux:   https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/")
        return 1

    env = {**os.environ, "PORT": str(PORT), "HOST": "127.0.0.1"}
    print(f"[1/4] Starting backend on port {PORT} (first start downloads models, can take a few minutes)...")
    backend = subprocess.Popen([sys.executable, "app.py"], cwd=ROOT, env=env)
    tunnel = None
    try:
        if not wait_for(f"http://127.0.0.1:{PORT}/api/health", 300):
            print("Backend did not come up. Check the messages above.")
            return 1

        print("[2/4] Opening Cloudflare tunnel...")
        tunnel = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://localhost:{PORT}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        url = None
        deadline = time.time() + 90
        while time.time() < deadline and url is None:
            line = tunnel.stdout.readline()
            if not line:
                break
            match = URL_RE.search(line)
            if match:
                url = match.group(0)
        if not url:
            print("Could not read the tunnel URL from cloudflared.")
            return 1
        print(f"      Tunnel URL: {url}")

        print("[3/4] Waiting for the tunnel to answer...")
        if not wait_for(f"{url}/api/health", 90):
            print("Tunnel URL does not answer yet; writing it anyway.")

        CONFIG.write_text(set_api_base(CONFIG.read_text(encoding="utf-8"), url), encoding="utf-8")
        print(f"[4/4] Updated {CONFIG.relative_to(ROOT)}")
        if args.no_push:
            print("      --no-push: commit and push docs/config.js yourself.")
        else:
            git("add", "docs/config.js")
            commit = git("commit", "-m", f"Point demo at active tunnel {url}")
            push = git("push") if commit.returncode == 0 else commit
            print("      Pushed. GitHub Pages updates in about 1 minute." if push.returncode == 0
                  else f"      Git step failed:\n{push.stdout}{push.stderr}")

        print("\nDemo is live: https://gracesellemanaging-lab.github.io/aiobject/")
        print("Keep this window open. Press Ctrl+C to stop.\n")
        while backend.poll() is None and tunnel.poll() is None:
            tunnel.stdout.readline()  # drain cloudflared output so it never blocks
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        for proc in (tunnel, backend):
            if proc and proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    sys.exit(main())
