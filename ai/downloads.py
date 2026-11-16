"""Download the open-source models used by the visualizer (first run only)."""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import requests

from .config import BROWSER_USER_AGENT, MODELS_DIR, MODEL_URLS

logger = logging.getLogger(__name__)


def missing_models() -> list[str]:
    return [name for name in MODEL_URLS if not (MODELS_DIR / name).is_file()]


def ensure_models(progress=None) -> dict[str, str]:
    """Download any missing model files. Returns a filename -> state map."""
    results: dict[str, str] = {}
    for name, url in MODEL_URLS.items():
        target = MODELS_DIR / name
        if target.is_file() and target.stat().st_size > 1024:
            results[name] = "ready"
            continue
        if progress:
            progress(f"Downloading {name}")
        try:
            _download(url, target)
            results[name] = "downloaded"
            logger.info("Downloaded %s", name)
        except Exception as exc:  # noqa: BLE001 - surfaced in health endpoint
            logger.exception("Could not download %s", name)
            results[name] = f"error: {exc}"
    return results


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": BROWSER_USER_AGENT}
    with requests.get(url, headers=headers, timeout=60, stream=True, allow_redirects=True) as response:
        response.raise_for_status()
        expected = int(response.headers.get("content-length") or 0)
        fd, temp_name = tempfile.mkstemp(prefix=target.name, suffix=".part", dir=target.parent)
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            received = 0
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    received += len(chunk)
            if expected and received < expected * 0.98:
                raise IOError(f"incomplete download ({received}/{expected} bytes)")
            os.replace(temp_path, target)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
