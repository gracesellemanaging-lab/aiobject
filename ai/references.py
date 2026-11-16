"""Online visual reference retrieval through Wikimedia / Wikipedia.

Images are only used as visual references.  Every image is cached locally so a
detection does not trigger repeated network requests.
"""
from __future__ import annotations

import io
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Iterable

import requests
from PIL import Image, ImageOps

from .config import (
    BROWSER_USER_AGENT,
    COMMONS_API,
    DATA_DIR,
    REFERENCE_DIR,
    REFERENCE_TIMEOUT,
    WIKI_API,
)
from .knowledge import reference_info

logger = logging.getLogger(__name__)

CACHE_PATH = DATA_DIR / "references.json"
PLACEHOLDER = "/static/img/reference-placeholder.svg"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:64] or "reference"


class ReferenceService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._cache: dict[str, dict] = {}
        self._priming: set[str] = set()
        self._net_lock = threading.Lock()
        self._last_net = 0.0
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": BROWSER_USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
        })
        self._load_cache()

    # ------------------------------------------------------------- persistence
    def _load_cache(self) -> None:
        if not CACHE_PATH.is_file():
            return
        try:
            payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                self._cache = payload
        except Exception:  # noqa: BLE001
            logger.warning("Reference cache could not be read", exc_info=True)

    def _save_cache(self) -> None:
        try:
            temp = CACHE_PATH.with_suffix(".tmp")
            temp.write_text(json.dumps(self._cache, ensure_ascii=False, indent=1), encoding="utf-8")
            temp.replace(CACHE_PATH)
        except Exception:  # noqa: BLE001
            logger.warning("Reference cache could not be saved", exc_info=True)

    # ---------------------------------------------------------------- public API
    def cached(self, key: str) -> dict | None:
        key = slugify(key)
        with self._lock:
            entry = self._cache.get(key)
        if not entry:
            return None
        if entry.get("status") == "ready":
            file_path = REFERENCE_DIR / entry.get("filename", "")
            if file_path.is_file():
                return self._public(entry)
        return None

    def get(self, key: str, label: str = "", query: str = "") -> dict:
        key = slugify(key or label or "reference")
        cached = self.cached(key)
        if cached:
            return cached

        key_lock = self._key_lock(key)
        with key_lock:
            cached = self.cached(key)
            if cached:
                return cached
            with self._lock:
                entry = self._cache.get(key)
            if entry and entry.get("status") == "unavailable" and time.time() - entry.get("fetched_at", 0) < 600:
                return {**self._public(entry), "image": PLACEHOLDER, "status": "unavailable"}

            info = reference_info(key, label, query)
            entry = self._resolve(key, info["title"], info["query"])
            with self._lock:
                old_file = self._cache.get(key, {}).get("filename")
                self._cache[key] = entry
                self._save_cache()
            if old_file and old_file != entry.get("filename"):
                try:
                    (REFERENCE_DIR / old_file).unlink(missing_ok=True)
                except OSError:
                    pass

            if entry.get("status") == "ready":
                return self._public(entry)
            return {**self._public(entry), "image": PLACEHOLDER, "status": "unavailable"}

    def prime(self, items: Iterable[tuple[str, str]]) -> None:
        """Warm the cache in the background; items are (reference_key, label)."""
        for key, label in items:
            key = slugify(key)
            if self.cached(key):
                continue
            with self._lock:
                if key in self._priming:
                    continue
                self._priming.add(key)

            def work(k=key, lab=label):
                try:
                    self.get(k, lab)
                finally:
                    with self._lock:
                        self._priming.discard(k)

            threading.Thread(target=work, name=f"ref-{key}", daemon=True).start()

    # ----------------------------------------------------------------- internals
    def _key_lock(self, key: str) -> threading.Lock:
        with self._lock:
            if key not in self._key_locks:
                self._key_locks[key] = threading.Lock()
            return self._key_locks[key]

    def _public(self, entry: dict) -> dict:
        filename = entry.get("filename")
        return {
            "key": entry.get("key"),
            "title": entry.get("title") or "Visual reference",
            "image": f"/static/refs/{filename}" if filename else PLACEHOLDER,
            "page_url": entry.get("page_url"),
            "source": entry.get("source") or "Wikimedia",
            "status": entry.get("status"),
            "query": entry.get("query"),
        }

    def _throttle(self) -> None:
        with self._net_lock:
            wait = 0.3 - (time.time() - self._last_net)
            if wait > 0:
                time.sleep(wait)
            self._last_net = time.time()

    def _resolve(self, key: str, title: str, query: str) -> dict:
        base = {
            "key": key,
            "title": title,
            "query": query,
            "fetched_at": time.time(),
        }
        candidates: list[dict] = []

        page_image = self._wikipedia_title_image(title)
        if page_image:
            candidates.append(page_image)
        if not candidates and query:
            candidates.extend(self._wikipedia_search_images(query))
        if not candidates:
            candidates.extend(self._commons_images(query or title))

        for candidate in candidates:
            try:
                filename = self._download_image(key, candidate["url"])
                return {
                    **base,
                    "status": "ready",
                    "filename": filename,
                    "title": candidate.get("title") or title,
                    "page_url": candidate.get("page_url"),
                    "source": candidate.get("source") or "Wikimedia Commons",
                }
            except Exception as exc:  # noqa: BLE001
                logger.info("Reference download failed for %s: %s", key, exc)

        return {**base, "status": "unavailable"}

    def _get_json(self, api: str, params: dict) -> dict | None:
        self._throttle()
        try:
            response = self._session.get(api, params=params, timeout=REFERENCE_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            logger.info("Wikimedia API request failed: %s", exc)
            return None

    def _wikipedia_title_image(self, title: str) -> dict | None:
        data = self._get_json(WIKI_API, {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "pageimages|info",
            "piprop": "thumbnail",
            "pithumbsize": "900",
            "inprop": "url",
            "titles": title,
            "redirects": "1",
        })
        pages = (data or {}).get("query", {}).get("pages", [])
        for page in pages:
            thumb = (page.get("thumbnail") or {}).get("source")
            if thumb:
                return {
                    "url": thumb,
                    "title": page.get("title") or title,
                    "page_url": page.get("fullurl"),
                    "source": "Wikipedia",
                }
        return None

    def _wikipedia_search_images(self, query: str) -> list[dict]:
        data = self._get_json(WIKI_API, {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": "4",
            "prop": "pageimages|info",
            "piprop": "thumbnail",
            "pithumbsize": "900",
            "inprop": "url",
        })
        pages = (data or {}).get("query", {}).get("pages", [])
        pages.sort(key=lambda some: some.get("index", 99))
        results = []
        for page in pages:
            thumb = (page.get("thumbnail") or {}).get("source")
            if thumb:
                results.append({
                    "url": thumb,
                    "title": page.get("title") or query,
                    "page_url": page.get("fullurl"),
                    "source": "Wikipedia",
                })
        return results

    def _commons_images(self, query: str) -> list[dict]:
        data = self._get_json(COMMONS_API, {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrnamespace": "6",
            "gsrsearch": f"{query} filetype:bitmap",
            "gsrlimit": "5",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": "900",
        })
        pages = (data or {}).get("query", {}).get("pages", [])
        if isinstance(pages, dict):
            pages = list(pages.values())
        pages.sort(key=lambda some: some.get("index", 99))
        results = []
        for page in pages:
            infos = page.get("imageinfo") or []
            if not infos:
                continue
            info = infos[0]
            url = info.get("thumburl") or info.get("url")
            if not url:
                continue
            results.append({
                "url": url,
                "title": re.sub(r"^File:", "", page.get("title") or "").rsplit(".", 1)[0],
                "page_url": info.get("descriptionurl"),
                "source": "Wikimedia Commons",
            })
        return results

    def _download_image(self, key: str, url: str) -> str:
        self._throttle()
        response = self._session.get(
            url,
            timeout=(6, 25),
            headers={"Referer": "https://en.wikipedia.org/", "Accept": "image/avif,image/webp,image/*,*/*;q=0.8"},
        )
        response.raise_for_status()
        content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
        if not content_type.startswith("image/") and not url.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            raise ValueError(f"not an image ({content_type})")
        if len(response.content) > 12 * 1024 * 1024:
            raise ValueError("reference image is too large")
        image = Image.open(io.BytesIO(response.content))
        image = ImageOps.exif_transpose(image)
        if getattr(image, "is_animated", False):
            image.seek(0)
        image = image.convert("RGB")
        if min(image.size) < 150:
            raise ValueError("reference image is too small")
        image.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
        filename = f"{slugify(key)}.jpg"
        target: Path = REFERENCE_DIR / filename
        image.save(target, format="JPEG", quality=86, optimize=True)
        return filename


reference_service = ReferenceService()
