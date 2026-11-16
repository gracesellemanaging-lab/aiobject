"""Visual similarity search using local CNN embeddings.

The uploaded image is embedded with the MobileNetV3 network that is already
loaded for food recognition.  Candidate images are fetched from Wikimedia
Commons (freely licensed, publicly accessible media), embedded with the same
network and ranked by a combination of deep-feature cosine similarity and colour
composition.  Only visual similarity is measured; the service never tries to
identify the person in the uploaded image.
"""
from __future__ import annotations

import hashlib
import io
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Iterable

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps

from .config import BROWSER_USER_AGENT, COMMONS_API, SIMILAR_DIR
from .registry import registry

logger = logging.getLogger(__name__)

MAX_QUERIES = 4
RESULTS_PER_QUERY = 6
MAX_CANDIDATES = 20
MAX_DOWNLOADS = 16
SEARCH_TIMEOUT = 18.0
MAX_SEARCH_SECONDS = 100.0

_HTML_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"\s+")


class SimilarityError(RuntimeError):
    """Raised when no licensed candidate images could be retrieved."""


def _strip_html(value: str | None, limit: int = 240) -> str:
    if not value:
        return ""
    text = _SPACES.sub(" ", _HTML_TAG.sub(" ", str(value))).strip()
    return text[:limit]


def _l2(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


class VisualSimilarityService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._embedding_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": BROWSER_USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "application/json, image/*",
        })
        self._last_api_call = 0.0

    # ------------------------------------------------------------- embeddings
    def embed(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (deep embedding, colour signature) for a BGR image."""
        import torch

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        model, _, transform = registry.classifier()
        tensor = transform(pil).unsqueeze(0)
        with registry.classifier_lock:
            with torch.inference_mode():
                features = model.features(tensor)
                features = model.avgpool(features)
                features = torch.flatten(features, 1)
        vector = features[0].detach().cpu().numpy().astype(np.float32)
        return _l2(vector), self.color_signature(image)

    @staticmethod
    def color_signature(image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(histogram, histogram, alpha=1.0, norm_type=cv2.NORM_L1)
        grid = []
        cell = resized.shape[0] // 3
        for row in range(3):
            for column in range(3):
                block = resized[row * cell:(row + 1) * cell, column * cell:(column + 1) * cell]
                if block.size == 0:
                    grid.extend([0.0, 0.0, 0.0])
                    continue
                mean = block.reshape(-1, 3).mean(axis=0)
                grid.extend((mean / 255.0).tolist())
        signature = np.concatenate([histogram.reshape(-1), np.asarray(grid, dtype=np.float32)])
        return _l2(signature)

    @staticmethod
    def _score(query_embedding: np.ndarray, query_colour: np.ndarray,
               embedding: np.ndarray, colour: np.ndarray) -> dict[str, float]:
        embedding_similarity = float(np.clip(np.dot(query_embedding, embedding), -1.0, 1.0))
        colour_similarity = float(np.clip(np.dot(query_colour, colour), -1.0, 1.0))
        embedding_normalized = max(0.0, (embedding_similarity + 1.0) / 2.0)
        colour_normalized = max(0.0, (colour_similarity + 1.0) / 2.0)
        combined = 0.68 * embedding_normalized + 0.32 * colour_normalized
        return {
            "similarity": round(min(0.995, max(0.01, combined)), 4),
            "similarity_percent": round(min(99.5, max(1.0, combined * 100)), 1),
            "embedding_similarity": round(embedding_similarity, 4),
            "colour_similarity": round(colour_similarity, 4),
        }

    # ----------------------------------------------------------------- search
    def search(self, image: np.ndarray, queries: Iterable[str], limit: int = 8) -> dict[str, Any]:
        started = time.perf_counter()
        query_list = self._normalise_queries(queries)
        if not query_list:
            query_list = self._fallback_queries(image)
        if not query_list:
            raise SimilarityError("No visual context could be derived from the image.")

        query_sources: list[str] = list(query_list)
        query_embedding, query_colour = self.embed(image)

        candidates = self._collect_candidates(query_list)
        if not candidates:
            raise SimilarityError(
                "No licensed candidate images were returned by Wikimedia Commons. "
                "Try a more descriptive photo or check the internet connection."
            )

        matches: list[dict[str, Any]] = []
        failures = 0
        with ThreadPoolExecutor(max_workers=6, thread_name_prefix="similar") as pool:
            futures = {pool.submit(self._prepare_candidate, candidate): candidate for candidate in candidates}
            done, pending = wait(futures, timeout=MAX_SEARCH_SECONDS)
            failures += len(pending)
            for future in pending:
                future.cancel()
            for future in done:
                try:
                    prepared = future.result()
                except Exception:  # noqa: BLE001 - one bad candidate must not fail the search
                    failures += 1
                    continue
                if not prepared:
                    failures += 1
                    continue
                prepared.update(self._score(
                    query_embedding, query_colour, prepared["embedding"], prepared["colour"]
                ))
                prepared.pop("embedding", None)
                prepared.pop("colour", None)
                matches.append(prepared)

        matches.sort(key=lambda item: item["similarity"], reverse=True)
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        if not matches:
            raise SimilarityError(
                "Candidate images were found, but none could be downloaded for comparison. "
                "Please try again in a moment."
            )
        return {
            "matches": matches[: max(1, min(12, limit))],
            "queries": query_sources,
            "method": "MobileNetV3 image embeddings (68%) + HSV colour signature (32%)",
            "candidate_count": len(candidates),
            "compared_count": len(matches),
            "failed_count": failures,
            "elapsed_ms": elapsed,
            "source_note": (
                "Candidates come from Wikimedia Commons and are used as publicly licensed "
                "visual references. Similarity is visual only and does not identify anyone."
            ),
        }

    def _normalise_queries(self, queries: Iterable[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for query in queries:
            text = _SPACES.sub(" ", str(query or "")).strip(" .,-")
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            output.append(text[:80])
            if len(output) >= MAX_QUERIES:
                break
        return output

    def _fallback_queries(self, image: np.ndarray) -> list[str]:
        try:
            from .classifier import _top_predictions

            predictions = _top_predictions(image, top_k=3)
            return [prediction["name"].lower().replace("_", " ") for prediction in predictions[:2]]
        except Exception:  # noqa: BLE001 - optional fallback
            logger.info("Classifier fallback queries unavailable", exc_info=True)
            return []

    # --------------------------------------------------------------- Wikimedia
    def _collect_candidates(self, queries: list[str]) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for query in queries:
            for candidate in self._commons_search(query):
                key = candidate.get("page_url") or candidate.get("url") or candidate.get("title")
                if not key or key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
                if len(candidates) >= MAX_CANDIDATES:
                    return candidates
        return candidates

    def _throttle(self) -> None:
        wait = 0.18 - (time.time() - self._last_api_call)
        if wait > 0:
            time.sleep(wait)
        self._last_api_call = time.time()

    def _commons_search(self, query: str) -> list[dict[str, Any]]:
        self._throttle()
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrnamespace": "6",
            "gsrsearch": f"{query} filetype:bitmap",
            "gsrlimit": str(RESULTS_PER_QUERY),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size",
            "iiurlwidth": "720",
        }
        try:
            response = self._session.get(COMMONS_API, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.info("Commons search failed for %s: %s", query, exc)
            return []

        pages = (payload.get("query") or {}).get("pages") or []
        if isinstance(pages, dict):
            pages = list(pages.values())
        pages.sort(key=lambda page: page.get("index", 99))
        results: list[dict[str, Any]] = []
        for page in pages:
            infos = page.get("imageinfo") or []
            if not infos:
                continue
            info = infos[0]
            url = info.get("thumburl") or info.get("url")
            width = int(info.get("width") or 0)
            height = int(info.get("height") or 0)
            if not url:
                continue
            if width and height and min(width, height) < 160:
                continue
            metadata = info.get("extmetadata") or {}
            license_name = self._meta_value(metadata, "LicenseShortName")
            artist = self._strip_credit(self._meta_value(metadata, "Artist"))
            description = _strip_html(self._meta_value(metadata, "ImageDescription"), limit=260)
            results.append({
                "title": re.sub(r"^File:", "", page.get("title") or "").rsplit(".", 1)[0] or query,
                "url": url,
                "page_url": info.get("descriptionurl"),
                "source": "Wikimedia Commons",
                "license": license_name or "See file page",
                "artist": artist,
                "description": description,
                "search_query": query,
            })
        return results

    @staticmethod
    def _meta_value(metadata: dict[str, Any], key: str) -> str | None:
        entry = metadata.get(key)
        if isinstance(entry, dict):
            value = entry.get("value")
            return str(value) if value is not None else None
        return None

    @staticmethod
    def _strip_credit(value: str | None) -> str:
        text = _strip_html(value, limit=120)
        return text or "Wikimedia contributor"

    # ---------------------------------------------------------------- download
    def _prepare_candidate(self, candidate: dict[str, Any]) -> dict[str, Any] | None:
        url = candidate.get("url")
        if not url:
            return None
        cache_key = hashlib.sha1(url.encode("utf-8")).hexdigest()[:20]
        local_name = f"{cache_key}.jpg"
        target = SIMILAR_DIR / local_name
        embedding = None
        colour = None
        with self._lock:
            cached = self._embedding_cache.get(url)
        if cached and target.is_file():
            embedding, colour = cached
        if not target.is_file():
            try:
                self._download(url, target)
            except Exception as exc:  # noqa: BLE001
                logger.info("Similar candidate download failed: %s", exc)
                return None
        if embedding is None or colour is None:
            try:
                image = self._load_local(target)
                embedding, colour = self.embed(image)
            except Exception as exc:  # noqa: BLE001
                logger.info("Similar candidate embedding failed: %s", exc)
                return None
            with self._lock:
                if len(self._embedding_cache) > 400:
                    self._embedding_cache.clear()
                self._embedding_cache[url] = (embedding, colour)
        return {
            "title": candidate.get("title"),
            "source": candidate.get("source") or "Wikimedia Commons",
            "license": candidate.get("license"),
            "artist": candidate.get("artist"),
            "page_url": candidate.get("page_url"),
            "preview_url": f"/static/similar/{local_name}",
            "description": candidate.get("description"),
            "search_query": candidate.get("search_query"),
            "embedding": embedding,
            "colour": colour,
        }

    def _download(self, url: str, target) -> None:
        last_error: Exception | None = None
        content = b""
        for attempt in range(2):
            try:
                response = self._session.get(
                    url,
                    timeout=(5, 20),
                    headers={"Referer": "https://commons.wikimedia.org/"},
                )
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                if content_type and not content_type.startswith("image/"):
                    raise ValueError(f"unexpected content type: {content_type}")
                content = response.content
                break
            except Exception as exc:  # noqa: BLE001 - retry once before giving up
                last_error = exc
                if attempt == 0:
                    time.sleep(0.6)
        if not content:
            raise last_error or ValueError("empty candidate download")
        if len(content) > 8 * 1024 * 1024:
            raise ValueError("candidate image is too large")
        image = Image.open(io.BytesIO(content))
        image = ImageOps.exif_transpose(image)
        if getattr(image, "is_animated", False):
            image.seek(0)
        image = image.convert("RGB")
        if min(image.size) < 120:
            raise ValueError("candidate image is too small")
        image.thumbnail((720, 720), Image.Resampling.LANCZOS)
        temp = target.with_suffix(".part.jpg")
        image.save(temp, format="JPEG", quality=82, optimize=True)
        temp.replace(target)

    @staticmethod
    def _load_local(path) -> np.ndarray:
        with Image.open(path) as pil:
            rgb = np.asarray(pil.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


similarity_service = VisualSimilarityService()
