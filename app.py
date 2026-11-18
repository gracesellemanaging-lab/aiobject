"""Flask application for AI Object Visualizer."""
from __future__ import annotations

import logging
import os
import re
import time

from flask import Flask, jsonify, render_template, request, send_from_directory

import database
from ai.config import DEFAULT_CONFIDENCE, MAX_UPLOAD_BYTES, SNAPSHOT_DIR
from ai.references import PLACEHOLDER, reference_service
from ai.registry import registry
from ai.similarity import SimilarityError, similarity_service
from ai.utils import decode_image
from scanners import SCANNERS, SCAN_MODES
from scanners.fruit_scanner import UNSUPPORTED_FRUITS
from scanners.fruit_scanner import FRUIT_SCAN_SUPPORTED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("visualizer")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + 2 * 1024 * 1024
app.config["JSON_SORT_KEYS"] = False

SESSION_PATTERN = re.compile(r"[^a-zA-Z0-9_-]")
REFERENCE_KEY_PATTERN = re.compile(r"[^a-zA-Z0-9_-]")


ALLOWED_ORIGINS = {
    origin.strip()
    for origin in os.environ.get(
        "ALLOWED_ORIGINS", "https://gracesellemanaging-lab.github.io"
    ).split(",")
    if origin.strip()
}


@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        return app.make_default_options_response()
    return None


@app.after_request
def add_cors(response):
    origin = request.headers.get("Origin")
    if origin and ("*" in ALLOWED_ORIGINS or origin in ALLOWED_ORIGINS):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.after_request
def add_headers(response):
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    else:
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
    return response


@app.get("/")
def index():
    return render_template("index.html", default_confidence=int(DEFAULT_CONFIDENCE * 100))


@app.get("/api/health")
def health():
    status = registry.status()
    status.update({
        "app": "AI Object Visualizer",
        "version": "1.0.0",
        "modes": list(SCAN_MODES),
        "default_confidence": DEFAULT_CONFIDENCE,
        "unsupported_fruits": UNSUPPORTED_FRUITS,
        "supported_fruits": FRUIT_SCAN_SUPPORTED,
    })
    return jsonify(status)


@app.post("/api/scan")
def scan():
    started = time.perf_counter()
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "No image was sent. Use multipart field 'image'."}), 400
    raw = file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        return jsonify({"error": "Image is larger than the 20 MB limit."}), 413
    if not raw:
        return jsonify({"error": "Empty image data."}), 400

    mode = (request.form.get("mode") or "smart").lower().strip()
    if mode not in SCANNERS:
        return jsonify({"error": f"Unknown mode '{mode}'."}), 400

    try:
        confidence = float(request.form.get("confidence") or DEFAULT_CONFIDENCE)
    except ValueError:
        confidence = DEFAULT_CONFIDENCE
    confidence = max(0.15, min(0.9, confidence))

    source = (request.form.get("source") or "upload").lower().strip()
    still = (request.form.get("still") or "").lower() in {"1", "true", "yes", "on"}
    # A manual "Scan Now" photo gets the full still-image pipeline (higher resolution,
    # tiled YOLO-World), while the continuous camera loop stays on the fast path.
    live = source == "camera" and not still
    save = (request.form.get("save") or "true").lower() not in {"0", "false", "no"}
    raw_session = request.form.get("session_id") or request.headers.get("X-Session-Id") or ""
    session_id = SESSION_PATTERN.sub("", raw_session)[:80] or None

    try:
        image = decode_image(raw)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    scanner = SCANNERS[mode]
    try:
        if mode == "gesture":
            result = scanner.scan(image, confidence, live=live, session_id=session_id)
        else:
            result = scanner.scan(image, confidence, live=live)
    except Exception as exc:  # noqa: BLE001 - model failure becomes a clean API error
        logger.exception("Scan failed in %s mode", mode)
        return jsonify({
            "error": "The scan could not be completed.",
            "detail": str(exc),
            "hint": "Check /api/health to confirm the AI models are ready.",
        }), 503

    saved = 0
    if save and result.get("detections"):
        try:
            saved = database.log_scan(mode, result["detections"], image, result.get("notes"))
        except Exception:  # noqa: BLE001 - history is non-critical
            logger.exception("History write failed")
    result["saved"] = saved
    result["meta"]["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return jsonify(result)


@app.post("/api/similar")
def similar():
    """Analyse an image and rank licensed online images by visual similarity."""
    started = time.perf_counter()
    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "No image was sent. Use multipart field 'image'."}), 400
    raw = file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        return jsonify({"error": "Image is larger than the 20 MB limit."}), 413
    if not raw:
        return jsonify({"error": "Empty image data."}), 400

    try:
        image = decode_image(raw)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    focus = (request.form.get("mode") or "smart").lower().strip()
    scanner = SCANNERS.get(focus, SCANNERS["smart"])
    explicit_query = (request.form.get("query") or "").strip()
    try:
        confidence = float(request.form.get("confidence") or DEFAULT_CONFIDENCE)
    except ValueError:
        confidence = DEFAULT_CONFIDENCE
    confidence = max(0.15, min(0.9, confidence))

    try:
        limit = int(request.form.get("limit") or 8)
    except ValueError:
        limit = 8
    limit = max(1, min(12, limit))

    try:
        analysis = scanner.scan(image, confidence, live=False)
    except Exception as exc:  # noqa: BLE001 - model failure becomes a clean API error
        logger.exception("Similarity analysis failed")
        return jsonify({
            "error": "The image could not be analysed.",
            "detail": str(exc),
            "hint": "Check /api/health to confirm the AI models are ready.",
        }), 503

    queries: list[str] = []
    if explicit_query:
        queries.append(explicit_query)
    queries.extend(analysis.get("keywords") or [])
    if not queries:
        queries.extend(item["label"] for item in analysis.get("detections", [])[:3])

    try:
        matches = similarity_service.search(image, queries, limit=limit)
    except SimilarityError as exc:
        return jsonify({
            "error": str(exc),
            "description": analysis.get("description"),
            "detections": analysis.get("detections", []),
            "notes": analysis.get("notes", []),
            "queries": queries,
        }), 502
    except Exception as exc:  # noqa: BLE001
        logger.exception("Visual similarity search failed")
        return jsonify({
            "error": "The visual similarity search could not be completed.",
            "detail": str(exc),
        }), 503

    return jsonify({
        "focus": focus if focus in SCANNERS else "smart",
        "description": analysis.get("description"),
        "keywords": analysis.get("keywords") or [],
        "detections": analysis.get("detections", []),
        "notes": analysis.get("notes", []),
        "analysis_meta": analysis.get("meta", {}),
        "queries": matches["queries"],
        "matches": matches["matches"],
        "meta": {
            "method": matches["method"],
            "candidate_count": matches["candidate_count"],
            "compared_count": matches["compared_count"],
            "failed_count": matches["failed_count"],
            "source_note": matches["source_note"],
            "search_ms": matches["elapsed_ms"],
            "total_ms": round((time.perf_counter() - started) * 1000, 1),
            "server_time": time.time(),
        },
    })


@app.get("/api/reference")
def reference():
    key = (request.args.get("key") or "").strip()
    label = (request.args.get("label") or "").strip()
    query = (request.args.get("query") or "").strip()
    if not key and not label:
        return jsonify({"error": "A reference key or label is required."}), 400
    key = REFERENCE_KEY_PATTERN.sub("-", key.lower())[:80]
    cached = reference_service.cached(key) if key else None
    if cached:
        return jsonify(cached)

    reference_service.prime([(key, label or key)])
    return jsonify({
        "key": key,
        "title": label or key.replace("-", " ").title(),
        "image": PLACEHOLDER,
        "page_url": None,
        "source": "Wikimedia",
        "status": "pending",
        "query": query,
    }), 202


@app.get("/api/history")
def history():
    try:
        limit = int(request.args.get("limit", 30))
        offset = int(request.args.get("offset", 0))
    except ValueError:
        return jsonify({"error": "limit and offset must be numbers."}), 400
    scanner = request.args.get("scanner")
    return jsonify(database.get_history(limit=limit, offset=offset, scanner=scanner))


@app.delete("/api/history")
def delete_history():
    removed = database.clear_history()
    return jsonify({"deleted": removed})


@app.get("/api/stats")
def stats():
    return jsonify(database.get_stats())


@app.get("/snapshots/<path:filename>")
def snapshot(filename: str):
    return send_from_directory(SNAPSHOT_DIR, filename, max_age=3600)


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Image is larger than the 20 MB limit."}), 413


@app.errorhandler(404)
def not_found(_error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Endpoint not found."}), 404
    return render_template("index.html", default_confidence=int(DEFAULT_CONFIDENCE * 100)), 404


@app.errorhandler(500)
def server_error(_error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Internal server error."}), 500
    return "Internal server error", 500


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "127.0.0.1")
    registry.start_background()
    print(f"\n  AI Object Visualizer is starting on http://{host}:{port}\n")
    try:
        from waitress import serve

        serve(app, host=host, port=port, threads=6, channel_timeout=120)
    except ImportError:
        app.run(host=host, port=port, threaded=True)
