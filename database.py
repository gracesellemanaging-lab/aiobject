"""SQLite persistence for scan history and dashboard statistics."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from ai.config import DB_PATH, SNAPSHOT_DIR
from ai.utils import save_snapshot

_lock = threading.RLock()
_initialized = False
DEDUPE_WINDOW = 8.0
MAX_SAVED_PER_SCAN = 6


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def init_db() -> None:
    global _initialized
    with _lock:
        if _initialized:
            return
        connection = _connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    scanner TEXT NOT NULL,
                    label TEXT NOT NULL,
                    raw_label TEXT,
                    category TEXT,
                    confidence REAL,
                    source TEXT,
                    description TEXT,
                    reference_key TEXT,
                    snapshot TEXT,
                    attributes TEXT,
                    details TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_detections_ts ON detections (ts DESC);
                CREATE INDEX IF NOT EXISTS idx_detections_scanner ON detections (scanner);
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
                """
            )
            connection.commit()
            _initialized = True
        finally:
            connection.close()


def _get_meta(connection: sqlite3.Connection, key: str, default: int = 0) -> int:
    row = connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    try:
        return int(row["value"]) if row else default
    except (TypeError, ValueError):
        return default


def _set_meta(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )


def log_scan(scanner: str, detections: list[dict], image, notes: list[str] | None = None,
             save_thumbnails: bool = True) -> int:
    """Persist up to a few non-duplicate detections from one scan."""
    if not detections:
        return 0
    init_db()
    inserted = 0
    now = time.time()
    with _lock:
        connection = _connect()
        try:
            recent_keys = {
                row["raw_label"] or row["label"]
                for row in connection.execute(
                    "SELECT label, raw_label FROM detections WHERE scanner = ? AND ts >= ?",
                    (scanner, now - DEDUPE_WINDOW),
                ).fetchall()
            }
            total_meta = _get_meta(connection, "total_detections")
            total_scans = _get_meta(connection, "total_scans")
            for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True)[:MAX_SAVED_PER_SCAN]:
                key = detection.get("reference_key") or detection["label"]
                if key in recent_keys:
                    continue
                recent_keys.add(key)
                snapshot = None
                if save_thumbnails:
                    snapshot = save_snapshot(
                        image,
                        detection.get("bbox") if detection.get("localized") else None,
                        prefix=scanner,
                    )
                connection.execute(
                    """
                    INSERT INTO detections
                        (ts, scanner, label, raw_label, category, confidence, source,
                         description, reference_key, snapshot, attributes, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        now,
                        scanner,
                        detection["label"],
                        detection.get("reference_key"),
                        detection.get("category"),
                        detection.get("confidence"),
                        detection.get("source"),
                        detection.get("description"),
                        detection.get("reference_key"),
                        snapshot,
                        json.dumps(detection.get("attributes") or {}, ensure_ascii=False),
                        json.dumps({
                            "details": detection.get("details") or {},
                            "note": detection.get("note"),
                            "notes": notes or [],
                        }, ensure_ascii=False),
                    ),
                )
                inserted += 1

            if inserted:
                _set_meta(connection, "total_detections", total_meta + inserted)
                _set_meta(connection, "total_scans", total_scans + 1)
                _set_meta(connection, "last_scan", round(now, 3))
                if not _get_meta(connection, "first_scan"):
                    _set_meta(connection, "first_scan", round(now, 3))
                connection.commit()
            return inserted
        finally:
            connection.close()


def _row_to_dict(row: sqlite3.Row) -> dict:
    def loads(value):
        if not value:
            return {}
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return {}

    details = loads(row["details"])
    return {
        "id": row["id"],
        "timestamp": row["ts"],
        "scanner": row["scanner"],
        "label": row["label"],
        "raw_label": row["raw_label"],
        "category": row["category"],
        "confidence": row["confidence"],
        "confidence_percent": round((row["confidence"] or 0) * 100, 1),
        "source": row["source"],
        "description": row["description"],
        "reference_key": row["reference_key"],
        "snapshot_url": f"/snapshots/{row['snapshot']}" if row["snapshot"] else None,
        "attributes": loads(row["attributes"]),
        "details": details.get("details") or {},
        "note": details.get("note"),
        "notes": details.get("notes") or [],
    }


def get_history(limit: int = 30, offset: int = 0, scanner: str | None = None) -> dict:
    init_db()
    limit = max(1, min(100, int(limit)))
    offset = max(0, int(offset))
    where = ""
    params: list[Any] = []
    if scanner and scanner in {"smart", "person", "object", "fruit", "gesture"}:
        where = "WHERE scanner = ?"
        params.append(scanner)
    with _lock:
        connection = _connect()
        try:
            total = connection.execute(f"SELECT COUNT(*) AS n FROM detections {where}", params).fetchone()["n"]
            rows = connection.execute(
                f"SELECT * FROM detections {where} ORDER BY ts DESC, id DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            items = [_row_to_dict(row) for row in rows]
        finally:
            connection.close()
    return {"items": items, "total": total, "limit": limit, "offset": offset}


def clear_history() -> int:
    init_db()
    with _lock:
        connection = _connect()
        try:
            row = connection.execute("SELECT COUNT(*) AS n FROM detections").fetchone()
            count = row["n"] if row else 0
            connection.execute("DELETE FROM detections")
            connection.execute("DELETE FROM meta WHERE key IN ('total_scans', 'total_detections', 'last_scan', 'first_scan')")
            connection.commit()
        finally:
            connection.close()
    for path in SNAPSHOT_DIR.glob("*.jpg"):
        try:
            path.unlink()
        except OSError:
            pass
    return count


def get_stats() -> dict:
    init_db()
    now = time.time()
    day_ago = now - 86400
    week_ago = now - 7 * 86400
    with _lock:
        connection = _connect()
        try:
            total_detections = connection.execute("SELECT COUNT(*) AS n FROM detections").fetchone()["n"]
            total_scans = _get_meta(connection, "total_scans")
            today = connection.execute("SELECT COUNT(*) AS n FROM detections WHERE ts >= ?", (day_ago,)).fetchone()["n"]
            week = connection.execute("SELECT COUNT(*) AS n FROM detections WHERE ts >= ?", (week_ago,)).fetchone()["n"]
            people = connection.execute("SELECT COUNT(*) AS n FROM detections WHERE category = 'People'").fetchone()["n"]
            objects = connection.execute(
                "SELECT COUNT(*) AS n FROM detections WHERE category IS NOT NULL "
                "AND category NOT IN ('People', 'Fruits', 'Vegetables', 'Expressions', 'Gestures', 'Pose')"
            ).fetchone()["n"]
            fruits = connection.execute(
                "SELECT COUNT(*) AS n FROM detections WHERE category IN ('Fruits', 'Vegetables')"
            ).fetchone()["n"]
            expressions = connection.execute(
                "SELECT COUNT(*) AS n FROM detections WHERE category = 'Expressions'"
            ).fetchone()["n"]
            average = connection.execute(
                "SELECT AVG(confidence) AS value FROM detections WHERE confidence IS NOT NULL"
            ).fetchone()["value"]
            expression_rows = connection.execute(
                "SELECT label, COUNT(*) AS n, AVG(confidence) AS confidence FROM detections "
                "WHERE category = 'Expressions' GROUP BY label ORDER BY n DESC LIMIT 6"
            ).fetchall()
            categories = connection.execute(
                "SELECT category, COUNT(*) AS n FROM detections GROUP BY category ORDER BY n DESC LIMIT 8"
            ).fetchall()
            labels = connection.execute(
                "SELECT label, category, COUNT(*) AS n FROM detections GROUP BY label ORDER BY n DESC LIMIT 8"
            ).fetchall()
            scanners = connection.execute(
                "SELECT scanner, COUNT(*) AS n FROM detections GROUP BY scanner ORDER BY n DESC"
            ).fetchall()
            last = connection.execute("SELECT MAX(ts) AS ts FROM detections").fetchone()["ts"]
            days = connection.execute(
                "SELECT ts FROM detections WHERE ts >= ?", (now - 6 * 86400,)
            ).fetchall()
        finally:
            connection.close()

    daily_counter: Counter = Counter()
    for row in days:
        daily_counter[datetime.fromtimestamp(row["ts"]).strftime("%Y-%m-%d")] += 1
    daily = []
    for offset in range(6, -1, -1):
        day = (datetime.now() - timedelta(days=offset)).strftime("%Y-%m-%d")
        daily.append({"date": day, "count": daily_counter.get(day, 0)})

    avg_confidence = round(float(average), 4) if average is not None else 0.0
    return {
        "total_scans": total_scans,
        "total_detections": total_detections,
        "detections_24h": today,
        "detections_7d": week,
        "people": people,
        "objects": objects,
        "fruits": fruits,
        "expressions": expressions,
        "avg_confidence": avg_confidence,
        "avg_confidence_percent": round(avg_confidence * 100, 1),
        "last_scan": last,
        "by_expression": [
            {"label": row["label"], "count": row["n"], "confidence_percent": round((row["confidence"] or 0) * 100, 1)}
            for row in expression_rows
        ],
        "by_category": [{"category": row["category"] or "Unknown", "count": row["n"]} for row in categories],
        "top_labels": [
            {"label": row["label"], "category": row["category"], "count": row["n"]} for row in labels
        ],
        "by_scanner": [{"scanner": row["scanner"], "count": row["n"]} for row in scanners],
        "daily": daily,
    }


init_db()
