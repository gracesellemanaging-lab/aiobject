"""HTTP requests that back off when Wikimedia asks us to slow down."""
from __future__ import annotations

import time

import requests

RETRY_STATUS = {429, 500, 502, 503, 504}


def request(session: requests.Session, method: str, url: str, *, attempts: int = 2, **kwargs) -> requests.Response:
    response = session.request(method, url, **kwargs)
    for _ in range(attempts - 1):
        if response.status_code not in RETRY_STATUS:
            return response
        try:
            delay = float(response.headers.get("Retry-After") or 1.5)
        except ValueError:
            delay = 1.5
        time.sleep(min(max(delay, 0.5), 5.0))
        response.close()
        response = session.request(method, url, **kwargs)
    return response
