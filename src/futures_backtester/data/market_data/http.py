"""HTTP access with an on-disk cache, retries and per-host rate limiting.

Raw responses are cached under ``.cache/market_data`` in the working directory
(override with ``MARKET_DATA_CACHE``), so rebuilding a dataset does not hit the
sources again. Set ``MARKET_DATA_REFRESH=1`` to force fresh downloads.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

CACHE_DIR = Path(os.environ.get("MARKET_DATA_CACHE", Path.cwd() / ".cache" / "market_data"))
REFRESH = os.environ.get("MARKET_DATA_REFRESH", "0") == "1"

# SEC asks automated clients to identify themselves. No personal data is sent by
# default; set SEC_USER_AGENT="Your Name your@email" to comply with SEC policy.
DEFAULT_USER_AGENT = "market-data-builder/1.0 (research script)"
SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", DEFAULT_USER_AGENT)

# Seconds between consecutive requests to the same host.
MIN_INTERVAL = {
    "data.sec.gov": 0.12,  # SEC fair-access limit is 10 requests/second
    "www.sec.gov": 0.12,
    "query1.finance.yahoo.com": 0.35,
    "query2.finance.yahoo.com": 0.35,
}

_SESSION = requests.Session()
_LAST_CALL: dict[str, float] = {}


def _cache_path(url: str, params: dict | None) -> Path:
    key = hashlib.sha256((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
    return CACHE_DIR / key[:2] / key[2:34]


def get(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    cache: bool = True,
    timeout: float = 60.0,
    retries: int = 5,
) -> bytes:
    """Return the response body for ``url``, using the on-disk cache when possible."""

    path = _cache_path(url, params)
    if cache and not REFRESH and path.exists():
        return path.read_bytes()

    host = urlparse(url).netloc
    if "sec.gov" in host:
        agent = SEC_USER_AGENT
    elif "stlouisfed.org" in host:
        agent = "curl/8.5.0"  # FRED's CSV endpoint stalls browser-like agents from scripts
    else:
        agent = "Mozilla/5.0"
    request_headers = {"User-Agent": agent}
    request_headers.update(headers or {})

    last_error: Exception | None = None
    for attempt in range(retries):
        wait = MIN_INTERVAL.get(host, 0.0) - (time.monotonic() - _LAST_CALL.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL[host] = time.monotonic()
        try:
            response = _SESSION.get(url, params=params, headers=request_headers, timeout=timeout)
        except requests.RequestException as exc:  # network hiccup: back off and retry
            last_error = exc
            time.sleep(2**attempt)
            continue
        if response.status_code == 429 or response.status_code >= 500:
            last_error = requests.HTTPError(f"{response.status_code} for {response.url}")
            time.sleep(2 ** (attempt + 1))
            continue
        response.raise_for_status()
        body = response.content
        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        return body
    raise RuntimeError(f"failed to download {url}: {last_error}")


def get_json(url: str, **kwargs) -> dict:
    return json.loads(get(url, **kwargs))
