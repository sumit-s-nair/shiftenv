# session_manager.py — Persistent HTTP session management
# Uses `requests.Session` with retry logic, connection pooling, and context manager.
# Must be migrated to `httpx.Client` with `httpx.HTTPTransport`.
#
# Migration challenges:
#   - requests.Session() → httpx.Client()
#   - requests.adapters.HTTPAdapter → httpx.HTTPTransport
#   - urllib3.util.retry.Retry → httpx.HTTPTransport(retries=N)
#   - session.mount("https://", adapter) → httpx.Client(transport=transport)
#   - timeout tuple (connect, read) → httpx.Timeout(connect=C, read=R)
#   - Context manager: both support __enter__/__exit__ but semantics differ

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from config import build_auth_headers, DEFAULT_BASE_URL, DEFAULT_TIMEOUT


class SessionManager:
    """Manages a persistent HTTP session with retry logic and auth."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, token: str = "",
                 max_retries: int = 3):
        self._session = requests.Session()
        self._base_url = base_url

        # Configure retry with backoff
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=10)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        # Auth headers
        if token:
            self._session.headers.update(build_auth_headers(token))

        # Timeout as (connect, read) tuple — must become httpx.Timeout in migration
        self._timeout = (5, 30)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._session.close()
        return False

    def get(self, path: str) -> dict:
        """GET a path relative to base_url."""
        url = f"{self._base_url.rstrip('/')}/{path.lstrip('/')}"
        response = self._session.get(url, timeout=self._timeout)
        response.raise_for_status()
        return response.json()

    def post(self, path: str, data: dict) -> dict:
        """POST JSON to a path relative to base_url."""
        url = f"{self._base_url.rstrip('/')}/{path.lstrip('/')}"
        response = self._session.post(url, json=data, timeout=self._timeout)
        response.raise_for_status()
        return response.json()

    def set_header(self, key: str, value: str) -> None:
        """Add or update a custom header."""
        self._session.headers[key] = value

    def close(self) -> None:
        """Close the underlying session."""
        self._session.close()
