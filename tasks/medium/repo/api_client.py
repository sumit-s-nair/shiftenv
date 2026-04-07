# api_client.py — Basic HTTP operations
# Uses `requests` for GET/POST/HEAD. Must be migrated to `httpx`.
#
# Migration challenges:
#   - requests.get/post/head → httpx.get/post/head
#   - allow_redirects=True → follow_redirects=True
#   - timeout as int → httpx.Timeout object preferred

import requests
from config import build_auth_headers, DEFAULT_TIMEOUT


def fetch_data(url: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Fetch JSON data from a URL via GET request."""
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def post_json(url: str, data: dict, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Send JSON data via POST and return the response."""
    response = requests.post(url, json=data, timeout=timeout)
    response.raise_for_status()
    return response.json()


def check_health(url: str) -> int:
    """Send a HEAD request, following redirects. Returns status code."""
    response = requests.head(url, allow_redirects=True, timeout=DEFAULT_TIMEOUT)
    return response.status_code


def fetch_with_auth(url: str, token: str) -> dict:
    """Fetch JSON using bearer token authentication."""
    headers = build_auth_headers(token)
    response = requests.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()
