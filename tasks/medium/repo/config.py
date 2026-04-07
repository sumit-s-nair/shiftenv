# config.py — Shared configuration for the API client
# This file does NOT need migration. It's a dependency for realism.

DEFAULT_BASE_URL = "https://api.example.com/v1"
DEFAULT_TIMEOUT = 30
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}


def build_url(base: str, path: str) -> str:
    """Join base URL with an API path."""
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def build_auth_headers(token: str) -> dict:
    """Build authorization headers from a bearer token."""
    headers = dict(DEFAULT_HEADERS)
    headers["Authorization"] = f"Bearer {token}"
    return headers
