# test_api.py — Grader for the medium task (requests → httpx)
# Tests pass ONLY after all 3 files are migrated: api_client, session_manager, downloader.
# Uses unittest.mock — no extra deps.

import ast
import inspect
import os
import sys
import importlib.util
import pytest
from unittest.mock import patch, MagicMock

REPO_DIR = os.path.join(os.path.dirname(__file__), "..", "repo")
API_FILE = os.path.join(REPO_DIR, "api_client.py")
SESSION_FILE = os.path.join(REPO_DIR, "session_manager.py")
DOWNLOAD_FILE = os.path.join(REPO_DIR, "downloader.py")
ALL_FILES = [API_FILE, SESSION_FILE, DOWNLOAD_FILE]

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)


def _get_imports(filepath):
    with open(filepath, "r") as f:
        tree = ast.parse(f.read(), filename=filepath)
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def _get_source(filepath):
    with open(filepath, "r") as f:
        return f.read()


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# LAYER 1 — Import checks (6 tests)
# ===========================================================================
class TestImportsApiClient:
    def test_no_requests(self):
        assert "requests" not in _get_imports(API_FILE), \
            "api_client.py still imports requests"

    def test_has_httpx(self):
        assert "httpx" in _get_imports(API_FILE), \
            "api_client.py does not import httpx"


class TestImportsSessionManager:
    def test_no_requests(self):
        assert "requests" not in _get_imports(SESSION_FILE), \
            "session_manager.py still imports requests"

    def test_has_httpx(self):
        assert "httpx" in _get_imports(SESSION_FILE), \
            "session_manager.py does not import httpx"


class TestImportsDownloader:
    def test_no_requests(self):
        assert "requests" not in _get_imports(DOWNLOAD_FILE), \
            "downloader.py still imports requests"

    def test_has_httpx(self):
        assert "httpx" in _get_imports(DOWNLOAD_FILE), \
            "downloader.py does not import httpx"


# ===========================================================================
# LAYER 2 — API surface checks (8 tests)
# ===========================================================================
class TestAPISurface:
    def test_no_allow_redirects_in_api_client(self):
        src = _get_source(API_FILE)
        assert "allow_redirects" not in src, \
            "api_client.py still uses allow_redirects (should be follow_redirects)"

    def test_follow_redirects_in_api_client(self):
        src = _get_source(API_FILE)
        assert "follow_redirects" in src, \
            "api_client.py should use follow_redirects"

    def test_no_allow_redirects_in_downloader(self):
        src = _get_source(DOWNLOAD_FILE)
        assert "allow_redirects" not in src, \
            "downloader.py still uses allow_redirects"

    def test_follow_redirects_in_downloader(self):
        src = _get_source(DOWNLOAD_FILE)
        assert "follow_redirects" in src, \
            "downloader.py should use follow_redirects"

    def test_no_iter_content_in_downloader(self):
        src = _get_source(DOWNLOAD_FILE)
        assert "iter_content" not in src, \
            "downloader.py still uses iter_content (should be iter_bytes)"

    def test_no_requests_session(self):
        src = _get_source(SESSION_FILE)
        assert "requests.Session" not in src, \
            "session_manager.py still uses requests.Session"

    def test_httpx_client_in_session_manager(self):
        src = _get_source(SESSION_FILE)
        assert "httpx.Client" in src or "Client(" in src, \
            "session_manager.py should use httpx.Client"

    def test_httpx_timeout_in_session_manager(self):
        """Tuple timeout (5, 30) must become httpx.Timeout object."""
        src = _get_source(SESSION_FILE)
        assert "Timeout" in src, \
            "session_manager.py should use httpx.Timeout, not a tuple"
        assert "(5, 30)" not in src and "(5,30)" not in src, \
            "session_manager.py should convert tuple timeout to httpx.Timeout"


# ===========================================================================
# LAYER 3 — Functional tests (8 tests)
# ===========================================================================
class TestApiClientFunctional:
    @patch("httpx.get")
    def test_fetch_data(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"id": 1}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        mod = _load_module("api_client", API_FILE)
        result = mod.fetch_data("https://example.com/data")
        assert result == {"id": 1}
        mock_get.assert_called_once()

    @patch("httpx.post")
    def test_post_json(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        mod = _load_module("api_client", API_FILE)
        result = mod.post_json("https://example.com/api", {"key": "val"})
        assert result == {"status": "ok"}

    @patch("httpx.head")
    def test_check_health(self, mock_head):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_head.return_value = mock_resp

        mod = _load_module("api_client", API_FILE)
        assert mod.check_health("https://example.com") == 200
        call_kwargs = mock_head.call_args
        assert "follow_redirects" in str(call_kwargs), \
            "check_health should pass follow_redirects"


class TestSessionManagerFunctional:
    def test_class_exists(self):
        mod = _load_module("session_manager", SESSION_FILE)
        assert hasattr(mod, "SessionManager")

    def test_context_manager(self):
        """SessionManager must work as a context manager."""
        mod = _load_module("session_manager", SESSION_FILE)
        sm_cls = mod.SessionManager
        assert hasattr(sm_cls, "__enter__") and hasattr(sm_cls, "__exit__"), \
            "SessionManager must support context manager protocol"

    def test_has_close_method(self):
        mod = _load_module("session_manager", SESSION_FILE)
        sm = mod.SessionManager.__new__(mod.SessionManager)
        assert callable(getattr(sm, "close", None))


class TestDownloaderFunctional:
    def test_download_file_callable(self):
        mod = _load_module("downloader", DOWNLOAD_FILE)
        assert callable(mod.download_file)

    def test_download_parallel_is_async(self):
        """download_parallel must be an async function after migration."""
        mod = _load_module("downloader", DOWNLOAD_FILE)
        assert inspect.iscoroutinefunction(mod.download_parallel), \
            "download_parallel should be async (use httpx.AsyncClient)"
