# test_crypto.py — Grader for the easy task (random → secrets)
# Tests pass ONLY after BOTH crypto_utils.py AND session.py are migrated.
# token_manager.py is deprecated and not tested.

import ast
import os
import sys
import importlib.util
import pytest

REPO_DIR = os.path.join(os.path.dirname(__file__), "..", "repo")
CRYPTO_FILE = os.path.join(REPO_DIR, "crypto_utils.py")
SESSION_FILE = os.path.join(REPO_DIR, "session.py")

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


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# 1. Import checks — BOTH files must drop `random` and use `secrets`
# ===========================================================================
class TestImports:
    def test_crypto_no_random(self):
        assert "random" not in _get_imports(CRYPTO_FILE), \
            "crypto_utils.py still imports random"

    def test_crypto_has_secrets(self):
        assert "secrets" in _get_imports(CRYPTO_FILE), \
            "crypto_utils.py does not import secrets"

    def test_session_no_random(self):
        assert "random" not in _get_imports(SESSION_FILE), \
            "session.py still imports random"

    def test_session_has_secrets(self):
        assert "secrets" in _get_imports(SESSION_FILE), \
            "session.py does not import secrets"


# ===========================================================================
# 2. crypto_utils.py — functional tests
# ===========================================================================
class TestGenerateToken:
    def test_returns_string(self):
        mod = _load_module("crypto_utils", CRYPTO_FILE)
        assert isinstance(mod.generate_token(16), str)

    def test_correct_length(self):
        mod = _load_module("crypto_utils", CRYPTO_FILE)
        assert len(mod.generate_token(16)) == 32

    def test_hex_only(self):
        mod = _load_module("crypto_utils", CRYPTO_FILE)
        token = mod.generate_token(16)
        assert all(c in "0123456789abcdef" for c in token)

    def test_uniqueness_1000(self):
        """1000 tokens must all be unique (crypto-grade randomness)."""
        mod = _load_module("crypto_utils", CRYPTO_FILE)
        tokens = {mod.generate_token(16) for _ in range(1000)}
        assert len(tokens) == 1000, f"Only {len(tokens)} unique out of 1000"


class TestGeneratePassword:
    def test_correct_length(self):
        mod = _load_module("crypto_utils", CRYPTO_FILE)
        assert len(mod.generate_password(20)) == 20

    def test_custom_charset(self):
        mod = _load_module("crypto_utils", CRYPTO_FILE)
        pw = mod.generate_password(10, charset="abc")
        assert all(c in "abc" for c in pw)


class TestGenerateOTP:
    def test_correct_digits(self):
        mod = _load_module("crypto_utils", CRYPTO_FILE)
        otp = mod.generate_otp(8)
        assert len(otp) == 8 and otp.isdigit()

    def test_zero_padded(self):
        mod = _load_module("crypto_utils", CRYPTO_FILE)
        for _ in range(50):
            otp = mod.generate_otp(6)
            assert len(otp) == 6


class TestGenerateUUIDToken:
    def test_returns_32_hex(self):
        mod = _load_module("crypto_utils", CRYPTO_FILE)
        token = mod.generate_uuid_token()
        assert len(token) == 32
        assert all(c in "0123456789abcdef" for c in token)

    def test_uniqueness(self):
        mod = _load_module("crypto_utils", CRYPTO_FILE)
        tokens = {mod.generate_uuid_token() for _ in range(500)}
        assert len(tokens) == 500


class TestShuffleDeck:
    def test_returns_same_list(self):
        mod = _load_module("crypto_utils", CRYPTO_FILE)
        deck = list(range(52))
        result = mod.shuffle_deck(deck)
        assert result is deck  # in-place shuffle

    def test_all_elements_preserved(self):
        mod = _load_module("crypto_utils", CRYPTO_FILE)
        deck = list(range(52))
        mod.shuffle_deck(deck)
        assert sorted(deck) == list(range(52))

    def test_shuffle_changes_order(self):
        """Overwhelmingly likely to change order of 52 cards."""
        mod = _load_module("crypto_utils", CRYPTO_FILE)
        deck = list(range(52))
        original = list(deck)
        mod.shuffle_deck(deck)
        assert deck != original, "Shuffle did not change order"


# ===========================================================================
# 3. session.py — functional tests
# ===========================================================================
class TestSessionId:
    def test_contains_user_id(self):
        mod = _load_module("session", SESSION_FILE)
        sid = mod.generate_session_id("user42")
        assert sid.startswith("user42_")

    def test_token_part_is_hex(self):
        mod = _load_module("session", SESSION_FILE)
        sid = mod.generate_session_id("u1")
        token_part = sid.split("_", 1)[1]
        assert len(token_part) == 32
        assert all(c in "0123456789abcdef" for c in token_part)

    def test_uniqueness(self):
        mod = _load_module("session", SESSION_FILE)
        sids = {mod.generate_session_id("user") for _ in range(500)}
        assert len(sids) == 500


class TestCSRFToken:
    def test_returns_sha256_hex(self):
        mod = _load_module("session", SESSION_FILE)
        token = mod.generate_csrf_token()
        assert len(token) == 64  # SHA-256 hex digest
        assert all(c in "0123456789abcdef" for c in token)

    def test_uniqueness(self):
        mod = _load_module("session", SESSION_FILE)
        tokens = {mod.generate_csrf_token() for _ in range(500)}
        assert len(tokens) == 500


class TestTokenExpiry:
    def test_not_expired(self):
        import time
        mod = _load_module("session", SESSION_FILE)
        assert mod.is_token_expired(time.time(), max_age=3600) is False

    def test_expired(self):
        import time
        mod = _load_module("session", SESSION_FILE)
        old_time = time.time() - 7200  # 2 hours ago
        assert mod.is_token_expired(old_time, max_age=3600) is True
