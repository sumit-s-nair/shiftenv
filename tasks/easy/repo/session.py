# session.py — Session and CSRF token management
# Currently uses `random` for token generation. Must be migrated to `secrets`.
#
# Migration challenges:
#   - random.getrandbits → secrets.token_urlsafe or secrets.token_hex
#   - is_token_expired is a red herring — pure logic, no random usage

import random
import hashlib
import time


def generate_session_id(user_id: str) -> str:
    """Generate a session ID by combining user_id with a random hex token.
    
    Format: {user_id}_{32 random hex chars}
    """
    token_part = "".join(
        random.choice("0123456789abcdef") for _ in range(32)
    )
    return f"{user_id}_{token_part}"


def generate_csrf_token() -> str:
    """Generate a CSRF token for form protection.
    
    Uses random.getrandbits to create a 256-bit token, then hashes it.
    """
    raw_bits = random.getrandbits(256)
    raw_hex = f"{raw_bits:064x}"
    return hashlib.sha256(raw_hex.encode()).hexdigest()


def is_token_expired(created_at: float, max_age: int = 3600) -> bool:
    """Check if a token has expired based on creation time.
    
    This function does NOT use random — it's pure logic.
    The agent should NOT modify this function (red herring).
    
    Args:
        created_at: Unix timestamp when token was created
        max_age: Maximum age in seconds (default 1 hour)
    Returns:
        True if the token is expired
    """
    return (time.time() - created_at) > max_age
