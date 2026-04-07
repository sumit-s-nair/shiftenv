# crypto_utils.py — Core cryptographic utilities
# Currently uses `random` (insecure for crypto!)
# Must be migrated to `secrets`
#
# Migration challenges:
#   - random.choices → secrets.token_hex (for hex tokens)
#   - random.choice → secrets.choice
#   - random.randint → secrets.randbelow
#   - random.getrandbits → secrets.token_bytes or secrets.randbits
#   - random.shuffle → secrets.SystemRandom().shuffle (no direct equivalent!)

import random
import string


def generate_token(length: int = 32) -> str:
    """Generate a random hex token of the given byte-length.
    
    Returns a hex string of length * 2 characters.
    """
    chars = "0123456789abcdef"
    return "".join(random.choices(chars, k=length * 2))


def generate_password(length: int = 16, charset: str | None = None) -> str:
    """Generate a random password from the given character set.
    
    If charset is None, uses ASCII letters + digits + punctuation.
    """
    if charset is None:
        charset = string.ascii_letters + string.digits + string.punctuation
    return "".join(random.choice(charset) for _ in range(length))


def generate_otp(digits: int = 6) -> str:
    """Generate a numeric one-time password with exactly `digits` digits.
    
    Returns a zero-padded string, e.g. "007432" for digits=6.
    """
    upper = 10 ** digits
    code = random.randint(0, upper - 1)
    return str(code).zfill(digits)


def generate_uuid_token() -> str:
    """Generate a UUID-like token using random bits.
    
    Format: 32 hex chars from 128 random bits.
    Uses random.getrandbits(128) — must use secrets equivalent.
    """
    bits = random.getrandbits(128)
    return f"{bits:032x}"


def shuffle_deck(deck: list) -> list:
    """Shuffle a list in-place using random.shuffle.
    
    This is the hardest function to migrate because secrets has no
    direct shuffle equivalent. Agent must use secrets.SystemRandom().shuffle()
    or implement a Fisher-Yates shuffle with secrets.randbelow().
    
    Returns the shuffled list (same object, mutated in-place).
    """
    random.shuffle(deck)
    return deck
