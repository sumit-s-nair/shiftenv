import requests
import requests as rq
import json

# --- 0. INTERNAL UTILITIES (Leaf Nodes / No Library) ---
# These have 0 internal dependencies and use 0 library calls.

def build_api_url(endpoint):
    return f"https://api.myapp.com/v1/{endpoint}"

def get_standard_headers():
    return {"Content-Type": "application/json", "User-Agent": "MigrationBot/1.0"}

def validate_payload(data):
    return isinstance(data, dict) and len(data) > 0

def log_error(msg):
    print(f"[ERROR] {msg}")


# --- 1. SIMPLE LEAF NODES (Direct Library Usage / 0 Internal Deps) ---

def quick_ping(url):
    """Uses requests, but calls nothing else."""
    return rq.get(url).status_code


# --- 2. MID-TIER FUNCTIONS (Library Usage + Internal Dependencies) ---

def fetch_user_profile(user_id):
    """
    Uses: requests.get
    Internal Deps: build_api_url, get_standard_headers
    """
    url = build_api_url(f"users/{user_id}")
    headers = get_standard_headers()
    return requests.get(url, headers=headers).json()

def submit_transaction(tx_data):
    """
    Uses: requests.post
    Internal Deps: build_api_url, validate_payload, log_error
    """
    if not validate_payload(tx_data):
        log_error("Invalid transaction data")
        return None
    
    url = build_api_url("transactions")
    return requests.post(url, json=tx_data).status_code

def update_settings(patch_data):
    """
    Uses: requests.patch
    Internal Deps: get_standard_headers, build_api_url
    """
    url = build_api_url("settings")
    return requests.patch(url, headers=get_standard_headers(), json=patch_data)


# --- 3. HIGH-TIER ORCHESTRATORS (Recursive Internal Dependencies) ---

def sync_account(user_id, data):
    """
    Uses: No direct requests (usually)
    Internal Deps: fetch_user_profile, submit_transaction
    """
    profile = fetch_user_profile(user_id)
    if profile.get("active"):
        return submit_transaction(data)
    return False

def maintenance_check():
    """
    Uses: No direct requests
    Internal Deps: quick_ping, log_error
    """
    status = quick_ping("https://status.myapp.com")
    if status != 200:
        log_error("System is down!")

def full_system_init(user_id):
    """
    The 'Root' node.
    Internal Deps: sync_account, maintenance_check, update_settings
    """
    maintenance_check()
    update_settings({"status": "initializing"})
    return sync_account(user_id, {"init": True})