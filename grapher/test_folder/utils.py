# utils.py

def build_url(user_id):
    return f"https://api.example.com/users/{user_id}"


def build_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }