# api.py
import httpx as requests
def fetch_user_data(user_id, token):
    from utils import build_url, build_headers

    url = build_url(user_id)
    headers = build_headers(token)

    response = requests.get(url, headers=headers)
    
    return response.json()
