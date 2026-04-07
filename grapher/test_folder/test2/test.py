import httpx as rq

def quick_ping(url):
    """Uses httpx now!"""
    return rq.get(url).status_code
