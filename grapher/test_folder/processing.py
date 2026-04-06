# processing.py

from api import fetch_user_data


def extract_username(user_id, token):
    data = fetch_user_data(user_id, token)
    return data.get("username", "")


def extract_email(user_id, token):
    data = fetch_user_data(user_id, token)
    return data.get("email", "")