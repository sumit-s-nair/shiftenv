# formatter.py

from processing import extract_username, extract_email


def format_user_info(user_id, token):
    username = extract_username(user_id, token)
    email = extract_email(user_id, token)

    return f"{username} <{email}>"