# main.py

from formatter import format_user_info


def main():
    user_id = 1
    token = "dummy_token"

    result = format_user_info(user_id, token)
    print(result)


if __name__ == "__main__":
    main()