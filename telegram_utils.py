import os
import re
import secrets
import string
import requests
from werkzeug.security import generate_password_hash
from datetime import datetime, timedelta
from db_helpers import database_read, database_write



def generate_temp_password(length=10):
    chars = string.ascii_letters + string.digits
    return "".join(secrets.choice(chars) for _ in range(length))


def make_storage_key(family_name):
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", family_name.strip().lower()).strip("_")
    return f"{clean}_{secrets.token_hex(4)}"


def send_telegram_message(chat_id, text):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Missing TELEGRAM_BOT_TOKEN")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    try:
        response = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML"
            },
            timeout=10
        )
        return response.status_code == 200
    except Exception as e:
        print("Telegram send failed:", e)
        return False


def create_telegram_link_token(user_id, family_id, minutes=10):
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(minutes=minutes)

    database_write(
        """
        INSERT INTO telegram_link_tokens
        (token, user_id, family_id, expires_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            token,
            user_id,
            family_id,
            expires_at.isoformat()
        )
    )

    return token        


def get_telegram_link_token(token):
    rows = database_read(
        """
        SELECT *
        FROM telegram_link_tokens
        WHERE token = ?
        """,
        (token,),
        one=True
    )

    return rows


def mark_telegram_link_token_used(token):
    database_write(
        """
        UPDATE telegram_link_tokens
        SET used_at = ?
        WHERE token = ?
        """,
        (
            datetime.utcnow().isoformat(),
            token
        )
    )


def delete_expired_telegram_link_tokens():
    database_write(
        """
        DELETE FROM telegram_link_tokens
        WHERE expires_at < ?
        """,
        (datetime.utcnow().isoformat(),)
    )


def link_telegram_account(token, telegram_chat_id, telegram_username=None):
    now = datetime.utcnow().isoformat()

    token_row = database_read(
        """
        SELECT *
        FROM telegram_link_tokens
        WHERE token = ?
        """,
        (token,),
        one=True
    )

    if not token_row:
        return False, "Invalid or expired Telegram link."
    print(token_row)
    print(type(token_row))

    if token_row["used_at"]:
        return False, "This Telegram link was already used."

    if token_row["expires_at"] < now:
        return False, "This Telegram link has expired. Please create a new one from the dashboard."

    existing_user = database_read(
        """
        SELECT id
        FROM users
        WHERE telegram_chat_id = ?
        """,
        (str(telegram_chat_id),),
        one=True
    )

    if existing_user:
        return False, "This Telegram account is already connected to another Family Focus account."

    user_id = token_row["user_id"]

    database_write(
        """
        UPDATE users
        SET telegram_chat_id = ?,
            telegram_username = ?
        WHERE id = ?
        """,
        (
            str(telegram_chat_id),
            telegram_username,
            user_id
        )
    )

    database_write(
        """
        UPDATE telegram_link_tokens
        SET used_at = ?
        WHERE token = ?
        """,
        (
            now,
            token
        )
    )

    return True, "Telegram connected successfully."