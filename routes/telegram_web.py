# routes/telegram_web.py

from flask import Blueprint, abort, redirect, session, url_for

from db_helpers import get_user_by_telegram_chat_id
from telegram_utils import create_telegram_link_token,verify_telegram_album_token


telegram_web_bp = Blueprint("telegram_web", __name__)


@telegram_web_bp.route("/telegram/album/<token>")
def telegram_album_login(token):
    try:
        data = verify_telegram_album_token(token, max_age=600)
    except Exception:
        abort(403)

    if not data:
        abort(403)

    family_id = data.get("family_id")
    telegram_chat_id = str(data.get("telegram_chat_id", ""))

    if not family_id or not telegram_chat_id:
        abort(403)

    user = get_user_by_telegram_chat_id(telegram_chat_id)

    if not user or user["family_id"] != family_id:
        abort(403)

    session.clear()
    session["user_id"] = user["id"]
    session["family_id"] = family_id
    session["email"] = user["email"]
    session["login_source"] = "telegram_signed_link"

    return redirect(url_for("family_album"))

@telegram_web_bp.route("/telegram/connect")
@login_required
def telegram_connect():
    token = create_telegram_link_token(
        user_id=session["user_id"],
        family_id=session["family_id"]
    )

    bot_username = os.getenv("TELEGRAM_BOT_USERNAME")

    if not bot_username:
        flash("Telegram bot is not configured.", "error")
        return redirect(url_for("dashboard"))

    return redirect(
        f"https://t.me/{bot_username}?start=link_{token}"
    )    