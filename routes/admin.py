# routes/admin.py

from flask import Blueprint, flash, redirect, render_template, request, url_for
from auth import login_required
from werkzeug.security import generate_password_hash

from db_helpers import (
    database_read,
    database_write,
    ensure_family_dirs,
    get_telegram_registration_requests,
)
from telegram_utils import (
    generate_temp_password,
    make_storage_key,
    send_telegram_message,
)


admin_bp = Blueprint("admin", __name__)


@admin_bp.route("/admin/telegram-requests")
@login_required
def telegram_requests_admin():
    requests_rows = get_telegram_registration_requests()

    return render_template(
        "telegram_requests_admin.html",
        requests_rows=requests_rows
    )


@admin_bp.route(
    "/admin/telegram-requests/<int:request_id>/approve",
    methods=["POST"]
)
@login_required
def approve_telegram_request(request_id):
    req = database_read(
        """
        SELECT *
        FROM telegram_registration_requests
        WHERE id = ?
        """,
        (request_id,),
        one=True
    )

    if not req:
        flash("Telegram request not found.", "error")
        return redirect(url_for(".telegram_requests_admin"))

    if req["status"] != "pending":
        flash("Request already handled.", "warning")
        return redirect(url_for(".telegram_requests_admin"))

    family_name = req["family_name"].strip()
    email = req["email"].strip().lower()

    storage_key = make_storage_key(family_name)
    temp_password = generate_temp_password()
    password_hash = generate_password_hash(temp_password)

    family_id = database_write(
        """
        INSERT INTO families (
            family_name,
            client_id,
            storage_key,
            created_at
        )
        VALUES (?, ?, ?, datetime('now'))
        """,
        (family_name, storage_key, storage_key)
    )

    ensure_family_dirs(family_id)

    database_write(
        """
        INSERT INTO users (
            family_id,
            email,
            password_hash,
            telegram_chat_id,
            telegram_username,
            role,
            is_active,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, 'admin', 1, datetime('now'))
        """,
        (
            family_id,
            email,
            password_hash,
            req["telegram_chat_id"],
            req["telegram_username"]
        )
    )

    database_write(
        """
        UPDATE telegram_registration_requests
        SET status = 'approved'
        WHERE id = ?
        """,
        (request_id,)
    )

    login_url = request.host_url.rstrip("/") + url_for("login")

    send_telegram_message(
        req["telegram_chat_id"],
        (
            "🎉 <b>Family Focus Registration Approved</b>\n\n"
            f"Family: {family_name}\n\n"
            f"Login URL:\n{login_url}\n\n"
            f"Email:\n{email}\n\n"
            f"Temporary Password:\n{temp_password}\n\n"
            "Please log in and change your password."
        )
    )

    flash("Telegram registration approved successfully.", "success")
    return redirect(url_for(".telegram_requests_admin"))


@admin_bp.route(
    "/admin/telegram-requests/<int:request_id>/reject",
    methods=["POST"]
)
@login_required
def reject_telegram_request(request_id):
    req = database_read(
        """
        SELECT *
        FROM telegram_registration_requests
        WHERE id = ?
        """,
        (request_id,),
        one=True
    )

    if not req:
        flash("Telegram request not found.", "error")
        return redirect(url_for(".telegram_requests_admin"))

    if req["status"] != "pending":
        flash("Request already handled.", "warning")
        return redirect(url_for(".telegram_requests_admin"))

    database_write(
        """
        UPDATE telegram_registration_requests
        SET status = 'rejected'
        WHERE id = ?
        """,
        (request_id,)
    )

    send_telegram_message(
        req["telegram_chat_id"],
        (
            "❌ Your Family Focus registration request was not approved.\n\n"
            "Please contact the Family Focus administrator."
        )
    )

    flash("Telegram registration rejected.", "warning")
    return redirect(url_for(".telegram_requests_admin"))