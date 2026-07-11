# routes/telegram_api.py

from flask import Blueprint, request, jsonify
from deepface import DeepFace

from db_helpers import (
    get_user_by_telegram_chat_id,
    get_owned_member_for_family,
    get_telegram_connection_status,
        database_read,
    get_user_by_telegram_chat_id,
    link_telegram_account,
    get_family_members_by_telegram_chat_id
)

from services.profile_photo_service import save_member_profile_photo


telegram_api_bp = Blueprint("telegram_api", __name__)


@telegram_api_bp.route("/api/telegram/connection-status", methods=["POST"])
def api_telegram_connection_status():
    data = request.get_json(silent=True) or {}
    chat_id = str(data.get("telegram_chat_id", "")).strip()

    if not chat_id:
        return jsonify({
            "linked": False,
            "error": "missing telegram_chat_id"
        }), 400

    user = get_telegram_connection_status(chat_id)

    if not user:
        return jsonify({
            "linked": False
        })

    return jsonify({
        "linked": True,
        "family_id": user["family_id"],
        "family_name": user["family_name"],
        "user_name": f'{user["first_name"] or ""} {user["last_name"] or ""}'.strip()
    })

@telegram_api_bp..route("/api/telegram/family-members", methods=["GET"])
def api_telegram_family_members():
    telegram_chat_id = request.args.get("telegram_chat_id")

    if not telegram_chat_id:
        return jsonify({
            "success": False,
            "message": "Missing telegram_chat_id"
        }), 400

    members = get_family_members_by_telegram_chat_id(telegram_chat_id)

    if members is None:
        return jsonify({
            "success": False,
            "message": "Telegram account is not linked"
        }), 404

    return jsonify({
        "success": True,
        "members": members
    })


@telegram_api_bp.route("/api/telegram/upload-member-photo", methods=["POST"])
def telegram_upload_member_photo():
    telegram_chat_id = request.form.get("telegram_chat_id")
    member_id = request.form.get("member_id")

    if not telegram_chat_id:
        return jsonify({"ok": False, "error": "Missing telegram_chat_id"}), 400

    if not member_id:
        return jsonify({"ok": False, "error": "Missing member_id"}), 400

    if "photo" not in request.files:
        return jsonify({"ok": False, "error": "Missing photo"}), 400

    user = get_user_by_telegram_chat_id(str(telegram_chat_id))

    if not user:
        return jsonify({
            "ok": False,
            "error": "Telegram account is not linked."
        }), 403

    family_id = user["family_id"]

    member = get_owned_member_for_family(int(member_id), family_id)

    if not member:
        return jsonify({
            "ok": False,
            "error": "Member does not belong to this family."
        }), 403

    result = save_member_profile_photo(
        family_id=family_id,
        member_id=int(member_id),
        file=request.files["photo"],
        DeepFace=DeepFace
    )

    if not result.get("ok"):
        return jsonify(result), 400

    photo = result["photo"]

    return jsonify({
        "ok": True,
        "message": f"Profile photo added for {member['name']}",
        "member_name": member["name"],
        "file_path": photo["file_path"],
        "photo_id": photo["id"],
        "learned": photo["learned"]
    })

    @app.route("/api/telegram/status", methods=["POST"])


def api_telegram_status():
    data = request.get_json(silent=True) or {}
    chat_id = str(data.get("telegram_chat_id", "")).strip()

    if not chat_id:
        return jsonify({"error": "missing telegram_chat_id"}), 400

    row = database_read(
        """
        SELECT family_name, email, status, created_at
        FROM telegram_registration_requests
        WHERE telegram_chat_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (chat_id,),
        one=True,
    )

    if not row:
        return jsonify({"found": False})

    return jsonify({
        "found": True,
        "family_name": row["family_name"],
        "email": row["email"],
        "status": row["status"],
        "created_at": row["created_at"],
    })


@telegram_api_bp.route("/api/telegram/link-account", methods=["POST"])
def api_telegram_link_account():
    data = request.get_json(silent=True) or {}

    token = data.get("token")
    telegram_chat_id = data.get("telegram_chat_id")
    telegram_username = data.get("telegram_username")

    if not token or not telegram_chat_id:
        return jsonify({
            "success": False,
            "message": "Missing token or telegram_chat_id."
        }), 400

    success, message = link_telegram_account(
        token=token,
        telegram_chat_id=telegram_chat_id,
        telegram_username=telegram_username
    )

    return jsonify({
        "success": success,
        "message": message
    })

@telegram_api_bp.route("/api/telegram/start-info", methods=["POST"])
def api_telegram_start_info():
    data = request.get_json(silent=True) or {}

    chat_id = str(data.get("telegram_chat_id", "")).strip()

    if not chat_id:
        return jsonify({"linked": False}), 400

    user = get_user_by_telegram_chat_id(chat_id)

    if not user:
        return jsonify({"linked": False})

    family_id = user["family_id"]

    family = database_read(
        """
        SELECT family_name
        FROM families
        WHERE id = ?
        """,
        (family_id,),
        one=True,
    )

    member_count = database_read(
        """
        SELECT COUNT(*) AS cnt
        FROM family_members
        WHERE family_id = ?
        """,
        (family_id,),
        one=True,
    )["cnt"]

    return jsonify({
        "linked": True,
        "family_name": family["family_name"],
        "member_count": member_count
    })

