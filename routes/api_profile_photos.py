from pathlib import Path
from flask import Blueprint, jsonify, request, current_app

from db_helpers import database_read
from services.jwt_auth import (
    api_login_required,
    get_api_family_id
)
from services.profile_photo_service import save_profile_photo_for_member

api_profile_photos_bp = Blueprint(
    "api_profile_photos",
    __name__,
    url_prefix="/api/members"
)


@api_profile_photos_bp.route("/<int:member_id>/profile-photos", methods=["POST"])
@api_login_required
def api_upload_member_profile_photos(member_id):
    family_id = get_api_family_id()

    files = request.files.getlist("photos")

    if not files:
        return jsonify({
            "ok": False,
            "error": "No photos uploaded."
        }), 400

    saved = []
    skipped = []

    DeepFace = current_app.config.get("DeepFace")

    if DeepFace is None:
        return jsonify({
            "ok": False,
            "error": "DeepFace is not available."
        }), 500

    for file in files:
        result = save_profile_photo_for_member(
            family_id=family_id,
            member_id=member_id,
            file=file,
            DeepFace=DeepFace
        )

        if result["ok"]:
            saved.append(result["photo"])
        else:
            skipped.append({
                "filename": file.filename if file else None,
                "error": result["error"]
            })

    return jsonify({
        "ok": True,
        "saved_count": len(saved),
        "skipped_count": len(skipped),
        "photos": saved,
        "skipped": skipped
    }), 201