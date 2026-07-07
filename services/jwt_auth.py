import os
import jwt
from functools import wraps
from datetime import datetime, timedelta, timezone
from flask import request, jsonify, current_app


JWT_EXPIRES_HOURS = 24 * 30


def create_jwt_token(user):
    payload = {
        "user_id": user["id"],
        "family_id": user["family_id"],
        "email": user["email"],
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRES_HOURS)
    }

    return jwt.encode(
        payload,
        current_app.config["JWT_SECRET_KEY"],
        algorithm="HS256"
    )


def api_login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            return jsonify({"ok": False, "error": "Missing auth token"}), 401

        token = auth_header.replace("Bearer ", "").strip()

        try:
            payload = jwt.decode(
                token,
                current_app.config["JWT_SECRET_KEY"],
                algorithms=["HS256"]
            )

            request.user_id = payload["user_id"]
            request.family_id = payload["family_id"]
            request.email = payload["email"]

        except jwt.ExpiredSignatureError:
            return jsonify({"ok": False, "error": "Token expired"}), 401
        except Exception:
            return jsonify({"ok": False, "error": "Invalid token"}), 401

        return f(*args, **kwargs)

    return wrapper