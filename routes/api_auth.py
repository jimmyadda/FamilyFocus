from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from db_helpers import database_read, database_write
from services.jwt_auth import create_jwt_token, api_login_required


api_auth_bp = Blueprint("api_auth", __name__, url_prefix="/api/auth")


@api_auth_bp.route("/login", methods=["POST"])
def api_auth_login():
    data = request.get_json(silent=True) or {}

    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()

    user = database_read("""
        SELECT users.*, families.family_name
        FROM users
        JOIN families ON families.id = users.family_id
        WHERE users.email = ?
          AND users.is_active = 1
    """, (email,), one=True)

    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"ok": False, "error": "Invalid email or password"}), 401

    token = create_jwt_token(user)

    database_write(
        "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
        (user["id"],)
    )

    return jsonify({
        "ok": True,
        "token": token,
        "user": {
            "id": user["id"],
            "family_id": user["family_id"],
            "email": user["email"],
            "first_name": user["first_name"],
            "last_name": user["last_name"],
            "family_name": user["family_name"],
            "role": user["role"]
        }
    })


@api_auth_bp.route("/me", methods=["GET"])
@api_login_required
def api_auth_me():
    user = database_read("""
        SELECT users.*, families.family_name
        FROM users
        JOIN families ON families.id = users.family_id
        WHERE users.id = ?
          AND users.family_id = ?
          AND users.is_active = 1
    """, (request.user_id, request.family_id), one=True)

    if not user:
        return jsonify({"ok": False, "error": "User not found"}), 404

    return jsonify({
        "ok": True,
        "user": {
            "id": user["id"],
            "family_id": user["family_id"],
            "email": user["email"],
            "first_name": user["first_name"],
            "last_name": user["last_name"],
            "family_name": user["family_name"],
            "role": user["role"]
        }
    })

@api_auth_bp.route("/register", methods=["POST"])
def api_auth_register():
    data = request.get_json(silent=True) or {}

    family_name = data.get("family_name", "").strip()
    first_name = data.get("first_name", "").strip()
    last_name = data.get("last_name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()

    if not family_name or not email or not password:
        return jsonify({
            "ok": False,
            "error": "Family name, email and password are required."
        }), 400

    existing = database_read(
        "SELECT id FROM users WHERE email = ?",
        (email,),
        one=True
    )

    if existing:
        return jsonify({
            "ok": False,
            "error": "Email already exists."
        }), 409

    client_id = family_name.lower().replace(" ", "_")

    database_write("""
        INSERT INTO families (family_name, client_id)
        VALUES (?, ?)
    """, (family_name, client_id))

    family = database_read("""
        SELECT id
        FROM families
        WHERE client_id = ?
        ORDER BY id DESC
        LIMIT 1
    """, (client_id,), one=True)

    if not family:
        return jsonify({
            "ok": False,
            "error": "Registration failed while creating family."
        }), 500

    family_id = family["id"]

    password_hash = generate_password_hash(password)

    database_write("""
        INSERT INTO users (
            family_id,
            email,
            password_hash,
            first_name,
            last_name,
            role,
            is_active
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        family_id,
        email,
        password_hash,
        first_name,
        last_name,
        "admin",
        1
    ))

    user_row = database_read("""
        SELECT users.*, families.family_name
        FROM users
        JOIN families ON families.id = users.family_id
        WHERE users.email = ?
          AND users.family_id = ?
        LIMIT 1
    """, (email, family_id), one=True)

    if not user_row:
        return jsonify({
            "ok": False,
            "error": "Registration failed while loading user."
        }), 500

    token = create_jwt_token(user_row)

    return jsonify({
        "ok": True,
        "token": token,
        "user": {
            "id": user_row["id"],
            "family_id": user_row["family_id"],
            "email": user_row["email"],
            "first_name": user_row["first_name"],
            "last_name": user_row["last_name"],
            "family_name": user_row["family_name"],
            "role": user_row["role"]
        }
    }), 201

    
@api_auth_bp.route("/logout", methods=["POST"])
@api_login_required
def api_auth_logout():

    return jsonify({
        "ok": True,
        "message": "Logged out successfully."
    })        