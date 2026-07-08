from flask import Blueprint, jsonify, request

from db_helpers import (
    database_read,database_write
    
)

from services.jwt_auth import api_login_required,get_api_family_id

api_members_bp = Blueprint(
    "api_members",
    __name__,
    url_prefix="/api/members"
)

@api_members_bp.route("", methods=["GET"])
@api_login_required
def api_get_members():

    family_id =  get_api_family_id()

    members = database_read("""
        SELECT
            fm.id,
            fm.name,

            (
                SELECT COUNT(*)
                FROM member_photos mp
                WHERE mp.member_id = fm.id
            ) AS profile_photos,

            (
                SELECT COUNT(DISTINCT pd.photo_id)
                FROM photo_detections pd
                WHERE pd.member_id = fm.id
                  AND pd.status='confirmed'
            ) AS detections

        FROM family_members fm
        WHERE fm.family_id = ?
        ORDER BY fm.name
    """, (family_id,))

    return jsonify({
        "ok": True,
        "members": members
    })

@api_members_bp.route("/<int:member_id>", methods=["GET"])
@api_login_required
def api_get_member(member_id):    
    family_id = get_api_family_id()

    member = database_read("""
        SELECT
            fm.id,
            fm.name,
            fm.created_at,

            (
                SELECT COUNT(*)
                FROM member_photos mp
                WHERE mp.member_id = fm.id
                  AND mp.family_id = fm.family_id
            ) AS profile_photos,

            (
                SELECT COUNT(*)
                FROM member_embeddings me
                WHERE me.member_id = fm.id
                  AND me.family_id = fm.family_id
            ) AS embeddings,

            (
                SELECT COUNT(DISTINCT pd.photo_id)
                FROM photo_detections pd
                WHERE pd.member_id = fm.id
                  AND pd.family_id = fm.family_id
                  AND pd.status = 'confirmed'
            ) AS detections

        FROM family_members fm
        WHERE fm.id = ?
          AND fm.family_id = ?
        LIMIT 1
    """, (member_id, family_id), one=True)

    if not member:
        return jsonify({
            "ok": False,
            "error": "Member not found"
        }), 404

    return jsonify({
        "ok": True,
        "member": member
    })    

@api_members_bp.route("", methods=["POST"])
@api_login_required
def api_create_member():

    family_id = request.family_id

    data = request.get_json(silent=True) or {}

    name = data.get("name", "").strip()

    if not name:
        return jsonify({
            "ok": False,
            "error": "Member name is required."
        }), 400

    existing = database_read("""
        SELECT id
        FROM family_members
        WHERE family_id = ?
          AND lower(name) = lower(?)
    """, (family_id, name), one=True)

    if existing:
        return jsonify({
            "ok": False,
            "error": "Member already exists."
        }), 409

    database_write("""
        INSERT INTO family_members (
            family_id,
            name,
            created_at
        )
        VALUES (?, ?, datetime('now'))
    """, (family_id, name))

    member = database_read("""
        SELECT *
        FROM family_members
        WHERE family_id = ?
          AND lower(name) = lower(?)
        ORDER BY id DESC
        LIMIT 1
    """, (family_id, name), one=True)

    return jsonify({
        "ok": True,
        "member": member
    }), 201    

@api_members_bp.route("/<int:member_id>", methods=["PUT"])
@api_login_required
def api_update_member(member_id):

    family_id = request.family_id

    data = request.get_json(silent=True) or {}

    name = data.get("name", "").strip()

    if not name:
        return jsonify({
            "ok": False,
            "error": "Member name is required."
        }), 400

    # Verify member exists and belongs to current family
    member = database_read("""
        SELECT *
        FROM family_members
        WHERE id = ?
          AND family_id = ?
    """, (member_id, family_id), one=True)

    if not member:
        return jsonify({
            "ok": False,
            "error": "Member not found."
        }), 404

    # Prevent duplicate names (excluding this member)
    existing = database_read("""
        SELECT id
        FROM family_members
        WHERE family_id = ?
          AND lower(name) = lower(?)
          AND id <> ?
    """, (
        family_id,
        name,
        member_id
    ), one=True)

    if existing:
        return jsonify({
            "ok": False,
            "error": "A family member with this name already exists."
        }), 409

    database_write("""
        UPDATE family_members
        SET name = ?
        WHERE id = ?
          AND family_id = ?
    """, (
        name,
        member_id,
        family_id
    ))

    member = database_read("""
        SELECT *
        FROM family_members
        WHERE id = ?
          AND family_id = ?
    """, (
        member_id,
        family_id
    ), one=True)

    return jsonify({
        "ok": True,
        "member": member
    })

@api_members_bp.route("/<int:member_id>", methods=["DELETE"])
@api_login_required
def api_delete_member(member_id):

    family_id = request.family_id

    member = database_read("""
        SELECT *
        FROM family_members
        WHERE id = ?
          AND family_id = ?
    """, (member_id, family_id), one=True)

    if not member:
        return jsonify({
            "ok": False,
            "error": "Member not found."
        }), 404

    database_write("""
        DELETE FROM family_members
        WHERE id = ?
          AND family_id = ?
    """, (member_id, family_id))

    return jsonify({
        "ok": True,
        "message": "Member deleted."
    })     