from flask import Blueprint, jsonify, request

from db_helpers import (
    database_read,database_write
    
)
from services.member_service import (
    MemberAlreadyExistsError,
    MemberNotFoundError,
    MemberValidationError,
    create_member,
    delete_member,
    get_member_detail,
    list_members,
    update_member,
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
    family_id = get_api_family_id()
    members = list_members(family_id)

    return jsonify({
        "ok": True,
        "members": members
    })

@api_members_bp.route("/<int:member_id>", methods=["GET"])
@api_login_required
def api_get_member(member_id):    
    family_id = get_api_family_id()

    try:
        member = get_member_detail(family_id, member_id)
    except MemberNotFoundError as error:
        return jsonify({
            "ok": False,
            "error": str(error)
        }), 404

    return jsonify({
        "ok": True,
        "member": member
    })



@api_members_bp.route("", methods=["POST"])
@api_login_required
def api_create_member():
    family_id = get_api_family_id()
    data = request.get_json(silent=True) or {}

    try:
        member = create_member(
            family_id=family_id,
            name=data.get("name")
        )
    except MemberValidationError as error:
        return jsonify({"ok": False, "error": str(error)}), 400
    except MemberAlreadyExistsError as error:
        return jsonify({"ok": False, "error": str(error)}), 409

    return jsonify({
        "ok": True,
        "member": member
    }), 201   

@api_members_bp.route("/<int:member_id>", methods=["PUT"])
@api_login_required
def api_update_member(member_id):
    family_id = get_api_family_id()
    data = request.get_json(silent=True) or {}

    try:
        member = update_member(
            family_id=family_id,
            member_id=member_id,
            name=data.get("name")
        )
    except MemberNotFoundError as error:
        return jsonify({"ok": False, "error": str(error)}), 404
    except MemberValidationError as error:
        return jsonify({"ok": False, "error": str(error)}), 400
    except MemberAlreadyExistsError as error:
        return jsonify({"ok": False, "error": str(error)}), 409

    return jsonify({
        "ok": True,
        "member": member
    })

@api_members_bp.route("/<int:member_id>", methods=["DELETE"])
@api_login_required
def api_delete_member(member_id):
    family_id = get_api_family_id()

    try:
        deleted = delete_member(
            family_id=family_id,
            member_id=member_id
        )
    except MemberNotFoundError as error:
        return jsonify({"ok": False, "error": str(error)}), 404

    return jsonify({
        "ok": True,
        "deleted": deleted
    })