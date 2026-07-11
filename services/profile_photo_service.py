import uuid
from pathlib import Path
from werkzeug.utils import secure_filename
from services.file_utils import allowed_file
from db_helpers import (
    database_read,
    database_write,
    get_family_profiles_dir
)
from services.embedding_service import create_profile_embedding_from_saved_photo



def save_member_profile_photo(
    family_id,
    member_id,
    file,
    DeepFace
):
    if not file or file.filename == "":
        return {
            "ok": False,
            "error": "Empty file."
        }

    if not allowed_file(file.filename):
        return {
            "ok": False,
            "error": "Invalid file type."
        }

    member = database_read("""
        SELECT *
        FROM family_members
        WHERE id = ?
          AND family_id = ?
    """, (member_id, family_id), one=True)

    if not member:
        return {
            "ok": False,
            "error": "Member not found."
        }

    profile_dir = get_family_profiles_dir(family_id)
    member_dir = profile_dir / str(member_id)
    member_dir.mkdir(parents=True, exist_ok=True)

    original = secure_filename(file.filename)
    ext = original.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4()}.{ext}"

    save_path = member_dir / filename

    file.save(save_path)

    db_file_path = str(save_path).replace("\\", "/")

    database_write("""
        INSERT INTO member_photos (family_id, member_id, file_path)
        VALUES (?, ?, ?)
    """, (family_id, member_id, db_file_path))

    photo_row = database_read("""
        SELECT id
        FROM member_photos
        WHERE family_id = ?
          AND member_id = ?
          AND file_path = ?
        ORDER BY id DESC
        LIMIT 1
    """, (family_id, member_id, db_file_path), one=True)

    if not photo_row:
        return {
            "ok": False,
            "error": "Photo saved but DB row not found."
        }

    learned = create_profile_embedding_from_saved_photo(
        family_id=family_id,
        member_id=member_id,
        photo_id=photo_row["id"],
        photo_path=str(save_path),
        DeepFace=DeepFace
    )

    return {
        "ok": True,
        "photo": {
            "id": photo_row["id"],
            "member_id": member_id,
            "file_path": db_file_path,
            "filename": filename,
            "learned": learned
        }
    }