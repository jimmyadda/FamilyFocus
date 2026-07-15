"""Business logic for Family Focus family members.

This module is intentionally independent of Flask request/session objects.
Every public operation receives ``family_id`` explicitly so it can be shared
by the browser, JWT API, Telegram, and future native app.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Any

from db_helpers import (
    database_read,
    database_write,
    get_family_profiles_dir,
)


class MemberServiceError(Exception):
    """Base exception for member service operations."""


class MemberNotFoundError(MemberServiceError):
    """Raised when a member is not owned by the requested family."""


class MemberValidationError(MemberServiceError):
    """Raised when member input is missing or invalid."""


class MemberAlreadyExistsError(MemberServiceError):
    """Raised when a family already has a member with the same name."""


def _normalize_name(name: Any) -> str:
    normalized = str(name or "").strip()

    if not normalized:
        raise MemberValidationError("Member name is required.")

    if len(normalized) > 100:
        raise MemberValidationError("Member name must be 100 characters or fewer.")

    return normalized


def list_members(family_id: int) -> list[dict[str, Any]]:
    """Return all members belonging to a family with profile-photo metadata."""
    return database_read(
        """
        SELECT
            fm.id,
            fm.family_id,
            fm.name,
            fm.created_at,
            COUNT(mp.id) AS photo_count,
            (
                SELECT mp2.file_path
                FROM member_photos mp2
                WHERE mp2.member_id = fm.id
                  AND mp2.family_id = fm.family_id
                ORDER BY mp2.id ASC
                LIMIT 1
            ) AS profile_photo
        FROM family_members fm
        LEFT JOIN member_photos mp
               ON mp.member_id = fm.id
              AND mp.family_id = fm.family_id
        WHERE fm.family_id = ?
        GROUP BY fm.id, fm.family_id, fm.name, fm.created_at
        ORDER BY fm.name COLLATE NOCASE
        """,
        (family_id,),
    )


def get_member(family_id: int, member_id: int) -> dict[str, Any] | None:
    """Return one family-owned member, or ``None`` when not found."""
    return database_read(
        """
        SELECT id, family_id, name, created_at, centroid, centroid_encrypted
        FROM family_members
        WHERE id = ?
          AND family_id = ?
        """,
        (member_id, family_id),
        one=True,
    )


def require_member(family_id: int, member_id: int) -> dict[str, Any]:
    """Return one owned member or raise ``MemberNotFoundError``."""
    member = get_member(family_id, member_id)

    if not member:
        raise MemberNotFoundError("Family member not found.")

    return member


def get_member_by_name(family_id: int, name: str) -> dict[str, Any] | None:
    normalized = _normalize_name(name)

    return database_read(
        """
        SELECT id, family_id, name, created_at, centroid, centroid_encrypted
        FROM family_members
        WHERE family_id = ?
          AND name = ? COLLATE NOCASE
        LIMIT 1
        """,
        (family_id, normalized),
        one=True,
    )


def get_member_id_by_name(family_id: int, name: str) -> int | None:
    member = get_member_by_name(family_id, name)
    return int(member["id"]) if member else None


def create_member(family_id: int, name: str) -> dict[str, Any]:
    """Create and return a new member for a family."""
    normalized = _normalize_name(name)

    if get_member_by_name(family_id, normalized):
        raise MemberAlreadyExistsError(
            "A family member with this name already exists."
        )

    try:
        member_id = database_write(
            """
            INSERT INTO family_members (family_id, name, created_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            (family_id, normalized),
        )
    except sqlite3.IntegrityError as error:
        raise MemberAlreadyExistsError(
            "A family member with this name already exists."
        ) from error

    return require_member(family_id, member_id)


def update_member(family_id: int, member_id: int, name: str) -> dict[str, Any]:
    """Rename a family-owned member and return the updated row."""
    require_member(family_id, member_id)
    normalized = _normalize_name(name)

    duplicate = get_member_by_name(family_id, normalized)
    if duplicate and int(duplicate["id"]) != int(member_id):
        raise MemberAlreadyExistsError(
            "A family member with this name already exists."
        )

    try:
        database_write(
            """
            UPDATE family_members
            SET name = ?
            WHERE id = ?
              AND family_id = ?
            """,
            (normalized, member_id, family_id),
        )
    except sqlite3.IntegrityError as error:
        raise MemberAlreadyExistsError(
            "A family member with this name already exists."
        ) from error

    return require_member(family_id, member_id)


def get_member_detail(family_id: int, member_id: int) -> dict[str, Any]:
    """Return a member plus profile photos and confirmed detection count."""
    member = require_member(family_id, member_id)

    photos = database_read(
        """
        SELECT id, family_id, member_id, file_path, created_at
        FROM member_photos
        WHERE member_id = ?
          AND family_id = ?
        ORDER BY created_at DESC, id DESC
        """,
        (member_id, family_id),
    )

    for photo in photos:
        photo["filename"] = Path(photo["file_path"]).name

    detected_row = database_read(
        """
        SELECT COUNT(DISTINCT photo_id) AS count
        FROM photo_detections
        WHERE family_id = ?
          AND member_id = ?
          AND status = 'confirmed'
        """,
        (family_id, member_id),
        one=True,
    )

    member["photos"] = photos
    member["photo_count"] = len(photos)
    member["detected_count"] = int((detected_row or {}).get("count", 0))
    return member


def delete_member(
    family_id: int,
    member_id: int,
    *,
    delete_profile_files: bool = True,
) -> dict[str, Any]:
    """Delete a member and member-owned learning data.

    Family album photos are retained. Their detections for this member are
    removed, preventing deletion of a person from deleting shared photos.
    """
    member = require_member(family_id, member_id)

    profile_rows = database_read(
        """
        SELECT id, file_path
        FROM member_photos
        WHERE family_id = ?
          AND member_id = ?
        """,
        (family_id, member_id),
    )

    photo_ids = [int(row["id"]) for row in profile_rows]

    database_write(
        "DELETE FROM photo_detections WHERE family_id = ? AND member_id = ?",
        (family_id, member_id),
    )
    database_write(
        """
        DELETE FROM learning_reviews
        WHERE family_id = ?
          AND (predicted_member_id = ? OR reviewed_member_id = ?)
        """,
        (family_id, member_id, member_id),
    )
    database_write(
        "DELETE FROM member_embeddings WHERE family_id = ? AND member_id = ?",
        (family_id, member_id),
    )
    database_write(
        "DELETE FROM member_photos WHERE family_id = ? AND member_id = ?",
        (family_id, member_id),
    )
    database_write(
        "DELETE FROM family_members WHERE family_id = ? AND id = ?",
        (family_id, member_id),
    )

    if delete_profile_files:
        for row in profile_rows:
            try:
                Path(row["file_path"]).unlink(missing_ok=True)
            except OSError as error:
                print("MEMBER DELETE: could not remove profile photo:", error)

        member_dir = get_family_profiles_dir(family_id) / str(member_id)
        try:
            if member_dir.exists():
                shutil.rmtree(member_dir)
        except OSError as error:
            print("MEMBER DELETE: could not remove member directory:", error)

    return {
        "id": member_id,
        "family_id": family_id,
        "name": member["name"],
        "deleted_profile_photo_ids": photo_ids,
    }
