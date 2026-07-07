import os, re, uuid, mimetypes
from pathlib import Path
from itsdangerous import URLSafeTimedSerializer
from flask import current_app, abort
from PIL import Image, ImageOps
import filetype
from cryptography.fernet import Fernet
import json
import jwt
from datetime import datetime, timedelta, timezone
from functools import wraps


SAFE_FILENAME_RE = re.compile(r"^[a-f0-9]{32}\.(jpg|jpeg|png|webp)$", re.I)
MAX_UPLOAD_SIZE = 25 * 1024 * 1024  # 25MB

ALLOWED_MIME = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def validate_file_size(file_storage):
    file_storage.seek(0, os.SEEK_END)
    size = file_storage.tell()
    file_storage.seek(0)

    if size > MAX_UPLOAD_SIZE:
        abort(413, "File too large")


def detect_image_type(file_storage):
    head = file_storage.read(261)
    file_storage.seek(0)

    kind = filetype.guess(head)
    if not kind or kind.mime not in ALLOWED_MIME:
        abort(400, "Invalid image type")

    return kind.mime, ALLOWED_MIME[kind.mime]


def make_uuid_filename(extension):
    return f"{uuid.uuid4().hex}{extension}"


def safe_resolve(base_dir, filename):
    if not SAFE_FILENAME_RE.match(filename):
        abort(400, "Invalid filename")

    base = Path(base_dir).resolve()
    final_path = (base / filename).resolve()

    if base not in final_path.parents:
        abort(403)

    return final_path


def strip_exif_and_save(upload_file, dest_path):
    image = Image.open(upload_file)
    image = ImageOps.exif_transpose(image)

    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    save_kwargs = {}

    if dest_path.suffix.lower() in [".jpg", ".jpeg"]:
        image = image.convert("RGB")
        save_kwargs = {"quality": 92, "optimize": True}

    image.save(dest_path, **save_kwargs)


def get_serializer():
    return URLSafeTimedSerializer(current_app.config["SECRET_KEY"])


def sign_photo_token(filename, family_id):
    return get_serializer().dumps({
        "filename": filename,
        "family_id": family_id,
    })


def verify_photo_token(token, max_age=300):
    try:
        return get_serializer().loads(token, max_age=max_age)
    except Exception:
        abort(403)


def get_fernet():
    return Fernet(current_app.config["EMBEDDING_ENCRYPTION_KEY"])


def encrypt_embedding(embedding):
    raw = json.dumps(embedding).encode("utf-8")
    return get_fernet().encrypt(raw).decode("utf-8")


def decrypt_embedding(encrypted_embedding):
    raw = get_fernet().decrypt(encrypted_embedding.encode("utf-8"))
    return json.loads(raw.decode("utf-8"))

def safe_delete_file(path):
    try:
        if path and path.exists() and path.is_file():
            path.unlink()
            return True
    except Exception as e:
        print("DELETE FILE ERROR:", e)

    return False    

def sign_telegram_album_token(family_id, telegram_chat_id):
    return get_serializer().dumps({
        "family_id": family_id,
        "telegram_chat_id": str(telegram_chat_id),
        "type": "telegram_album"
    })


def verify_telegram_album_token(token, max_age=600):
    data = get_serializer().loads(token, max_age=max_age)

    if data.get("type") != "telegram_album":
        abort(403)

    return data

