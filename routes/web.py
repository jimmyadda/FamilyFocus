import base64
import json
import shutil
import sqlite3
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Blueprint, abort, current_app, flash, get_flashed_messages, jsonify,
    redirect, render_template, request, send_file, send_from_directory,
    session, url_for,
)
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from auth import get_current_family_id, login_required
from config import (
    CAMERA_MATCH_THRESHOLD, MIN_FACE_SIZE_CAMERA, MIN_FACE_SIZE_IMAGE,
    PHOTO_MATCH_THRESHOLD, TEMP_DIR,
)
from db_helpers import (
    create_family_photo, create_photo_detection, database_read, database_write,
    get_confirmed_family_photos, get_current_user, get_family_base_dir,
    get_family_members_with_detections, get_family_possible_dir,
    get_family_possible_faces_dir, get_family_profiles_dir,
    get_family_results_dir, get_family_skipped_dir, get_member_detected_album,
    get_owned_detection, get_owned_member, get_owned_photo, get_photo_people,
    get_possible_family_photos, save_learning_review, extract_faces_for_review,
)
from face_utils import crop_face, detect_faces_with_opencv, identify_face, represent_face
from security_utils import decrypt_embedding, encrypt_embedding
from services.detection_service import image_contains_known_family, scan_one_family_photo
from services.embedding_service import create_profile_embedding_from_saved_photo
from services.file_utils import allowed_file

web_bp = Blueprint("web", __name__)


def _deepface():
    return current_app.config.get("DeepFace")


class _DeepFaceProxy:
    def __getattr__(self, name):
        deepface = _deepface()
        if deepface is None:
            raise RuntimeError("DeepFace is not installed.")
        return getattr(deepface, name)


DeepFace = _DeepFaceProxy()

def decode_base64_to_temp_file(data_url):
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url

    try:
        image_bytes = base64.b64decode(encoded)
    except Exception:
        return None

    np_arr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if bgr is None:
        return None

    temp_path = TEMP_DIR / f"{uuid.uuid4()}.jpg"
    cv2.imwrite(str(temp_path), bgr)

    return temp_path

def get_all_family_embeddings():
    family_id = get_current_family_id()

    rows = database_read("""
        SELECT 
            me.id,
            me.member_id,
            me.embedding,
            me.embedding_encrypted,
            fm.name
        FROM member_embeddings me
        JOIN family_members fm 
            ON fm.id = me.member_id
        WHERE me.family_id = ?
          AND fm.family_id = ?
    """, (family_id, family_id))

    print("LOADED EMBEDDINGS:", len(rows), "FAMILY:", family_id)

    family_embeddings = []

    for row in rows:
        if row["embedding_encrypted"]:
            embedding = decrypt_embedding(row["embedding_encrypted"])
        else:
            embedding = json.loads(row["embedding"])  # old records fallback

        family_embeddings.append({
            "member_id": row["member_id"],
            "name": row["name"],
            "embedding": embedding
        })

    return family_embeddings

def get_image_serializer():
    return URLSafeTimedSerializer(current_app.config["SECRET_KEY"])

def create_photo_token(photo_id, family_id):
    serializer = get_image_serializer()
    return serializer.dumps({
        "photo_id": photo_id,
        "family_id": family_id
    })

def verify_photo_token(photo_id, family_id, max_age=600):
    token = request.args.get("token")

    if not token:
        abort(403)

    serializer = get_image_serializer()

    try:
        data = serializer.loads(token, max_age=max_age)
    except SignatureExpired:
        abort(403)
    except BadSignature:
        abort(403)

    if data.get("photo_id") != photo_id:
        abort(403)

    if data.get("family_id") != family_id:
        abort(403)

    return True

def get_all_members():
    family_id = get_current_family_id()

    return database_read("""
        SELECT
            fm.id,
            fm.family_id,
            fm.name,
            fm.created_at,

            COUNT(mp.id) AS photo_count,

            (
                SELECT file_path
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

        GROUP BY fm.id

        ORDER BY fm.name
    """, (family_id,))

def get_member(member_id):
    family_id = get_current_family_id()

    rows = database_read("""
        SELECT *
        FROM family_members
        WHERE id = ?
        AND family_id = ?
    """, (
        member_id,
        family_id
    ))

    return rows[0] if rows else None

def get_member_by_name(name):
    family_id = get_current_family_id()

    rows = database_read("""
        SELECT *
        FROM family_members
        WHERE name = ?
        AND family_id = ?
    """, (
        name,
        family_id
    ))

    return rows[0] if rows else None

def get_member_id_by_name(name, family_id):
    rows = database_read(
        """
        SELECT id
        FROM family_members
        WHERE name = ?
          AND family_id = ?
        """,
        (name, family_id)
    )

    if not rows:
        return None

    return rows[0]["id"]

def save_accepted_face_embedding(member_name, face_crop_path, DeepFace, original_photo_path=None):
    family_id = get_current_family_id()
    member = get_member_by_name(member_name)

    if not member:
        print("Member not found:", member_name)
        return False

    member_id = member["id"]

    embedding = represent_face(face_crop_path, _deepface())

    if embedding is None:
        print("LEARN: represent_face returned None")
        print("LEARN face_crop_path:", face_crop_path)
        print("LEARN exists:", Path(face_crop_path).exists())
        return False

    current_embeddings = get_all_family_embeddings()

    check_name, check_distance, check_status = identify_face(
        face_crop_path,
        current_embeddings,
        DeepFace,
        threshold=PHOTO_MATCH_THRESHOLD
    )

    print(
        f"LEARNING CHECK | "
        f"name={check_name} "
        f"distance={check_distance:.4f} "
        f"status={check_status}"
    )



    if check_distance > 0.55:
        print(
            f"Manual learning warning: face is far from existing profile "
            f"({check_distance:.4f}), but user selected {member_name}. Learning anyway."
        )

    profiles_dir = get_family_profiles_dir(family_id)
    member_profile_dir = profiles_dir / str(member_id)
    member_profile_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(face_crop_path).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"

    profile_filename = f"{uuid.uuid4()}{ext}"
    profile_path = member_profile_dir / profile_filename
    
    source_photo_path = original_photo_path if original_photo_path else face_crop_path
    shutil.copy(str(source_photo_path), str(profile_path))

    database_write(
        """
        INSERT INTO member_photos (family_id, member_id, file_path)
        VALUES (?, ?, ?)
        """,
        (family_id, member_id, str(profile_path))
    )

    photo_row = database_read(
        """
        SELECT id
        FROM member_photos
        WHERE family_id = ?
          AND member_id = ?
          AND file_path = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (family_id, member_id, str(profile_path)),
        one=True
    )

    photo_id = photo_row["id"]

    encrypted_embedding = encrypt_embedding(embedding)

    database_write(
        """
        INSERT INTO member_embeddings (
            family_id,
            member_id,
            photo_id,
            embedding,
            embedding_encrypted
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            family_id,
            member_id,
            photo_id,
            None,
            encrypted_embedding
        )
    )

    print(f"LEARN: Learning completed for {member_name}")
    return True

@web_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        family_name = request.form.get("family_name", "").strip()
        first_name = request.form.get("first_name", "").strip()
        last_name = request.form.get("last_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        client_id = family_name.lower().replace(" ", "_")   

        if not family_name or not email or not password:
            flash("Family name, email and password are required.")
            return redirect(url_for("register"))

        existing = database_read(
            "SELECT id FROM users WHERE email = ?",
            (email,)
        )

        if existing:
            flash("Email already exists. Please log in.")
            return redirect(url_for("login"))
        
        database_write("""
            INSERT INTO families (
                family_name,
                client_id
            )
            VALUES (?, ?)
        """, (
            family_name,
            client_id
        ))

        family_id = database_read("SELECT last_insert_rowid() AS id")[0]["id"]

        password_hash = generate_password_hash(password)
        client_id = family_name.lower().replace(" ", "_")
        
        database_write("""
            INSERT INTO users (
                family_id,
                email,
                password_hash,
                first_name,
                last_name,
                role
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            family_id,
            email,
            password_hash,
            first_name,
            last_name,
            "admin"
        ))

        user_id = database_read("SELECT last_insert_rowid() AS id")[0]["id"]

        session["user_id"] = user_id
        session["family_id"] = family_id
        session["email"] = email
        session["first_name"] = first_name
        session["family_name"] = family_name

        flash("Family account created.")
        return redirect(url_for("home"))

    return render_template("register.html")

@web_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        user_rows = database_read("""
            SELECT users.*, families.family_name
            FROM users
            JOIN families ON families.id = users.family_id
            WHERE users.email = ?
            AND users.is_active = 1
        """, (email,))

        if not user_rows:
            flash("Invalid email or password.")
            return redirect(url_for("login"))

        user = user_rows[0]

        if not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.")
            return redirect(url_for("login"))

        session["user_id"] = user["id"]
        session["family_id"] = user["family_id"]
        session["email"] = user["email"]
        session["first_name"] = user["first_name"]
        session["family_name"] = user["family_name"]

        database_write(
            "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
            (user["id"],)
        )

        flash("Logged in successfully.")
        return redirect(url_for("home"))

    return render_template("login.html")

@web_bp.route("/logout")
def logout():
    session_token = session.get("session_token")
    if session_token:
        database_write(
            "DELETE FROM user_sessions WHERE token = ?",
            (session_token,)
        )
    session.pop("user_id", None)
    session.pop("family_id", None)
    session.pop("family_name", None)
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))    

@web_bp.route("/")
@login_required
def home():
    family_id = get_current_family_id()

    stats = {
        "members": database_read("""
            SELECT COUNT(*) AS count
            FROM family_members
            WHERE family_id = ?
        """, (family_id,))[0]["count"],

        "detected": database_read("""
            SELECT COUNT(DISTINCT photo_id) AS count
            FROM photo_detections
            WHERE family_id = ?
            AND status = 'confirmed'
        """, (family_id,))[0]["count"],

        "possible": database_read("""
            SELECT COUNT(*) AS count
            FROM photo_detections
            WHERE family_id = ?
            AND status = 'possible'
        """, (family_id,))[0]["count"],
    }

    members = database_read("""
        SELECT *
        FROM family_members
        WHERE family_id = ?
        ORDER BY name
    """, (family_id,))

    recent_photos = database_read("""
        SELECT DISTINCT fp.*
        FROM family_photos fp
        JOIN photo_detections pd ON pd.photo_id = fp.id
        WHERE fp.family_id = ?
        AND pd.family_id = ?
        AND pd.status = 'confirmed'
        ORDER BY fp.created_at DESC
        LIMIT 8
    """, (family_id, family_id))
    #onboarding: if no photos, show recent uploads
    members_count = database_read(
    "SELECT COUNT(*) AS count FROM family_members WHERE family_id = ?",
        (family_id,), one=True)["count"]

    profile_photos_count = database_read(
        """
        SELECT COUNT(*) AS count
        FROM member_photos mp
        JOIN family_members fm ON fm.id = mp.member_id
        WHERE fm.family_id = ?
        """,
        (family_id,),
        one=True
    )["count"]

    embeddings_count = database_read(
        """
        SELECT COUNT(*) AS count
        FROM member_embeddings me
        JOIN family_members fm ON fm.id = me.member_id
        WHERE fm.family_id = ?
        """,
        (family_id,),
        one=True
    )["count"]

    show_onboarding = (
        members_count == 0
        or profile_photos_count == 0
        or embeddings_count == 0
    )
    members_count = database_read(
        "SELECT COUNT(*) AS c FROM family_members WHERE family_id = ?",
        (family_id,),
        one=True
    )["c"]

    profile_photos_count = database_read(
        """
        SELECT COUNT(*) AS c
        FROM member_photos mp
        JOIN family_members fm ON fm.id = mp.member_id
        WHERE fm.family_id = ?
        """,
        (family_id,),
        one=True
    )["c"]

    detected_count = database_read(
        "SELECT COUNT(*) AS c FROM photo_detections WHERE family_id = ?",
        (family_id,),
        one=True
    )["c"]

    if members_count == 0:
        current_step = 1
    elif profile_photos_count == 0:
        current_step = 2
    elif detected_count == 0:
        current_step = 4
    else:
        current_step = 5


    user = get_current_user()
    return render_template(
        "dashboard.html",
        stats=stats,
        user=user,
        members=members,
        recent_photos=recent_photos,
        show_onboarding=show_onboarding,
        current_step=current_step
    )

@web_bp.route("/members/create", methods=["POST"])
@login_required
def create_member():
    family_id = get_current_family_id()
    name = request.form.get("name", "").strip()

    if name:
        try:
            database_write(
                """
                INSERT INTO family_members (family_id, name, created_at)
                VALUES (?, ?, datetime('now'))
                """,
                (family_id, name)
            )
        except sqlite3.IntegrityError:
            pass

    return redirect(url_for("home"))

@web_bp.route("/members/<int:member_id>")
@web_bp.route("/member/<int:member_id>")
@login_required
def member_page(member_id):
    family_id = get_current_family_id()
    member = get_owned_member(member_id)

    rows = database_read(
        "SELECT * FROM family_members WHERE id = ?",
        (member_id,)
    )

    member = rows[0] if rows else None

    if not member:
        return "Family member not found", 404

    photos = database_read(
        """
        SELECT *
        FROM member_photos
        WHERE member_id = ?
        AND family_id = ?
        ORDER BY created_at DESC
        """,
        (member_id, family_id)
    )

    for photo in photos:
        photo["filename"] = Path(photo["file_path"]).name

    detected_count_row = database_read(
        """
        SELECT COUNT(DISTINCT photo_id) AS count
        FROM photo_detections
        WHERE family_id = ?
        AND member_id = ?
        AND status = 'confirmed'
        """,
        (family_id, member_id)
    )

    detected_count = detected_count_row[0]["count"]

    return render_template(
        "member.html",
        member=member,
        photos=photos,
        detected_count=detected_count,
        deepface_available=_deepface() is not None
    )

@web_bp.route("/member-by-name/<member_name>/album")
@login_required
def member_album_by_name(member_name):
    family_id = get_current_family_id()
    member_id = get_member_id_by_name(member_name, family_id)

    if not member_id:
        flash("Family member not found.")
        return redirect(url_for("family_album"))

    return redirect(url_for("member_album", member_id=member_id))

@web_bp.route("/album")
@login_required
def family_album():
    family_id = get_current_family_id()

    selected_member_id = request.args.get("member_id", type=int)

    members = get_family_members_with_detections(family_id)

    if selected_member_id:
        confirmed_photos = get_member_detected_album(
            family_id,
            selected_member_id
        )
    else:
        confirmed_photos = get_confirmed_family_photos(family_id)

    for photo in confirmed_photos:
        photo["people"] = get_photo_people(family_id, photo["id"])
        photo["token"] = create_photo_token(photo["id"], family_id)

    possible_photos = get_possible_family_photos(family_id)

    for photo in possible_photos:
        photo["token"] = create_photo_token(photo["id"], family_id)

    return render_template(
        "album.html",
        confirmed_photos=confirmed_photos,
        possible_photos=possible_photos,
        members=members,
        selected_member_id=selected_member_id
    )

@web_bp.route("/member/<int:member_id>/album")
@login_required
def member_album(member_id):
    family_id = get_current_family_id()
    memberown = get_owned_member(member_id)
    member = database_read(
        """
        SELECT *
        FROM family_members
        WHERE id = ?
          AND family_id = ?
        """,
        (member_id, family_id)
    )

    if not member:
        flash("Member not found.")
        return redirect(url_for("index"))

    photos = get_member_detected_album(family_id, member_id)

    return render_template(
        "member_album.html",
        member=member[0],
        photos=photos
    )

@web_bp.route("/detections/<int:detection_id>/confirm", methods=["POST"])
@login_required
def confirm_detection(detection_id):
    family_id = get_current_family_id()
    detection = get_owned_detection(detection_id)
    detection = database_read(
        """
        SELECT *
        FROM photo_detections
        WHERE id = ?
          AND family_id = ?
        """,
        (detection_id, family_id)
    )

    if not detection:
        return {"success": False, "message": "Detection not found"}, 404

    database_write(
        """
        UPDATE photo_detections
        SET status = 'confirmed',
            confirmed_by_user = 1
        WHERE id = ?
          AND family_id = ?
        """,
        (detection_id, family_id)
    )

    return {"success": True}

@web_bp.route("/detections/<int:detection_id>/reject", methods=["POST"])
@login_required
def reject_detection(detection_id):
    family_id = get_current_family_id()
    detection = get_owned_detection(detection_id)
    database_write(
        """
        UPDATE photo_detections
        SET status = 'rejected'
        WHERE id = ?
          AND family_id = ?
        """,
        (detection_id, family_id)
    )

    return {"success": True}

@web_bp.route("/member/<int:member_id>/upload", methods=["POST"])
@web_bp.route("/members/<int:member_id>/upload", methods=["POST"])
@login_required
def upload_member_photos(member_id):
    family_id = get_current_family_id()
    profile_dir = get_family_profiles_dir(family_id)
    try:
        files = request.files.getlist("photos")
        print("FILES RECEIVED:", len(files))

        for file in files:
            print("FILE:", file.filename, file.content_type)

    except Exception as e:
        print("ERROR READING FILES:", e)
        flash("Upload failed while reading files.")
        return redirect(url_for("member_page", member_id=member_id))

    rows = database_read(
        "SELECT * FROM family_members WHERE id = ?",
        (member_id,)
    )

    if not rows:
        return "Family member not found", 404

    if not files:
        flash("No photos selected.")
        return redirect(url_for("member_page", member_id=member_id))

    member_dir = profile_dir / str(member_id)
    member_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0

    for file in files:
        if not file or file.filename == "":
            skipped += 1
            continue

        if not allowed_file(file.filename):
            print("SKIPPED NOT ALLOWED:", file.filename)
            skipped += 1
            continue

        original = secure_filename(file.filename)
        ext = original.rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4()}.{ext}"

        save_path = member_dir / filename
        
        file.save(save_path)

        family_id = get_current_family_id()

        database_write(
            """
            INSERT INTO member_photos (family_id, member_id, file_path)
            VALUES (?, ?, ?)
            """,
            (family_id, member_id, str(save_path).replace("\\", "/"))
        )

        photo_row = database_read(
            """
            SELECT id
            FROM member_photos
            WHERE family_id = ?
            AND member_id = ?
            AND file_path = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (family_id, member_id, str(save_path).replace("\\", "/")),
            one=True
        )

        if photo_row:
            learned = create_profile_embedding_from_saved_photo(
                family_id=family_id,
                member_id=member_id,
                photo_id=photo_row["id"],
                photo_path=str(save_path),
                DeepFace=_deepface()
            )
            print("WEB PROFILE LEARNED:", learned)

        saved += 1

    flash(f"Uploaded {saved} photos. Skipped {skipped}.")
    return redirect(url_for("member_page", member_id=member_id))

@web_bp.route("/member-photo/<int:photo_id>/delete", methods=["POST"])
@login_required
def delete_member_photo(photo_id):

    photo = database_read(
        """
        SELECT *
        FROM member_photos
        WHERE id = ?
        """,
        (photo_id,)
    )

    if not photo:
        flash("Photo not found.")
        return redirect(request.referrer)

    photo = photo[0]

    try:
        Path(photo["file_path"]).unlink(missing_ok=True)
    except Exception:
        pass

    database_write(
        """
        DELETE FROM member_photos
        WHERE id = ?
        """,
        (photo_id,)
    )

    database_write(
        """
        DELETE FROM member_embeddings
        WHERE photo_id = ?
        """,
        (photo_id,)
    )

    flash("Profile photo deleted.")

    return redirect(request.referrer)

@web_bp.route("/profile-file/<int:member_id>/<filename>")
@login_required
def serve_profile_file(member_id, filename):

    # Verify member belongs to current family
    member = get_owned_member(member_id)

    safe_filename = secure_filename(filename)

    if safe_filename != filename:
        abort(403)

    family_id = member["family_id"]

    profile_dir = get_family_profiles_dir(family_id)
    folder = profile_dir / str(member_id)

    file_path = folder / safe_filename

    if not file_path.exists():
        abort(404)

    return send_from_directory(folder, safe_filename)

@web_bp.app_errorhandler(413)
def too_large(e):
    print("413 TOO LARGE:", request.content_length)
    flash("Upload too large. Try fewer photos.")
    return redirect(request.referrer or url_for("index"))

@web_bp.route("/member/<int:member_id>/rebuild-embeddings", methods=["POST"])
@login_required
def rebuild_embeddings(member_id):
    family_id = get_current_family_id()
    profile_dir = get_family_profiles_dir(family_id)
    if _deepface() is None:
        flash("DeepFace is not installed.")
        return redirect(url_for("member_page", member_id=member_id))

    photos = database_read(
        "SELECT * FROM member_photos WHERE member_id = ?",
        (member_id,)
    )

    if not photos:
        flash("No photos uploaded for this member yet.")
        return redirect(url_for("member_page", member_id=member_id))

    database_write(
        "DELETE FROM member_embeddings WHERE member_id = ?",
        (member_id,)
    )

    created = 0
    failed = 0
    accepted = database_read("""
    SELECT COUNT(*) AS total
    FROM learning_reviews
    WHERE member_id=? AND family_id = ? AND action='accepted'
    """, (member_id,family_id))[0]["total"]

    rejected = database_read("""
    SELECT COUNT(*) AS total
    FROM learning_reviews
    WHERE member_id=? AND family_id = ? AND action='rejected'
    """, (member_id,family_id))[0]["total"]
    for photo in photos:
        photo_path = Path(photo["file_path"])
        
        if not photo_path.exists():
            photo_path = profile_dir / str(member_id) / Path(photo["file_path"]).name


        if not photo_path.exists():
            failed += 1
            continue

        try:
            result = DeepFace.represent(
                img_path=str(photo_path),
                model_name="Facenet512",
                detector_backend="retinaface",
                enforce_detection=True
            )

            if not result:
                failed += 1
                continue

            embedding = result[0]["embedding"]
            encrypted_embedding = encrypt_embedding(embedding)
            database_write(
                """
                INSERT INTO member_embeddings (
                    family_id,
                    member_id,
                    photo_id,
                    embedding,
                    embedding_encrypted
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    family_id,
                    member_id,
                    photo["id"],
                    encrypted_embedding,
                    None
                )
            )

            created += 1

        except Exception as error:
            print("Embedding error:")
            print("PHOTO:", photo_path)
            print("ERROR:", repr(error))
            failed += 1

    flash(f"Face profile rebuilt. Created {created} embeddings. Failed {failed}.")
    return redirect(url_for("member_page", member_id=member_id,accepted_count=accepted,
    rejected_count=rejected))

@web_bp.route("/camera")
@login_required
def camera():
    return render_template(
        "camera.html",
        deepface_available=_deepface() is not None
    )

@web_bp.route("/api/recognize-frame", methods=["POST"])
@login_required
def recognize_frame():
    if _deepface() is None:
        return jsonify({
            "ok": False,
            "error": "DeepFace is not installed.",
            "matches": []
        }), 500

    data = request.get_json(silent=True) or {}
    image_data = data.get("image")

    if not image_data:
        return jsonify({
            "ok": False,
            "error": "Missing image.",
            "matches": []
        }), 400

    family_embeddings = get_all_family_embeddings()

    if not family_embeddings:
        return jsonify({
            "ok": False,
            "error": "No face profiles found. Rebuild profiles first.",
            "matches": []
        }), 400

    frame_path = decode_base64_to_temp_file(image_data)

    if frame_path is None:
        return jsonify({
            "ok": False,
            "error": "Could not decode camera frame.",
            "matches": []
        }), 400

    try:
        face_boxes = detect_faces_with_opencv(frame_path)
        matches = []

        for box in face_boxes:
            if box["width"] < MIN_FACE_SIZE_CAMERA or box["height"] < MIN_FACE_SIZE_CAMERA:
                matches.append({
                    "match": False,
                    "name": "Unknown",
                    "distance": 999,
                    "confidence": 0,
                    "box": box
                })
                continue

            face_path = crop_face(frame_path, box)

            if face_path is None:
                continue

            try:
                name, distance = identify_face(
                    face_path,
                    family_embeddings,
                    DeepFace,
                    threshold=CAMERA_MATCH_THRESHOLD
                )

                is_match = name is not None
                confidence = max(0, min(1, 1 - distance))

                matches.append({
                    "match": is_match,
                    "name": name if is_match else "Unknown",
                    "distance": round(float(distance), 4),
                    "confidence": round(float(confidence), 4),
                    "box": box
                })

            except Exception as error:
                print("Recognition error:", error)

                matches.append({
                    "match": False,
                    "name": "Unknown",
                    "distance": 999,
                    "confidence": 0,
                    "box": box
                })

            finally:
                try:
                    face_path.unlink()
                except Exception:
                    pass

        return jsonify({
            "ok": True,
            "faces_found": len(face_boxes),
            "matches": matches
        })

    finally:
        try:
            frame_path.unlink()
        except Exception:
            pass

@web_bp.route("/family-file/<path:filename>")
def family_file(filename):
    family_id = get_current_family_id()
    base_dir = get_family_base_dir(family_id).resolve()
    file_path = (base_dir / filename).resolve()
    if base_dir not in file_path.parents and file_path != base_dir:
        abort(403)
    if not file_path.exists():
        abort(404)

    return send_file(file_path)

@web_bp.route("/filter-photos")
@login_required
def filter_photos_page():
    return render_template("filter_photos.html")

@web_bp.route("/filter-photos", methods=["POST"])
@login_required
def filter_photos_upload():
    files = request.files.getlist("photos")

    if not files:
        flash("No photos selected.")
        return redirect(url_for("filter_photos_page"))

    kept = []
    possible = []
    skipped = 0

    family_id = get_current_family_id()
    family_embeddings = get_all_family_embeddings()

    print("FAMILY EMBEDDINGS LOADED:", len(family_embeddings))

    for file in files:
        result = scan_one_family_photo(
            file,
            family_id,
            family_embeddings,
            DeepFace,
            path_builder=lambda kind, name: url_for(
                "serve_result_file" if kind == "result" else "serve_possible_file",
                filename=name,
            ),
        )
        print(
            f"SCAN RESULT | "
            f"status={result['status']} "
            f"file={file.filename}"
        )

        if result["status"] == "matched":
            kept.append(result)

        elif result["status"] == "possible":
            possible.append(result)

        else:
            skipped += 1

    get_flashed_messages()

    return render_template(
        "filter_results.html",
        kept=kept,
        possible=possible,
        skipped=skipped
    )

@web_bp.route("/results/<filename>")
@login_required
def serve_result_file(filename):

    family_id = get_current_family_id()
    results_dir = get_family_results_dir(family_id)

    safe_filename = secure_filename(filename)

    if safe_filename != filename:
        abort(403)

    file_path = results_dir / safe_filename

    if not file_path.exists():
        abort(404)

    return send_from_directory(results_dir, safe_filename)

@web_bp.route("/serve-skipped/<path:filename>")
@login_required
def serve_skipped_file(filename):
    family_id = get_current_family_id()
    
    skipped_dir = get_family_skipped_dir(family_id)
    filename = filename.replace("skipped/", "")
    # Path traversal protection
    safe_filename = secure_filename(filename)
    if safe_filename != filename:
        abort(403)
    return send_from_directory(
        skipped_dir,
        filename
    )

@web_bp.route("/possible/<filename>")
@login_required
def serve_possible_file(filename):
    family_id = get_current_family_id()
    possible_dir = get_family_possible_dir(family_id)

    safe_filename = secure_filename(filename)

    if safe_filename != filename:
        abort(403)

    file_path = possible_dir / safe_filename

    if not file_path.exists():
        abort(404)

    return send_from_directory(possible_dir, safe_filename)

@web_bp.route("/review-possible/<filename>/<action>")
@login_required
def review_possible_photo(filename, action):
    member_name = request.args.get("member_name")
    face_crop = request.args.get("face_crop")
    family_id = get_current_family_id()
    results_dir = get_family_results_dir(family_id)
    possible_faces_dir = get_family_possible_faces_dir(family_id)
    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

    possible_path = results_dir  / filename
    result_path = results_dir  / filename
    if face_crop:
        safe_face_crop = secure_filename(face_crop)

        if safe_face_crop != face_crop:
            abort(403)

        face_crop_path = possible_faces_dir / safe_face_crop
    else:
        face_crop_path = None
    if not possible_path.exists():
        message = "Possible photo not found."

        if is_ajax:
            return jsonify({
                "ok": False,
                "message": message
            })

        flash(message)
        return redirect(url_for("filter_photos_page"))

    member_id = None

    if member_name:
        member = get_member_by_name(member_name)
        if member:
            member_id = member["id"]

    if action == "yes":
        if member_name and face_crop_path and face_crop_path.exists():
            save_accepted_face_embedding(
                member_name,
                str(face_crop_path),
                DeepFace
            )

        if member_id:
            save_learning_review(
                family_id=family_id,
                photo_path=possible_path,
                face_crop_path=face_crop_path,
                predicted_member_id=member_id,
                reviewed_member_id=member_id,
                distance=None,
                action="accepted"
            )

        shutil.move(str(possible_path), str(result_path))

        message = f"Accepted and learned as {member_name}."

    elif action == "no":
        if member_id:
            save_learning_review(
                family_id=family_id,
                photo_path=possible_path,
                face_crop_path=face_crop_path,
                predicted_member_id=member_id,
                reviewed_member_id=None,
                distance=None,
                action="rejected"
            )

        possible_path.unlink()

        message = "Possible match rejected."

    else:
        message = "Invalid action."

        if is_ajax:
            return jsonify({
                "ok": False,
                "message": message
            })

        flash(message)
        return redirect(url_for("filter_photos_page"))

    if face_crop:
        face_crop_path = possible_path / face_crop
        if face_crop_path.exists():
            face_crop_path.unlink()

    if is_ajax:
        return jsonify({
            "ok": True,
            "message": message
        })

    flash(message)
    return redirect(url_for("filter_photos_page"))

@web_bp.route("/review")
@login_required
def review_page():
    family_id = get_current_family_id()

    members = database_read("""
        SELECT id, name
        FROM family_members
        WHERE family_id = ?
        ORDER BY name
    """, (family_id,))

    rows = database_read("""
        SELECT *
        FROM learning_reviews
        WHERE family_id = ?
          AND action IN ('unknown', 'manual', 'rejected')
          AND photo_path IS NOT NULL
        ORDER BY created_at DESC
    """, (family_id,))

    photos = {}

    for item in rows:
        key = item["photo_path"]

        if key not in photos:
            photos[key] = {
                "photo_path": key,
                "faces": []
            }

        image_w = item.get("image_w")
        image_h = item.get("image_h")

        if item.get("box_x") is not None and image_w and image_h:
            item["box_x_percent"] = (item["box_x"] / image_w) * 100
            item["box_y_percent"] = (item["box_y"] / image_h) * 100
            item["box_w_percent"] = (item["box_w"] / image_w) * 100
            item["box_h_percent"] = (item["box_h"] / image_h) * 100
        else:
            item["box_x_percent"] = None

        photos[key]["faces"].append(item)

    album_review_items = database_read("""
        SELECT
            pd.id AS detection_id,
            pd.photo_id,
            pd.member_id,
            pd.distance,
            fp.file_path,
            fp.original_filename,
            fm.name AS member_name
        FROM photo_detections pd
        JOIN family_photos fp ON fp.id = pd.photo_id
        LEFT JOIN family_members fm ON fm.id = pd.member_id
        WHERE pd.family_id = ?
          AND pd.status = 'possible'
        ORDER BY pd.created_at DESC
    """, (family_id,))
    for item in album_review_items:
        item["image_token"] = create_photo_token(item["photo_id"], family_id)
    
    
    return render_template(
        "review.html",
        members=members,
        review_photos=list(photos.values()),
        album_review_items=album_review_items
    )

@web_bp.route("/review/delete-learning-card", methods=["POST"])
@login_required
def delete_learning_review_card():
    family_id = get_current_family_id()
    data = request.get_json() or {}
    review_ids = data.get("review_ids", [])

    if not review_ids:
        return jsonify({
            "success": False,
            "message": "No review IDs provided"
        }), 400

    placeholders = ",".join(["?"] * len(review_ids))
    params = list(review_ids) + [family_id]

    reviews = database_read(
        f"""
        SELECT id, photo_path, face_crop_path
        FROM learning_reviews
        WHERE id IN ({placeholders})
          AND family_id = ?
        """,
        params
    )

    database_write(
        f"""
        DELETE FROM learning_reviews
        WHERE id IN ({placeholders})
          AND family_id = ?
        """,
        params
    )

    skipped_dir = get_family_skipped_dir(family_id)
    faces_dir = get_family_possible_faces_dir(family_id)

    for review in reviews:
        photo_path = review.get("photo_path")
        face_crop_path = review.get("face_crop_path")

        if face_crop_path:
            crop_file = faces_dir / Path(face_crop_path).name
            if crop_file.exists() and crop_file.is_file():
                crop_file.unlink()

        if photo_path:
            remaining = database_read(
                """
                SELECT id
                FROM learning_reviews
                WHERE family_id = ?
                  AND photo_path = ?
                LIMIT 1
                """,
                (family_id, photo_path),
                one=True
            )

            if not remaining:
                photo_file = skipped_dir / Path(photo_path).name
                if photo_file.exists() and photo_file.is_file():
                    photo_file.unlink()

    return jsonify({"success": True})

@web_bp.route("/review/delete/<int:review_id>", methods=["POST"])
@login_required
def delete_review(review_id):
    family_id = get_current_family_id()

    rows = database_read("""
        SELECT *
        FROM learning_reviews
        WHERE id = ?
          AND family_id = ?
    """, (review_id, family_id))

    if not rows:
        abort(404)

    review = rows[0]

    # Optional: delete physical files if they exist
    for path_value in [
        review.get("face_crop_path"),
    ]:
        if not path_value:
            continue

        try:
            file_path = Path(path_value)

            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            print("Could not delete review file:", e)

    database_write(
        "DELETE FROM learning_reviews WHERE id = ? AND family_id = ?",
        (review_id, family_id)
    )

    return jsonify({
        "success": True,
        "review_id": review_id
    })

@web_bp.route("/review-face/<int:review_id>", methods=["POST"])
@login_required
def review_face(review_id):
    family_id = get_current_family_id()
    reviewed_member_id = request.form.get("member_id", type=int)

    if not reviewed_member_id:
        return jsonify({"success": False, "error": "Missing member_id"}), 400

    skipped_dir = get_family_skipped_dir(family_id)
    results_dir = get_family_results_dir(family_id)

    review = database_read(
        """
        SELECT *
        FROM learning_reviews
        WHERE id = ?
          AND family_id = ?
        """,
        (review_id, family_id),
        one=True
    )

    if not review:
        return jsonify({"success": False, "error": "Review item not found"}), 404

    member = database_read(
        """
        SELECT *
        FROM family_members
        WHERE id = ?
          AND family_id = ?
        """,
        (reviewed_member_id, family_id),
        one=True
    )

    if not member:
        return jsonify({"success": False, "error": "Member not found"}), 404

    skipped_filename = Path(review["photo_path"]).name
    skipped_path = skipped_dir / skipped_filename
    result_path = results_dir / skipped_filename

    if skipped_path.exists() and not result_path.exists():
        shutil.copy(str(skipped_path), str(result_path))

    photo_id = review["photo_id"]

    if not photo_id:
        photo_id = create_family_photo(
            family_id=family_id,
            file_path=result_path,
            original_filename=skipped_filename,
            source="manual-review"
        )

    create_photo_detection(
        family_id=family_id,
        photo_id=photo_id,
        member_id=reviewed_member_id,
        face_crop_path=review["face_crop_path"],
        distance=review["distance"],
        status="confirmed",
        confirmed_by_user=1
    )

    learned = False

    if review["face_crop_path"]:
        face_crop_path = Path(review["face_crop_path"])

        if not face_crop_path.exists():
            face_crop_path = get_family_base_dir(family_id) / review["face_crop_path"]

        print("REVIEW LEARN face_crop_path:", face_crop_path)
        print("REVIEW LEARN exists:", face_crop_path.exists())

        if face_crop_path.exists():
            original_photo_path = get_family_skipped_dir(family_id) / Path(review["photo_path"]).name

            learned = save_accepted_face_embedding(
                member["name"],
                str(face_crop_path),
                DeepFace,
                original_photo_path=str(original_photo_path)
            )

    database_write(
        """
        UPDATE learning_reviews
        SET reviewed_member_id = ?,
            action = 'resolved'
        WHERE id = ?
          AND family_id = ?
        """,
        (reviewed_member_id, review_id, family_id)
    )

    return jsonify({
        "success": True,
        "member_name": member["name"],
        "learned": learned
    })

@web_bp.route("/rescan-skipped")
def rescan_skipped():
    family_id = get_current_family_id()
    family_embeddings = get_all_family_embeddings()
    results_dir = get_family_results_dir(family_id)
    skipped_dir = get_family_skipped_dir(family_id)
    count = 0

    for file in skipped_dir.iterdir():
        if not file.is_file():
            continue

        found, matched_names, matches, _, _ = image_contains_known_family(
            file,
            family_embeddings,
            DeepFace,
            family_id,
        )

        if found:
            result_path = results_dir / file.name
            shutil.move(str(file), str(result_path))
            count += 1

    flash(f"Rescanned skipped photos. Found {count} new matches.")
    return redirect(url_for("review_page"))

@web_bp.route("/album/photo/<int:photo_id>/member/<int:member_id>/review", methods=["POST"])
@login_required
def send_detection_to_review(photo_id, member_id):
    family_id = get_current_family_id()
    member = get_owned_member(member_id)
    database_write(
        """
        UPDATE photo_detections
        SET status = 'possible',
            confirmed_by_user = 0
        WHERE family_id = ?
          AND photo_id = ?
          AND member_id = ?
          AND status = 'confirmed'
        """,
        (family_id, photo_id, member_id)
    )

    flash("Detection sent back to review.")

    return redirect(request.referrer or url_for("family_album"))

@web_bp.route("/album-file/<int:photo_id>")
@login_required
def serve_album_file(photo_id):
    family_id = get_current_family_id()

    photo = get_owned_photo(photo_id)

    verify_photo_token(photo_id, family_id)

    file_path = Path(photo["file_path"])

    if not file_path.exists():
        abort(404)

    return send_from_directory(file_path.parent, file_path.name)

@web_bp.route("/scan-one-photo", methods=["POST"])
@login_required
def scan_one_photo_ajax():
    if "photo" not in request.files:
        return jsonify({"ok": False, "error": "No photo"}), 400

    family_id = get_current_family_id()
    family_embeddings = get_all_family_embeddings()

    result = scan_one_family_photo(
        request.files["photo"],
        family_id,
        family_embeddings,
        DeepFace,
        path_builder=lambda kind, name: url_for(
            "serve_result_file" if kind == "result" else "serve_possible_file",
            filename=name,
        ),
    )

    return jsonify({
        "ok": True,
        "result": result
    })

