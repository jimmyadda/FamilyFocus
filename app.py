import base64
import json
import shutil
import sqlite3
import uuid
from pathlib import Path

import cv2
from db_helpers import *

import numpy as np
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
    get_flashed_messages
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from auth import login_required, get_current_family_id
try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None

from config import (
    BASE_CLIENT_DIR,
    DB_PATH,
    PHOTO_POSSIBLE_THRESHOLD,
    PROFILE_UPLOAD_DIR,
    INCOMING_UPLOAD_DIR,
    RESULTS_UPLOAD_DIR,
    MIN_FACE_SIZE_CAMERA,
    MIN_FACE_SIZE_IMAGE,
    TEMP_UPLOAD_DIR,
    PHOTO_MATCH_THRESHOLD, 
    CAMERA_MATCH_THRESHOLD,
)

from face_utils import (
    detect_faces_with_opencv,
    crop_face,
    identify_face,
    create_annotated_family_image,
    represent_face,
)


app = Flask(__name__)
app.secret_key = "vsbvrbdbdbdbXCVvsvvsv156156VVVgrgergerg"  # Change this to a random secret key in production
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024

for folder in [
    BASE_CLIENT_DIR,
    PROFILE_UPLOAD_DIR,
    INCOMING_UPLOAD_DIR,
    RESULTS_UPLOAD_DIR,
    TEMP_UPLOAD_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

POSSIBLE_UPLOAD_DIR = BASE_CLIENT_DIR / "possible"
POSSIBLE_FACES_DIR = BASE_CLIENT_DIR / "possible_faces"

POSSIBLE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)    
SKIPPED_UPLOAD_DIR = BASE_CLIENT_DIR / "skipped"
SKIPPED_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "heic", "heif"}

TEMP_DIR = Path(DB_PATH).parent / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)





# -------------------------
# General helpers
# -------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


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
        family_embeddings.append({
            "member_id": row["member_id"],
            "name": row["name"],
            "embedding": json.loads(row["embedding"])
        })

    return family_embeddings

def image_contains_known_family(image_path, family_embeddings, DeepFace):
    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read image:", image_path)
        return False, [], [], [], []

    try:
        faces = DeepFace.extract_faces(
            img_path=str(image_path),
            detector_backend="retinaface",
            enforce_detection=False,
            align=True
        )
    except Exception as error:
        print("RetinaFace failed, falling back to OpenCV:", repr(error))

        opencv_faces = detect_faces_with_opencv(
            image,
            min_face_size=MIN_FACE_SIZE_IMAGE
        )

        faces = []

        for x, y, w, h in opencv_faces:
            faces.append({
                "face": crop_face(image, (int(x), int(y), int(w), int(h))),
                "facial_area": {
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h)
                }
            })

    matches = []
    possible_matches = []
    matched_names = []
    possible_names = []

    print("FACES FOUND:", len(faces))

    for i, face_obj in enumerate(faces):
        facial_area = face_obj.get("facial_area", {})

        x = int(facial_area.get("x", 0))
        y = int(facial_area.get("y", 0))
        w = int(facial_area.get("w", 0))
        h = int(facial_area.get("h", 0))

        if w < MIN_FACE_SIZE_IMAGE or h < MIN_FACE_SIZE_IMAGE:
            continue

        print("FACE BOX:", x, y, w, h)

        face_crop = face_obj.get("face")

        if face_crop is None:
            continue

        if face_crop.dtype != "uint8":
            face_crop = (face_crop * 255).astype("uint8")
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)

        temp_face_path = str(TEMP_UPLOAD_DIR / f"temp_filter_face_{i}.jpg")
        cv2.imwrite(temp_face_path, face_crop)

        identified = identify_face(
            temp_face_path,
            family_embeddings,
            DeepFace,
            threshold=PHOTO_MATCH_THRESHOLD
        )

        if not identified:
            member_name = None
            distance = 999.0
            status = "rejected"
        else:
            member_name, distance, status = identified

        print(
            f"FACE {i} | path={temp_face_path} | "
            f"name={member_name} | "
            f"distance={distance:.4f} | "
            f"status={status}"
        )

        if member_name is None:
            continue

        face_crop_filename = None

        if status == "possible":
            face_crop_filename = f"{Path(image_path).stem}_face_{i}.jpg"
            face_crop_path = POSSIBLE_FACES_DIR / face_crop_filename
            cv2.imwrite(str(face_crop_path), face_crop)

        match = {
            "face_index": i,
            "name": member_name,
            "member_id": None,
            "box": {
                "x": x,
                "y": y,
                "w": w,
                "h": h
            },
            "distance": distance,
            "status": status,
            "face_crop": face_crop_filename
        }

        if status == "matched":
            matches.append(match)

            if member_name not in matched_names:
                matched_names.append(member_name)

        elif status == "possible":
            possible_matches.append(match)

            if member_name not in possible_names:
                possible_names.append(member_name)

    return len(matches) > 0, matched_names, matches, possible_names, possible_matches


# -------------------------
# Members helpers
# -------------------------


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

def save_accepted_face_embedding(member_name, face_crop_path, DeepFace):
    member = get_member_by_name(member_name)
    family_id = get_current_family_id()

    if not member:
        print("Member not found:", member_name)
        return False

    member_id = member["id"]

    embedding = represent_face(face_crop_path, DeepFace)

    if embedding is None:
        print("Could not create embedding from accepted face")
        return False

    # Safety check before learning
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
        print("Learning rejected - face too far from profile")
        return False

    member_profile_dir = PROFILE_UPLOAD_DIR / str(member_id)
    member_profile_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(face_crop_path).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        ext = ".jpg"

    profile_filename = f"{uuid.uuid4()}{ext}"
    profile_path = member_profile_dir / profile_filename

    shutil.copy(str(face_crop_path), str(profile_path))
    
    
    family_id = get_current_family_id()
    database_write(
        """
        INSERT INTO member_photos (family_id,member_id, file_path)
        VALUES (?, ?, ?)
        """,
        (family_id,member_id, str(profile_path))
    )

    photo_id = database_read("SELECT last_insert_rowid() AS id")[0]["id"]

    database_write(
        """
        INSERT INTO member_embeddings (family_id,member_id, photo_id, embedding)
        VALUES (?, ?, ?, ?)
        """,
        (
            family_id,
            member_id,
            photo_id,
            json.dumps(embedding)
        )
    )

    print(f"Learned new embedding for {member_name}")
    return True


# -------------------------
# User Login and Registration
# -------------------------

@app.route("/register", methods=["GET", "POST"])
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
        return redirect(url_for("dashboard"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
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
        return redirect(url_for("dashboard"))

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.")
    return redirect(url_for("login"))    
# -------------------------
# Main pages
# -------------------------
@app.route("/")
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

    return render_template(
        "dashboard.html",
        stats=stats,
        members=members,
        recent_photos=recent_photos
    )


# -------------------------
# Member Routes
# -------------------------
@app.route("/members/create", methods=["POST"])
def create_member():
    name = request.form.get("name", "").strip()

    if name:
        try:
            database_write(
                "INSERT INTO family_members (name) VALUES (?)",
                (name,)
            )
        except sqlite3.IntegrityError:
            pass

    return redirect(url_for("home"))


@app.route("/members/<int:member_id>")
@app.route("/member/<int:member_id>")
@login_required
def member_page(member_id):
    family_id = get_current_family_id()

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
        deepface_available=DeepFace is not None
    )

# album - new focus

@app.route("/album")
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

    possible_photos = get_possible_family_photos(family_id)

    return render_template(
        "album.html",
        confirmed_photos=confirmed_photos,
        possible_photos=possible_photos,
        members=members,
        selected_member_id=selected_member_id
    )

@app.route("/member/<int:member_id>/album")
@login_required
def member_album(member_id):
    family_id = get_current_family_id()

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

@app.route("/detections/<int:detection_id>/confirm", methods=["POST"])
@login_required
def confirm_detection(detection_id):
    family_id = get_current_family_id()

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


@app.route("/detections/<int:detection_id>/reject", methods=["POST"])
@login_required
def reject_detection(detection_id):
    family_id = get_current_family_id()

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
# -------------------------
# Profile photo upload
# -------------------------

@app.route("/member/<int:member_id>/upload", methods=["POST"])
@app.route("/members/<int:member_id>/upload", methods=["POST"])
@login_required
def upload_member_photos(member_id):
    print("UPLOAD ROUTE HIT")
    print("CONTENT LENGTH:", request.content_length)

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

    member_dir = PROFILE_UPLOAD_DIR / str(member_id)
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

        saved += 1

    flash(f"Uploaded {saved} photos. Skipped {skipped}.")
    return redirect(url_for("member_page", member_id=member_id))


@app.route("/member-photo/<int:photo_id>/delete", methods=["POST"])
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

@app.route("/profile-file/<int:member_id>/<filename>")
@login_required
def serve_profile_file(member_id, filename):
    folder = PROFILE_UPLOAD_DIR / str(member_id)
    return send_from_directory(folder, filename)

@app.errorhandler(413)
def too_large(e):
    print("413 TOO LARGE:", request.content_length)
    flash("Upload too large. Try fewer photos.")
    return redirect(request.referrer or url_for("index"))


# -------------------------
# Rebuild face profile
# -------------------------

@app.route("/member/<int:member_id>/rebuild-embeddings", methods=["POST"])
@login_required
def rebuild_embeddings(member_id):
    family_id = get_current_family_id()

    if DeepFace is None:
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
            photo_path = PROFILE_UPLOAD_DIR / str(member_id) / Path(photo["file_path"]).name


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

            database_write(
                """
                INSERT INTO member_embeddings
                (family_id,member_id, photo_id, embedding)
                VALUES (?, ?, ?, ?)
                """,
                (
                    family_id,
                    member_id,
                    photo["id"],
                    json.dumps(embedding)
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


# -------------------------
# Live camera recognition
# -------------------------

@app.route("/camera")
@login_required
def camera():
    return render_template(
        "camera.html",
        deepface_available=DeepFace is not None
    )

@app.route("/api/recognize-frame", methods=["POST"])
@login_required
def recognize_frame():
    if DeepFace is None:
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

# -------------------------
# Photo filtering
# -------------------------
@app.route("/family-file/<path:filename>")
def family_file(filename):
    family_id = get_current_family_id()

    base_dir = INSTANCE_DIR / str(family_id)

    return send_from_directory(base_dir, filename)


@app.route("/filter-photos")
@login_required
def filter_photos_page():
    return render_template("filter_photos.html")

@app.route("/filter-photos", methods=["POST"])
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
        if not file or file.filename == "":
            continue

        if not allowed_file(file.filename):
            skipped += 1
            continue

        original = secure_filename(file.filename)
        ext = original.rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4()}.{ext}"

        incoming_path = INCOMING_UPLOAD_DIR / filename
        file.save(incoming_path)

        found, matched_names, matches, possible_names, possible_matches = image_contains_known_family(
            incoming_path,
            family_embeddings,
            DeepFace
        )

        if found:
            result_path = RESULTS_UPLOAD_DIR / filename

            create_annotated_family_image(
                incoming_path,
                matches,
                result_path
            )

            photo_id = create_family_photo(
                family_id=family_id,
                file_path=result_path,
                original_filename=file.filename,
                source="web"
            )

            for match in matches:
                name = match.get("name")
                distance = match.get("distance")

                member_id = get_member_id_by_name(name, family_id)

                if member_id:
                    create_photo_detection(
                        family_id=family_id,
                        photo_id=photo_id,
                        member_id=member_id,
                        face_crop_path=None,
                        distance=distance,
                        status="confirmed",
                        confirmed_by_user=0
                    )

            kept.append({
                "filename": filename,
                "original": original,
                "names": matched_names,
                "matches": matches,
                "path": url_for("serve_result_file", filename=filename),
                "status": "matched"
            })

            try:
                incoming_path.unlink()
            except Exception:
                pass

        elif possible_matches:
            possible_path = POSSIBLE_UPLOAD_DIR / filename

            create_annotated_family_image(
                incoming_path,
                possible_matches,
                possible_path
            )

            photo_id = create_family_photo(
                family_id=family_id,
                file_path=possible_path,
                original_filename=file.filename,
                source="web"
            )

            for match in possible_matches:
                name = match.get("name")
                distance = match.get("distance")

                member_id = get_member_id_by_name(name, family_id)

                if member_id:
                    create_photo_detection(
                        family_id=family_id,
                        photo_id=photo_id,
                        member_id=member_id,
                        face_crop_path=None,
                        distance=distance,
                        status="possible",
                        confirmed_by_user=0
                    )

            possible.append({
                "filename": filename,
                "original": original,
                "names": possible_names,
                "matches": possible_matches,
                "path": url_for("serve_possible_file", filename=filename),
                "status": "possible"
            })

            try:
                incoming_path.unlink()
            except Exception:
                pass

        else:
            skipped_path = SKIPPED_UPLOAD_DIR / filename
            shutil.copy(str(incoming_path), str(skipped_path))

            unknown_faces = extract_faces_for_review(
                incoming_path,
                filename,
                DeepFace
            )

            for face in unknown_faces:
                save_learning_review(
                    family_id=family_id,
                    photo_path=filename,
                    face_crop_path=f"possible_faces/{face['face_crop_filename']}",
                    predicted_member_id=None,
                    reviewed_member_id=None,
                    distance=None,
                    action="unknown",
                    box_x=face["box_x"],
                    box_y=face["box_y"],
                    box_w=face["box_w"],
                    box_h=face["box_h"],
                    image_w=face["image_w"],
                    image_h=face["image_h"]
                )

            try:
                incoming_path.unlink()
            except Exception:
                pass

    get_flashed_messages()

    return render_template(
        "filter_results.html",
        kept=kept,
        possible=possible,
        skipped=skipped
    )

@app.route("/results/<filename>")
@login_required
def serve_result_file(filename):
    return send_from_directory(RESULTS_UPLOAD_DIR, filename)

@app.route("/serve-skipped/<path:filename>")
@login_required
def serve_skipped_file(filename):
    filename = filename.replace("skipped/", "")

    return send_from_directory(
        SKIPPED_UPLOAD_DIR,
        filename
    )
    
@app.route("/possible/<filename>")
@login_required
def serve_possible_file(filename):
    return send_from_directory(POSSIBLE_UPLOAD_DIR, filename)

@app.route("/review-possible/<filename>/<action>")
@login_required
def review_possible_photo(filename, action):
    member_name = request.args.get("member_name")
    face_crop = request.args.get("face_crop")
    family_id = get_current_family_id()

    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

    possible_path = POSSIBLE_UPLOAD_DIR / filename
    result_path = RESULTS_UPLOAD_DIR / filename
    face_crop_path = POSSIBLE_FACES_DIR / face_crop if face_crop else None
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
        face_crop_path = POSSIBLE_FACES_DIR / face_crop
        if face_crop_path.exists():
            face_crop_path.unlink()

    if is_ajax:
        return jsonify({
            "ok": True,
            "message": message
        })

    flash(message)
    return redirect(url_for("filter_photos_page"))

# Fix detections mistakenly marked skipped

@app.route("/review")
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

    return render_template(
        "review.html",
        members=members,
        review_photos=list(photos.values()),
        album_review_items=album_review_items
    )

@app.route("/review-face/<int:review_id>", methods=["POST"])
@login_required
def review_face(review_id):
    family_id = get_current_family_id()
    reviewed_member_id = request.form.get("member_id", type=int)

    review = database_read("""
        SELECT *
        FROM learning_reviews
        WHERE id = ?
          AND family_id = ?
    """, (review_id, family_id))

    if not review:
        flash("Review item not found.")
        return redirect(url_for("review_page"))

    review = review[0]

    member = database_read("""
        SELECT *
        FROM family_members
        WHERE id = ?
          AND family_id = ?
    """, (reviewed_member_id, family_id))

    if not member:
        flash("Member not found.")
        return redirect(url_for("review_page"))

    member = member[0]

    skipped_filename = review["photo_path"].replace("skipped/", "")
    skipped_path = SKIPPED_UPLOAD_DIR / skipped_filename
    result_path = RESULTS_UPLOAD_DIR / skipped_filename

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

    if review["face_crop_path"] and Path(review["face_crop_path"]).exists():
        save_accepted_face_embedding(
            member["name"],
            review["face_crop_path"],
            DeepFace
        )

    database_write("""
        UPDATE learning_reviews
        SET reviewed_member_id = ?,
            action = 'resolved'
        WHERE id = ?
          AND family_id = ?
    """, (reviewed_member_id, review_id, family_id))

    flash(f"Saved as {member['name']}.")
    return redirect(url_for("review_page"))

@app.route("/rescan-skipped")
def rescan_skipped():
    family_id = get_current_family_id()
    family_embeddings = get_all_family_embeddings(family_id)

    count = 0

    for file in SKIPPED_UPLOAD_DIR.iterdir():
        if not file.is_file():
            continue

        found, matched_names, matches = image_contains_known_family(
            file,
            family_embeddings,
            DeepFace
        )

        if found:
            result_path = RESULTS_UPLOAD_DIR / file.name
            shutil.move(str(file), str(result_path))
            count += 1

    flash(f"Rescanned skipped photos. Found {count} new matches.")
    return redirect(url_for("review_page"))

@app.route("/album/photo/<int:photo_id>/member/<int:member_id>/review", methods=["POST"])
@login_required
def send_detection_to_review(photo_id, member_id):
    family_id = get_current_family_id()

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

@app.route("/album-file/<int:photo_id>")
def serve_album_file(photo_id):
    family_id = get_current_family_id()

    rows = database_read(
        """
        SELECT file_path
        FROM family_photos
        WHERE id = ?
          AND family_id = ?
        """,
        (photo_id, family_id)
    )

    if not rows:
        return "File not found", 404

    file_path = Path(rows[0]["file_path"])

    return send_from_directory(file_path.parent, file_path.name)




# -------------------------
# Run app
# -------------------------

if __name__ == "__main__":
    init_db()
    seed_family_members()
    app.run(debug=True, host="0.0.0.0", port=5000)