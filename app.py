import base64
import json
import shutil
import sqlite3
import uuid
from pathlib import Path

import cv2
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
app.secret_key = "family-focus-dev-secret-key"
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


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "heic", "heif"}

TEMP_DIR = Path(DB_PATH).parent / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Database helpers
# -------------------------

def database_read(query, params=()):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

def database_write(query, params=()):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(query, params)
        conn.commit()
        return cur.lastrowid

def column_exists(table_name, column_name):
    rows = database_read(f"PRAGMA table_info({table_name})")
    return any(row["name"] == column_name for row in rows)

def add_column_if_missing(table_name, column_name, column_sql):
    if not column_exists(table_name, column_name):
        database_write(f"""
            ALTER TABLE {table_name}
            ADD COLUMN {column_name} {column_sql}
        """)

def init_db():
    PROFILE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    INCOMING_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    #Possible#
    POSSIBLE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    POSSIBLE_FACES_DIR.mkdir(parents=True, exist_ok=True)
    #########
    RESULTS_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    database_write("""
        CREATE TABLE IF NOT EXISTS family_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    database_write("""
        CREATE TABLE IF NOT EXISTS member_photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            member_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(member_id) REFERENCES family_members(id)
        )
    """)

    database_write("""
        CREATE TABLE IF NOT EXISTS member_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            member_id INTEGER NOT NULL,
            photo_id INTEGER,
            embedding TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    database_write("""
            CREATE TABLE IF NOT EXISTS learning_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                member_id INTEGER,
                action TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
    """)

    database_write("""
    CREATE TABLE IF NOT EXISTS families (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        family_name TEXT NOT NULL,
        client_id TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

    database_write("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            family_id INTEGER NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            telegram_chat_id TEXT,
            telegram_username TEXT,
            role TEXT DEFAULT 'admin',
            is_active INTEGER DEFAULT 1,
            last_login TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    #Family ID 

    add_column_if_missing(
        "family_members",
        "family_id",
        "INTEGER"
    )

    add_column_if_missing(
        "member_photos",
        "family_id",
        "INTEGER"
    )

    add_column_if_missing(
        "member_embeddings",
        "family_id",
        "INTEGER"
    )

    add_column_if_missing(
        "learning_reviews",
        "family_id",
        "INTEGER"
    ) 

    family_id = create_default_family()
    assign_existing_data_to_family(family_id)   

def seed_family_members():
    family_id = create_default_family()

    for name in [
        "Karin",
        "Erel",
        "Emma",
        "Eithan",
        "Jimmy"
    ]:
        try:
            database_write("""
                INSERT INTO family_members (
                    family_id,
                    name
                )
                VALUES (?, ?)
            """, (
                family_id,
                name
            ))
        except sqlite3.IntegrityError:
            pass

def create_default_family():
    family = database_read("""
        SELECT *
        FROM families
        WHERE client_id = ?
    """, ("default-family",))

    if family:
        return family[0]["id"]

    database_write("""
        INSERT INTO families (
            family_name,
            client_id
        )
        VALUES (?, ?)
    """, (
        "Adda Family",
        "default-family"
    ))

    family = database_read("""
        SELECT *
        FROM families
        WHERE client_id = ?
    """, ("default-family",))

    return family[0]["id"]

def assign_existing_data_to_family(family_id):
    database_write("""
        UPDATE family_members
        SET family_id = ?
        WHERE family_id IS NULL
    """, (family_id,))

    database_write("""
        UPDATE member_photos
        SET family_id = ?
        WHERE family_id IS NULL
    """, (family_id,))

    database_write("""
        UPDATE member_embeddings
        SET family_id = ?
        WHERE family_id IS NULL
    """, (family_id,))

    database_write("""
        UPDATE learning_reviews
        SET family_id = ?
        WHERE family_id IS NULL
    """, (family_id,))

def get_current_family_id():
    return 1


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


#Member#


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

    profile_filename = f"{uuid.uuid4()}.jpg"
    profile_path = PROFILE_UPLOAD_DIR / profile_filename

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
# Main pages
# -------------------------
@app.before_request
def log_every_request():
    print("REQUEST:", request.method, request.path)
    print("CONTENT LENGTH:", request.content_length)

@app.route("/")
def home():
    members = get_all_members()

    for member in members:
        print(
            member["name"],
            "->",
            member.get("profile_photo")
        )

    return render_template(
        "index.html",
        members=members
    )


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
def member_page(member_id):
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
        ORDER BY created_at DESC
        """,
        (member_id,)
    )

    for photo in photos:
        photo["filename"] = Path(photo["file_path"]).name

    return render_template(
        "member.html",
        member=member,
        photos=photos,
        deepface_available=DeepFace is not None
    )


# -------------------------
# Profile photo upload
# -------------------------

@app.route("/member/<int:member_id>/upload", methods=["POST"])
@app.route("/members/<int:member_id>/upload", methods=["POST"])
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


@app.route("/profile-file/<int:member_id>/<filename>")
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
def camera():
    return render_template(
        "camera.html",
        deepface_available=DeepFace is not None
    )

@app.route("/api/recognize-frame", methods=["POST"])
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

@app.route("/filter-photos")
def filter_photos_page():
    return render_template("filter_photos.html")

@app.route("/filter-photos", methods=["POST"])
def filter_photos_upload():
    files = request.files.getlist("photos")

    if not files:
        flash("No photos selected.")
        return redirect(url_for("filter_photos_page"))

    kept = []
    possible = []
    skipped = 0

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
            skipped += 1

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
def serve_result_file(filename):
    return send_from_directory(RESULTS_UPLOAD_DIR, filename)

@app.route("/possible/<filename>")
def serve_possible_file(filename):
    return send_from_directory(POSSIBLE_UPLOAD_DIR, filename)

@app.route("/review-possible/<filename>/<action>")
def review_possible_photo(filename, action):
    member_name = request.args.get("member_name")
    face_crop = request.args.get("face_crop")
    family_id = get_current_family_id()

    is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

    possible_path = POSSIBLE_UPLOAD_DIR / filename
    result_path = RESULTS_UPLOAD_DIR / filename

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
        if member_name and face_crop:
            face_crop_path = POSSIBLE_FACES_DIR / face_crop

            if face_crop_path.exists():
                save_accepted_face_embedding(
                    member_name,
                    str(face_crop_path),
                    DeepFace
                )

        if member_id:
            database_write(
                """
                INSERT INTO learning_reviews (
                    family_id,
                    member_id,
                    action
                )
                VALUES (?, ?, ?)
                """,
                (family_id, member_id, "accepted")
            )

        shutil.move(str(possible_path), str(result_path))

        message = f"Accepted and learned as {member_name}."

    elif action == "no":
        if member_id:
            database_write(
                """
                INSERT INTO learning_reviews (
                    family_id,
                    member_id,
                    action
                )
                VALUES (?, ?, ?)
                """,
                (family_id, member_id, "rejected")
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
# -------------------------
# Run app
# -------------------------

if __name__ == "__main__":
    init_db()
    seed_family_members()
    app.run(debug=True, host="0.0.0.0", port=5000)