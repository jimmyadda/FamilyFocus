from pathlib import Path
import sqlite3
import hashlib
import secrets
from flask import session, abort
import shutil
from pathlib import Path
from config import (
    DB_PATH,
    FAMILIES_DIR,
    TEMP_UPLOAD_DIR,
)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "heic", "heif"}

def calculate_file_hash(file_path):
    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()

def create_family_photo(family_id, file_path, original_filename=None, source="web"):
    file_hash = calculate_file_hash(file_path)
    existing = database_read(
        """
        SELECT id
        FROM family_photos
        WHERE family_id = ?
          AND file_hash = ?
        """,
        (family_id, file_hash)
    )

    if existing:
        return existing[0]['id']

    return database_write(
        """
        INSERT INTO family_photos (
            family_id,
            file_path,
            original_filename,
            source,
            file_hash
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (family_id, str(file_path), original_filename, source, file_hash)
    )

def create_photo_detection(
    family_id,
    photo_id,
    member_id,
    face_crop_path,
    distance,
    status,
    confirmed_by_user=0
    ):
    existing = database_read(
        """
        SELECT id, status
        FROM photo_detections
        WHERE family_id = ?
          AND photo_id = ?
          AND member_id = ?
        """,
        (family_id, photo_id, member_id)
    )

    if existing:
        existing_status = existing[0]["status"]

        if existing_status == "confirmed":
            return existing[0]["id"]

        if existing_status == "possible" and status == "confirmed":
            database_write(
                """
                UPDATE photo_detections
                SET status = 'confirmed',
                    distance = ?,
                    confirmed_by_user = ?
                WHERE id = ?
                """,
                (distance, confirmed_by_user, existing[0]["id"])
            )

        return existing[0]["id"]

    return database_write(
        """
        INSERT INTO photo_detections
        (family_id, photo_id, member_id, face_crop_path, distance, status, confirmed_by_user)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            family_id,
            photo_id,
            member_id,
            str(face_crop_path) if face_crop_path else None,
            distance,
            status,
            confirmed_by_user
        )
    )

def detection_exists(photo_id, member_id, status):
    rows = database_read(
        """
        SELECT id
        FROM photo_detections
        WHERE photo_id = ?
          AND member_id = ?
          AND status = ?
        """,
        (photo_id, member_id, status)
    )

    return len(rows) > 0

def ensure_family_storage_keys():
    families = database_read(
        "SELECT id, client_id, storage_key FROM families"
    )

    for family in families:
        if not family["storage_key"]:
            storage_key = secrets.token_hex(16)

            database_write(
                "UPDATE families SET storage_key = ? WHERE id = ?",
                (storage_key, family["id"])
            )

#Handle folder 

def migrate_family_folders_to_storage_key():
    families = database_read(
        "SELECT id, client_id, storage_key FROM families"
    )

    for family in families:
        client_id = family["client_id"]
        storage_key = family["storage_key"]

        if not client_id or not storage_key:
            continue

        old_dir = Path("instance") / client_id
        new_dir = Path("instance") / "families" / storage_key

        if not old_dir.exists():
            new_dir.mkdir(parents=True, exist_ok=True)
            continue

        if new_dir.exists():
            print(f"Family folder already migrated: {new_dir}")
            continue

        new_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_dir), str(new_dir))

        print(f"Migrated family folder: {old_dir} -> {new_dir}")

def get_family_storage_key(family_id):
    rows = database_read(
        "SELECT storage_key FROM families WHERE id = ?",
        (family_id,)
    )

    if not rows:
        raise Exception(f"No family found for family_id={family_id}")

    storage_key = rows[0].get("storage_key")

    if not storage_key:
        raise Exception(f"No storage_key found for family_id={family_id}")

    return storage_key

def get_family_base_dir(family_id):
    return FAMILIES_DIR / get_family_storage_key(family_id)

def get_family_profiles_dir(family_id):
    return get_family_base_dir(family_id) / "profiles"

def get_family_results_dir(family_id):
    return get_family_base_dir(family_id) / "results"

def get_family_possible_dir(family_id):
    return get_family_base_dir(family_id) / "possible"

def get_family_possible_faces_dir(family_id):
    return get_family_base_dir(family_id) / "possible_faces"

def get_family_skipped_dir(family_id):
    return get_family_base_dir(family_id) / "skipped"

def get_incoming_upload_dir(family_id):
    return get_family_base_dir(family_id) / "incoming"

def ensure_family_dirs(family_id):
    dirs = [
        get_family_profiles_dir(family_id),
        get_family_results_dir(family_id),
        get_incoming_upload_dir(family_id),
        get_family_possible_dir(family_id),
        get_family_possible_faces_dir(family_id),
        get_family_skipped_dir(family_id),
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def ensure_app_dirs():
    FAMILIES_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Database helpers
# -------------------------
def init_db():

    database_write("""
        CREATE TABLE IF NOT EXISTS family_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            family_id INTEGER,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(family_id, name)
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

            family_id INTEGER NOT NULL,

            photo_id INTEGER,
            photo_path TEXT,

            face_crop_path TEXT,

            box_x INTEGER,
            box_y INTEGER,
            box_w INTEGER,
            box_h INTEGER,
            predicted_member_id INTEGER,
            reviewed_member_id INTEGER,
            distance REAL,
            action TEXT NOT NULL,
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
    #Users Tables & Indexes
    database_write("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            family_id INTEGER NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            first_name TEXT,
            last_name TEXT,
            telegram_chat_id TEXT,
            telegram_username TEXT,
            role TEXT DEFAULT 'admin',
            is_active INTEGER DEFAULT 1,
            last_login TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (family_id) REFERENCES families(id)
        )
    """)
    
    database_write("""
    CREATE INDEX IF NOT EXISTS idx_users_family_id
    ON users(family_id) 
    """)

    database_write("""
    CREATE INDEX IF NOT EXISTS idx_users_telegram_chat_id
    ON users(telegram_chat_id)
    """)

    database_write("""
    CREATE TABLE IF NOT EXISTS user_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    token TEXT NOT NULL,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(id)
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

    database_write("""
        CREATE TABLE IF NOT EXISTS family_photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            family_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            original_filename TEXT,
            source TEXT DEFAULT 'web',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    database_write("""
        CREATE TABLE IF NOT EXISTS photo_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            family_id INTEGER NOT NULL,
            photo_id INTEGER NOT NULL,
            member_id INTEGER,
            face_crop_path TEXT,
            distance REAL,
            status TEXT NOT NULL DEFAULT 'possible',
            confirmed_by_user INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


    ## Telegram tables#
    database_write("""
        CREATE TABLE IF NOT EXISTS telegram_registration_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            family_name TEXT NOT NULL,
            email TEXT NOT NULL,
            telegram_chat_id TEXT NOT NULL,
            telegram_username TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT
        )
    """)
    database_write("""
        CREATE TABLE IF NOT EXISTS telegram_link_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token TEXT UNIQUE NOT NULL,
            user_id INTEGER NOT NULL,
            family_id INTEGER NOT NULL,
            expires_at TEXT NOT NULL,
            used_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    #Telegram batch uploads
    database_write("""
        CREATE TABLE IF NOT EXISTS telegram_upload_batches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        family_id INTEGER NOT NULL,
        telegram_chat_id TEXT NOT NULL,
        status TEXT DEFAULT 'processing',
        upload_count INTEGER DEFAULT 0,
        confirmed_count INTEGER DEFAULT 0,
        possible_count INTEGER DEFAULT 0,
        skipped_count INTEGER DEFAULT 0,
        result_url TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        finished_at TEXT
    )
    """)


    database_write("""
        CREATE TABLE IF NOT EXISTS telegram_upload_batch_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER NOT NULL,
            photo_id INTEGER,
            original_filename TEXT,
            status TEXT,
            detected_member_name TEXT,
            possible_member_name TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)


    add_column_if_missing(
        "family_photos",
        "family_id",
        "INTEGER"
    )

    add_column_if_missing(
        "photo_detections",
        "family_id",
        "INTEGER"
    )

    add_column_if_missing(
        "family_photos",
        "file_hash",
        "TEXT"
    )
    database_write("""
    CREATE UNIQUE INDEX IF NOT EXISTS idx_family_photo_hash
    ON family_photos(family_id, file_hash)
    """)
    
    database_write("""
        CREATE TABLE IF NOT EXISTS photo_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_id INTEGER NOT NULL,
            face_crop_path TEXT,
            box_x INTEGER,
            box_y INTEGER,
            box_w INTEGER,
            box_h INTEGER,
            predicted_member_id INTEGER,
            predicted_name TEXT,
            distance REAL,
            status TEXT, -- confirmed / possible / unknown / rejected / manual
            reviewed_member_id INTEGER,
            reviewed_at TEXT
        )
    """)

    #Users
    add_column_if_missing("users", "first_name", "TEXT")
    add_column_if_missing("users", "last_name", "TEXT")

     #Families
    add_column_if_missing("families", "storage_key", "TEXT")

    # 1. Make sure default family exists first
    family_id = create_default_family()

    # 2. Now make sure every family has storage_key
    ensure_family_storage_keys()

    # 3. Assign old rows to family_id
    assign_existing_data_to_family(family_id)

    # 4. Fix constraints after family_id exists
    fix_family_members_unique_constraint()

    # 5. Move old instance/adda folder to instance/families/<storage_key>
    migrate_family_folders_to_storage_key()

    # 6. Make sure required folders exist
    ensure_family_dirs(family_id)

    #Learning_Review#
    add_column_if_missing("learning_reviews", "family_id", "INTEGER")
    add_column_if_missing("learning_reviews", "photo_id", "INTEGER")
    add_column_if_missing("learning_reviews", "photo_path", "TEXT")
    add_column_if_missing("learning_reviews", "face_crop_path", "TEXT")

    add_column_if_missing("learning_reviews", "box_x", "INTEGER")
    add_column_if_missing("learning_reviews", "box_y", "INTEGER")
    add_column_if_missing("learning_reviews", "box_w", "INTEGER")
    add_column_if_missing("learning_reviews", "box_h", "INTEGER")

    add_column_if_missing("learning_reviews", "predicted_member_id", "INTEGER")
    add_column_if_missing("learning_reviews", "reviewed_member_id", "INTEGER")
    add_column_if_missing("learning_reviews", "distance", "REAL")
    add_column_if_missing("learning_reviews", "image_w", "INTEGER")
    add_column_if_missing("learning_reviews", "image_h", "INTEGER")


    families = database_read("SELECT id, client_id, storage_key FROM families")
    for family in families:
        if not family["storage_key"]:
            storage_key = secrets.token_hex(16)
            database_write(
                "UPDATE families SET storage_key = ? WHERE id = ?",
                (storage_key, family["id"])
            )
  
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
    database_write("""
        UPDATE family_photos
        SET family_id = ?
        WHERE family_id IS NULL
    """, (family_id,))

    database_write("""
        UPDATE photo_detections
        SET family_id = ?
        WHERE family_id IS NULL
    """, (family_id,))


def database_read(query, params=(), one=False):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        cursor = conn.execute(query, params)

        if one:
            row = cursor.fetchone()
            return dict(row) if row else None

        rows = cursor.fetchall()
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
        
def get_confirmed_family_photos(family_id):
    return database_read(
        """
        SELECT DISTINCT fp.*
        FROM family_photos fp
        JOIN photo_detections pd ON pd.photo_id = fp.id
        WHERE fp.family_id = ?
          AND pd.status = 'confirmed'
        ORDER BY fp.created_at DESC
        """,
        (family_id,)
    )

def get_possible_family_photos(family_id):
    return database_read(
        """
        SELECT 
            pd.*,
            fp.file_path,
            fm.name AS member_name
        FROM photo_detections pd
        JOIN family_photos fp ON fp.id = pd.photo_id
        LEFT JOIN family_members fm ON fm.id = pd.member_id
        WHERE pd.family_id = ?
          AND pd.status = 'possible'
        ORDER BY pd.created_at DESC
        """,
        (family_id,)
    )

def get_member_detected_album(family_id, member_id):
    return database_read(
        """
        SELECT DISTINCT
            fp.id,
            fp.file_path,
            fp.original_filename,
            fp.created_at
        FROM family_photos fp
        JOIN photo_detections pd ON pd.photo_id = fp.id
        WHERE fp.family_id = ?
          AND pd.member_id = ?
          AND pd.status = 'confirmed'
        ORDER BY fp.created_at DESC
        """,
        (family_id, member_id)
    )

def get_photo_people(family_id, photo_id):
    return database_read(
        """
        SELECT 
            fm.id,
            fm.name,
            pd.status,
            pd.distance
        FROM photo_detections pd
        JOIN family_members fm ON fm.id = pd.member_id
        WHERE pd.family_id = ?
          AND pd.photo_id = ?
          AND pd.status = 'confirmed'
        ORDER BY fm.name
        """,
        (family_id, photo_id)
    )

def get_family_members_with_detections(family_id):
    return database_read(
        """
        SELECT DISTINCT
            fm.id,
            fm.name
        FROM family_members fm
        JOIN photo_detections pd ON pd.member_id = fm.id
        WHERE fm.family_id = ?
          AND pd.status = 'confirmed'
        ORDER BY fm.name
        """,
        (family_id,)
    )
   
def fix_family_members_unique_constraint():
    database_write("""
        CREATE TABLE IF NOT EXISTS family_members_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            family_id INTEGER,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(family_id, name)
        )
    """)

    database_write("""
        INSERT OR IGNORE INTO family_members_new (
            id,
            family_id,
            name,
            created_at
        )
        SELECT
            id,
            family_id,
            name,
            created_at
        FROM family_members
    """)

    database_write("DROP TABLE family_members")

    database_write("""
        ALTER TABLE family_members_new
        RENAME TO family_members
    """)

def save_learning_review(
    family_id,
    photo_path,
    face_crop_path,
    predicted_member_id,
    reviewed_member_id,
    distance,
    action,
    box_x=None,
    box_y=None,
    box_w=None,
    box_h=None,
    image_w=None,
    image_h=None,
    photo_id=None
):
    database_write("""
        INSERT INTO learning_reviews (
            family_id,
            photo_id,
            photo_path,
            face_crop_path,
            box_x,
            box_y,
            box_w,
            box_h,
            image_w,
            image_h,
            predicted_member_id,
            reviewed_member_id,
            distance,
            action
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        family_id,
        photo_id,
        str(photo_path) if photo_path else None,
        str(face_crop_path) if face_crop_path else None,
        box_x,
        box_y,
        box_w,
        box_h,
        image_w,
        image_h,
        predicted_member_id,
        reviewed_member_id,
        distance,
        action
    ))

def extract_faces_for_review(image_path, filename, DeepFace):
    import cv2

    image = cv2.imread(str(image_path))

    if image is None:
        return []

    image_h, image_w = image.shape[:2]
    family_id = create_default_family()
    possible_faces = get_family_possible_faces_dir(family_id)
    faces_for_review = []

    try:
        faces = DeepFace.extract_faces(
            img_path=str(image_path),
            detector_backend="retinaface",
            enforce_detection=False,
            align=True
        )
    except Exception as e:
        print("Review face extract error:", e)
        return []

    for index, face in enumerate(faces):
        area = face.get("facial_area", {})

        x = int(area.get("x", 0))
        y = int(area.get("y", 0))
        w = int(area.get("w", 0))
        h = int(area.get("h", 0))

        if w <= 0 or h <= 0:
            continue

        crop = image[y:y+h, x:x+w]

        if crop.size == 0:
            continue

        face_crop_filename = f"{Path(filename).stem}_unknown_{index}.jpg"
        face_crop_path = possible_faces / face_crop_filename

        cv2.imwrite(str(face_crop_path), crop)

        faces_for_review.append({
            "face_crop_filename": face_crop_filename,
            "box_x": x,
            "box_y": y,
            "box_w": w,
            "box_h": h,
            "image_w": image_w,
            "image_h": image_h
        })

    return faces_for_review    

#Family ID & ownership
def require_family_id():
    family_id = session.get("family_id")
    if not family_id:
        abort(403)
    return family_id

def get_owned_member(member_id):
    family_id = require_family_id()

    member = database_read(
        """
        SELECT *
        FROM family_members
        WHERE id = ?
          AND family_id = ?
        """,
        (member_id, family_id),
        one=True
    )

    if not member:
        abort(403)

    return member   

def get_owned_photo(photo_id):
    family_id = require_family_id()

    photo = database_read(
        """
        SELECT *
        FROM family_photos
        WHERE id = ?
          AND family_id = ?
        """,
        (photo_id, family_id),
        one=True
    )

    if not photo:
        abort(403)

    return photo

def get_owned_detection(detection_id):
    family_id = require_family_id()

    detection = database_read(
        """
        SELECT *
        FROM photo_detections
        WHERE id = ?
          AND family_id = ?
        """,
        (detection_id, family_id),
        one=True
    )

    if not detection:
        abort(403)

    return detection  

def get_current_user():
    
    user_id = session.get("user_id")
    if not user_id:
        return None

    return database_read(
        """
        SELECT *
        FROM users
        WHERE id = ?
        """,
        (user_id,),
        one=True
    )          