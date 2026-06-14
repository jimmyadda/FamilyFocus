from pathlib import Path
import sqlite3

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

POSSIBLE_UPLOAD_DIR = BASE_CLIENT_DIR / "possible"
POSSIBLE_FACES_DIR = BASE_CLIENT_DIR / "possible_faces"

POSSIBLE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)    


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "heic", "heif"}

TEMP_DIR = Path(DB_PATH).parent / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)



def create_family_photo(family_id, file_path, original_filename=None, source="web"):
    return database_write(
        """
        INSERT INTO family_photos (family_id, file_path, original_filename, source)
        VALUES (?, ?, ?, ?)
        """,
        (family_id, str(file_path), original_filename, source)
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
    database_write(
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

# -------------------------
# Database helpers
# -------------------------
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
    family_id = create_default_family()
    assign_existing_data_to_family(family_id)   
    fix_family_members_unique_constraint()


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

def get_current_family_id():
    return 1

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