from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()


# Base folders
DB_PATH = os.getenv("DATABASE_PATH", "instance/family_focus.db")

INSTANCE_DIR = Path("instance")
TEMP_DIR = INSTANCE_DIR / "temp"

TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FAMILIES_DIR = INSTANCE_DIR / "families"
TEMP_UPLOAD_DIR = INSTANCE_DIR / "temp"

SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
EMBEDDING_ENCRYPTION_KEY = os.getenv("EMBEDDING_ENCRYPTION_KEY")
MAX_CONTENT_LENGTH = 300 * 1024 * 1024
# DeepFace settings
FACE_MODEL_NAME = "Facenet512"
FACE_DETECTOR_BACKEND = "opencv"
FACE_ENFORCE_DETECTION = False

# Recognition thresholds
# Lower = stricter. Start strict to reduce false positives.
FACE_MATCH_THRESHOLD = 0.35

PHOTO_MATCH_THRESHOLD = 0.35
PHOTO_POSSIBLE_THRESHOLD = 0.48
POSSIBLE_MATCH_MARGIN = 0.20
PHOTO_POSSIBLE_MIN_DISTANCE = 0.46

CAMERA_MATCH_THRESHOLD = 0.45
# Best match must be clearly better than second-best match
MATCH_MARGIN = 0.03

# Ignore tiny faces
MIN_FACE_SIZE_CAMERA = 45
MIN_FACE_SIZE_IMAGE = 40

# Camera performance
CAMERA_FRAME_MAX_WIDTH = 480
CAMERA_RECOGNITION_INTERVAL_MS = 1200

# Image filtering behavior
DRAW_UNKNOWN_IN_CAMERA = True
DRAW_UNKNOWN_IN_FILTERED_IMAGES = False