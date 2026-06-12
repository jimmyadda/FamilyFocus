from pathlib import Path

# Client / tenant
CLIENT_ID = "adda"

# Base folders
BASE_CLIENT_DIR = Path("instance") / CLIENT_ID
DB_PATH = BASE_CLIENT_DIR / "family_focus.db"

PROFILE_UPLOAD_DIR = BASE_CLIENT_DIR / "profiles"
INCOMING_UPLOAD_DIR = BASE_CLIENT_DIR / "incoming"
RESULTS_UPLOAD_DIR = BASE_CLIENT_DIR / "results"
TEMP_UPLOAD_DIR = BASE_CLIENT_DIR / "temp"

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