-- families already planned / added
CREATE TABLE IF NOT EXISTS families (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- every uploaded/filter photo
CREATE TABLE IF NOT EXISTS family_photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    family_id INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    original_filename TEXT,
    source TEXT DEFAULT 'web',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (family_id) REFERENCES families(id)
);

-- every detected face inside a photo
CREATE TABLE IF NOT EXISTS photo_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    family_id INTEGER NOT NULL,
    photo_id INTEGER NOT NULL,
    member_id INTEGER,
    face_crop_path TEXT,
    distance REAL,
    status TEXT NOT NULL DEFAULT 'possible',
    confirmed_by_user INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (family_id) REFERENCES families(id),
    FOREIGN KEY (photo_id) REFERENCES family_photos(id),
    FOREIGN KEY (member_id) REFERENCES family_members(id)
);

CREATE INDEX IF NOT EXISTS idx_family_photos_family
ON family_photos(family_id);

CREATE INDEX IF NOT EXISTS idx_photo_detections_family
ON photo_detections(family_id);

CREATE INDEX IF NOT EXISTS idx_photo_detections_member
ON photo_detections(member_id);

CREATE INDEX IF NOT EXISTS idx_photo_detections_photo
ON photo_detections(photo_id);

CREATE INDEX IF NOT EXISTS idx_photo_detections_status
ON photo_detections(status);