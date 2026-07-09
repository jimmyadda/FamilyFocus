import uuid
from pathlib import Path

import cv2

from config import MIN_FACE_SIZE_IMAGE

from face_utils import (
    detect_faces_with_opencv,
    crop_face,
    represent_face
)

from db_helpers import (
    insert_member_embedding,
    encrypt_embedding,
    rebuild_member_centroid
)


def create_profile_embedding_from_saved_photo(
    family_id,
    member_id,
    photo_id,
    photo_path,
    DeepFace
):
    image = cv2.imread(str(photo_path))

    if image is None:
        print("PROFILE EMBEDDING: Could not read image:", photo_path)
        return False

    faces = detect_faces_with_opencv(
        image,
        min_face_size=MIN_FACE_SIZE_IMAGE
    )

    if len(faces) == 0:
        print("PROFILE EMBEDDING: No face found")
        return False

    faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
    best_face = faces[0]

    face_crop = crop_face(image, best_face)

    temp_dir = Path("instance/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)

    face_crop_path = temp_dir / f"profile_face_{uuid.uuid4()}.jpg"
    cv2.imwrite(str(face_crop_path), face_crop)

    embedding = represent_face(face_crop_path, DeepFace)

    if embedding is None:
        print("PROFILE EMBEDDING: Could not create embedding")
        return False

    encrypted_embedding = encrypt_embedding(embedding)

    insert_member_embedding(
        family_id=family_id,
        member_id=member_id,
        photo_id=photo_id,
        encrypted_embedding=encrypted_embedding
    )

    print("PROFILE EMBEDDING CREATED:", member_id)

    rebuild_ok = rebuild_member_centroid(member_id)
    print("PROFILE CENTROID REBUILT:", rebuild_ok)

    return True