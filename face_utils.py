import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from config import (
    FACE_MATCH_THRESHOLD,
    MATCH_MARGIN,
    MIN_FACE_SIZE_CAMERA,
    MIN_FACE_SIZE_IMAGE,
    PHOTO_POSSIBLE_MIN_DISTANCE,
    PHOTO_POSSIBLE_THRESHOLD
)


def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)

    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def detect_faces_with_opencv(image, min_face_size=30):

    # If caller sends Path or string, load image first
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))

    # If image is still invalid, stop safely
    if image is None or not isinstance(image, np.ndarray):
        print("detect_faces_with_opencv ERROR:")
        print("Invalid image type:", type(image))
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(50, 50)
    )

    return faces


def crop_face(image, face_box, padding_ratio=0.45):
    x, y, w, h = face_box

    img_h, img_w = image.shape[:2]

    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    return image[y1:y2, x1:x2]


def represent_face(face_path, DeepFace):
    result = DeepFace.represent(
        img_path=str(face_path),
        model_name="Facenet512",
        detector_backend="skip",
        enforce_detection=False
    )

    if not result:
        return None

    return result[0]["embedding"]


def identify_face(face_path, family_embeddings, DeepFace, threshold=None):
    if not family_embeddings:
        return None, 999.0, "rejected"

    if threshold is None:
        threshold = FACE_MATCH_THRESHOLD

    face_embedding = represent_face(face_path, DeepFace)

    if face_embedding is None:
        return None, 999.0, "rejected"

    face_embedding = normalize_embedding(face_embedding)

    family_centroids = build_family_centroids(family_embeddings)

    results = []

    for item in family_centroids:
        distance = cosine_distance(
            face_embedding,
            item["embedding"]
        )

        results.append({
            "name": item["name"],
            "member_id": item["member_id"],
            "distance": distance,
            "count": item["count"]
        })

    if not results:
        return None, 999.0, "rejected"

    results.sort(key=lambda x: x["distance"])

    best = results[0]
    second = results[1] if len(results) > 1 else None

    margin = 999.0
    if second:
        margin = second["distance"] - best["distance"]

    if second and margin < MATCH_MARGIN:
        return None, best["distance"], "rejected"

    if best["distance"] <= threshold:
        return best["name"], best["distance"], "matched"

    if (
        best["distance"] >= PHOTO_POSSIBLE_MIN_DISTANCE
        and best["distance"] <= PHOTO_POSSIBLE_THRESHOLD
    ):
        print(
            f"POSSIBLE | {best['name']} | "
            f"distance={best['distance']:.4f}"
        )
        return best["name"], best["distance"], "possible"

    return None, best["distance"], "rejected"

def create_annotated_family_image(original_path, matches, output_path):
    image = cv2.imread(str(original_path))

    if image is None:
        return False

    for match in matches:
        if not match.get("match"):
            continue

        box = match["box"]
        name = match.get("name", "Family")

        x = int(box["x"])
        y = int(box["y"])
        w = int(box["width"])
        h = int(box["height"])

        color = (255, 0, 0)  # blue

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            color,
            3
        )
        print("ANNOTATING MATCHES:", matches)
        print("SAVING RESULT TO:", output_path)
        cv2.rectangle(
            image,
            (x, max(0, y - 30)),
            (x + 180, y),
            color,
            -1
        )

        cv2.putText(
            image,
            name,
            (x + 5, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    cv2.imwrite(str(output_path), image)
    return True



#region Centroid Embeddings

def normalize_embedding(embedding):
    arr = np.array(embedding, dtype="float32")
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm


def build_member_centroid(member_embeddings):
    """
    member_embeddings = list of embeddings for one member
    returns one normalized centroid embedding
    """
    if not member_embeddings:
        return None

    normalized = [
        normalize_embedding(e)
        for e in member_embeddings
        if e is not None
    ]

    if not normalized:
        return None

    centroid = np.mean(normalized, axis=0)
    centroid = normalize_embedding(centroid)

    return centroid.tolist()

def build_family_centroids(family_embeddings):
    grouped = {}

    for item in family_embeddings:
        member_id = item["member_id"]
        name = item["name"]

        if member_id not in grouped:
            grouped[member_id] = {
                "member_id": member_id,
                "name": name,
                "embeddings": []
            }

        grouped[member_id]["embeddings"].append(item["embedding"])

    centroids = []

    for member_id, data in grouped.items():
        centroid = build_member_centroid(data["embeddings"])

        if centroid is not None:
            centroids.append({
                "member_id": member_id,
                "name": data["name"],
                "embedding": centroid,
                "count": len(data["embeddings"])
            })

    return centroids

#endregion