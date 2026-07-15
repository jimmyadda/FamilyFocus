import shutil
import uuid
from pathlib import Path

import cv2
from werkzeug.utils import secure_filename
from time import perf_counter


from config import MIN_FACE_SIZE_IMAGE, PHOTO_MATCH_THRESHOLD, TEMP_UPLOAD_DIR
from db_helpers import (
    create_family_photo,
    create_photo_detection,
    database_read,
    extract_faces_for_review,
    get_family_possible_dir,
    get_family_possible_faces_dir,
    get_family_results_dir,
    get_family_skipped_dir,
    get_incoming_upload_dir,
    save_learning_review,
)
from face_utils import (
    create_annotated_family_image,
    crop_face,
    detect_faces_with_opencv,
    identify_face,
)
from services.file_utils import allowed_file


def get_member_id_by_name(name, family_id):
    row = database_read(
        """
        SELECT id
        FROM family_members
        WHERE name = ?
          AND family_id = ?
        """,
        (name, family_id),
        one=True,
    )
    return row["id"] if row else None


def _box_iou(a, b):
    ax1 = a.get("x")
    ay1 = a.get("y")
    aw = a.get("width", a.get("w"))
    ah = a.get("height", a.get("h"))

    bx1 = b.get("x")
    by1 = b.get("y")
    bw = b.get("width", b.get("w"))
    bh = b.get("height", b.get("h"))

    if None in (ax1, ay1, aw, ah, bx1, by1, bw, bh):
        return 0.0

    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (aw * ah) + (bw * bh) - intersection
    return intersection / union if union else 0.0


def image_contains_known_family(
    image_path,
    family_embeddings,
    DeepFace,
    family_id,
):
    extract_started = perf_counter()

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read image:", image_path)
        return False, [], [], [], []

    faces = []

    try:
        faces = DeepFace.extract_faces(
            img_path=str(image_path),
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
        )

        print(
            "TIMING OpenCV detector:",
            round(perf_counter() - extract_started, 3),
            "seconds"
        )

    except Exception as error:
        print("OpenCV detector error:", repr(error))

    valid_faces = []

    for face in faces:
        area = face.get("facial_area", {})

        w = int(area.get("w", 0))
        h = int(area.get("h", 0))

        if w < MIN_FACE_SIZE_IMAGE or h < MIN_FACE_SIZE_IMAGE:
            continue

        valid_faces.append(face)

    faces = valid_faces

    if not faces:
        print("OpenCV found no valid faces. Falling back to RetinaFace.")

        fallback_started = perf_counter()

        try:
            faces = DeepFace.extract_faces(
                img_path=str(image_path),
                detector_backend="retinaface",
                enforce_detection=False,
                align=True,
            )

            print(
                "TIMING RetinaFace fallback:",
                round(perf_counter() - fallback_started, 3),
                "seconds"
            )

        except Exception as error:
            print("RetinaFace fallback failed:", repr(error))
            faces = []

    matches = []
    possible_matches = []
    matched_names = []
    possible_names = []
    possible_faces_dir = get_family_possible_faces_dir(family_id)

    print("FACES FOUND:", len(faces))

    for index, face_obj in enumerate(faces):
        area = face_obj.get("facial_area", {})
        x = int(area.get("x", 0))
        y = int(area.get("y", 0))
        w = int(area.get("w", area.get("width", 0)))
        h = int(area.get("h", area.get("height", 0)))

        if w < MIN_FACE_SIZE_IMAGE or h < MIN_FACE_SIZE_IMAGE:
            continue

        face_crop = face_obj.get("face")
        if face_crop is None:
            continue

        if face_crop.dtype != "uint8":
            face_crop = (face_crop * 255).astype("uint8")
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)

        temp_face_path = TEMP_UPLOAD_DIR / f"temp_filter_face_{uuid.uuid4()}.jpg"
        cv2.imwrite(str(temp_face_path), face_crop)

        try:
            embedding_started = perf_counter()

            member_name, distance, status = identify_face(
                temp_face_path,
                family_embeddings,
                DeepFace,
                threshold=PHOTO_MATCH_THRESHOLD,
            )
            print(
                f"TIMING face {index} embedding:",
                round(perf_counter() - embedding_started, 3),
                "seconds"
            )
        finally:
            temp_face_path.unlink(missing_ok=True)

        print(
            f"FACE {index} | name={member_name} | "
            f"distance={distance:.4f} | status={status}"
        )

        if member_name is None:
            continue

        face_crop_filename = None
        if status == "possible":
            face_crop_filename = f"{Path(image_path).stem}_face_{index}.jpg"
            cv2.imwrite(
                str(possible_faces_dir / face_crop_filename),
                face_crop,
            )

        match = {
            "match": True,
            "face_index": index,
            "name": member_name,
            "member_id": get_member_id_by_name(member_name, family_id),
            "box": {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "width": w,
                "height": h,
            },
            "distance": float(distance),
            "status": status,
            "face_crop": face_crop_filename,
        }

        if status == "matched":
            matches.append(match)
            if member_name not in matched_names:
                matched_names.append(member_name)
        elif status == "possible":
            possible_matches.append(match)
            if member_name not in possible_names:
                possible_names.append(member_name)

    return (
        bool(matches),
        matched_names,
        matches,
        possible_names,
        possible_matches,
    )


def save_unknown_faces_for_review(
    family_id,
    incoming_path,
    filename,
    known_matches,
    DeepFace,
):
    skipped_dir = get_family_skipped_dir(family_id)
    skipped_path = skipped_dir / filename

    if not skipped_path.exists():
        shutil.copy(str(incoming_path), str(skipped_path))

    unknown_faces = extract_faces_for_review(
        incoming_path,
        filename,
        DeepFace,
        family_id=family_id,
    )

    known_boxes = [
        match["box"]
        for match in known_matches
        if match.get("box")
    ]

    saved_unknown = 0
    for face in unknown_faces:
        face_box = {
            "x": face["box_x"],
            "y": face["box_y"],
            "width": face["box_w"],
            "height": face["box_h"],
        }

        if any(_box_iou(face_box, box) > 0.35 for box in known_boxes):
            continue

        save_learning_review(
            family_id=family_id,
            photo_path=f"skipped/{filename}",
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
            image_h=face["image_h"],
        )
        saved_unknown += 1

    return saved_unknown


def _build_path(path_builder, kind, filename):
    return path_builder(kind, filename) if path_builder else None


def scan_one_family_photo(
    file,
    family_id,
    family_embeddings,
    DeepFace,
    source="web",
    path_builder=None,
):
    """Run the shared web/Telegram family-photo detection pipeline."""
    scan_started = perf_counter()

    ensure_dirs = (
        get_incoming_upload_dir(family_id),
        get_family_results_dir(family_id),
        get_family_possible_dir(family_id),
        get_family_possible_faces_dir(family_id),
        get_family_skipped_dir(family_id),
    )
    for directory in ensure_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    if not file or not file.filename:
        return {"status": "skipped", "reason": "empty"}

    if not allowed_file(file.filename):
        return {"status": "skipped", "reason": "invalid_file"}

    original = secure_filename(file.filename)
    extension = original.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4()}.{extension}"
    incoming_path = get_incoming_upload_dir(family_id) / filename
    file.save(incoming_path)

    try:
        recognition_started = perf_counter()
        (
            found,
            matched_names,
            matches,
            _possible_names,
            possible_matches,
        ) = image_contains_known_family(
            incoming_path,
            family_embeddings,
            DeepFace,
            family_id,
        )
        print(
            "TIMING recognition:",
            round(perf_counter() - recognition_started, 3),
            "seconds"
        )
        if found:
            result_path = get_family_results_dir(family_id) / filename
            # Save the clean original image in the family album
            shutil.copy2(
                str(incoming_path),
                str(result_path)
            )
            #create_annotated_family_image(incoming_path, matches, result_path)

            photo_id = create_family_photo(
                family_id=family_id,
                file_path=result_path,
                original_filename=file.filename,
                source=source,
            )

            for match in matches:
                member_id = match.get("member_id")
                if member_id:
                    create_photo_detection(
                        family_id=family_id,
                        photo_id=photo_id,
                        member_id=member_id,
                        face_crop_path=None,
                        distance=match.get("distance"),
                        status="confirmed",
                        confirmed_by_user=0,
                    )

            unknown_count = save_unknown_faces_for_review(
                family_id,
                incoming_path,
                filename,
                matches,
                DeepFace,
            )

            return {
                "status": "matched",
                "filename": filename,
                "original": original,
                "confirmed_count": len(matched_names),
                "possible_count": 0,
                "skipped_count": unknown_count,
                "unknown_faces": unknown_count,
                "confirmed_names": matched_names,
                "possible_names": [],
                "names": matched_names,
                "matches": matches,
                "path": _build_path(path_builder, "result", filename),
            }

        if possible_matches:
            possible_path = get_family_possible_dir(family_id) / filename
            shutil.copy2(
                str(incoming_path),
                str(possible_path)
            )

            photo_id = create_family_photo(
                family_id=family_id,
                file_path=possible_path,
                original_filename=file.filename,
                source=source,
            )

            for match in possible_matches:
                member_id = match.get("member_id")
                if not member_id:
                    continue

                detection_id = create_photo_detection(
                    family_id=family_id,
                    photo_id=photo_id,
                    member_id=member_id,
                    face_crop_path=None,
                    distance=match.get("distance"),
                    status="possible",
                    confirmed_by_user=0,
                )

                saved = database_read(
                    """
                    SELECT status
                    FROM photo_detections
                    WHERE id = ?
                      AND family_id = ?
                    """,
                    (detection_id, family_id),
                    one=True,
                )
                match["saved_status"] = (
                    saved["status"] if saved else "unknown"
                )

            visible = [
                item
                for item in possible_matches
                if item.get("saved_status") == "possible"
            ]
            rejected = [
                item
                for item in possible_matches
                if item.get("saved_status") == "rejected"
            ]

            # Extract before incoming_path is deleted.
            unknown_count = save_unknown_faces_for_review(
                family_id,
                incoming_path,
                filename,
                possible_matches,
                DeepFace,
            )

            return {
                "status": "possible" if visible else "already_rejected",
                "filename": filename,
                "original": original,
                "confirmed_count": 0,
                "possible_count": len(visible),
                "rejected_count": len(rejected),
                "skipped_count": unknown_count,
                "unknown_faces": unknown_count,
                "confirmed_names": [],
                "possible_names": [item.get("name") for item in visible],
                "rejected_names": [item.get("name") for item in rejected],
                "names": [item.get("name") for item in visible],
                "matches": possible_matches,
                "path": _build_path(path_builder, "possible", filename),
            }

        skipped_path = get_family_skipped_dir(family_id) / filename
        shutil.copy(str(incoming_path), str(skipped_path))

        unknown_faces = extract_faces_for_review(
            incoming_path,
            filename,
            DeepFace,
            family_id=family_id,
        )

        for face in unknown_faces:
            save_learning_review(
                family_id=family_id,
                photo_path=f"skipped/{filename}",
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
                image_h=face["image_h"],
            )
        print(
                "TIMING complete scan:",
                round(perf_counter() - scan_started, 3),
                "seconds"
            )
        return {
            "status": "skipped",
            "filename": filename,
            "original": original,
            "confirmed_count": 0,
            "possible_count": 0,
            "unknown_faces": len(unknown_faces),
            "skipped_count": len(unknown_faces),
            "confirmed_names": [],
            "possible_names": [],
        }
    finally:
        incoming_path.unlink(missing_ok=True)
