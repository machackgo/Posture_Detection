import cv2
import csv
import os
import mediapipe as mp
from posture_features import extract_features

# Where to save collected rows.
# You can override from terminal, e.g.:
#   POSTUREPAL_OUT_CSV=data/posture_dataset_test_session.csv python collect_data.py
OUT_CSV = os.getenv("POSTUREPAL_OUT_CSV", "data/real_world.csv")

# If Continuity Camera (iPhone) hijacks index 0, you can force an index, e.g.:
#   export POSTUREPAL_CAMERA_INDEX=1
PREFERRED_CAMERA_INDEX = os.getenv("POSTUREPAL_CAMERA_INDEX")

LABEL_MAP = {
    "g": 0,  # Good
    "s": 1,  # Slight Slouch
    "h": 2,  # Heavy Slouch
    "l": 3,  # Lean Left
    "r": 4,  # Lean Right
}


def _open_camera():
    # macOS: AVFoundation is most reliable.
    indices = [0, 1, 2, 3]

    # Try a preferred index first if provided.
    if PREFERRED_CAMERA_INDEX is not None:
        try:
            pref = int(PREFERRED_CAMERA_INDEX)
            indices = [pref] + [i for i in indices if i != pref]
        except ValueError:
            pass

    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            continue

        # reduce resolution to avoid stalls
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # warm-up reads (prevents first-frame hang sometimes)
        ok = False
        for _ in range(10):
            ok, _ = cap.read()
        if ok:
            print(f"Using camera index {idx}")
            return cap

        cap.release()

    return None


def main():
    os.makedirs("data", exist_ok=True)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=0,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = _open_camera()
    if cap is None:
        raise RuntimeError(
            "Could not open camera (tried indices 0-3). Close Zoom/Meet/FaceTime and allow Terminal (System Settings → Privacy & Security → Camera). If iPhone Continuity Camera is interfering, disconnect it or set POSTUREPAL_CAMERA_INDEX. "
        )

    # Force window creation (macOS reliability)
    cv2.namedWindow("Collect Posture Data", cv2.WINDOW_NORMAL)

    file_exists = os.path.exists(OUT_CSV) and os.path.getsize(OUT_CSV) > 0

    writer = None
    feature_names = None

    with open(OUT_CSV, "a", newline="") as f:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Show camera FIRST so window always appears
            cv2.putText(
                frame,
                "Click this window. Press g/s/h/l/r to save. Press q to quit.",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Collect Posture Data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            k = chr(key) if key != 255 else None
            pending_label = LABEL_MAP.get(k, None)

            # MediaPipe AFTER window shows
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = pose.process(rgb)
            rgb.flags.writeable = True

            feats = extract_features(res.pose_landmarks, mp_pose)

            if pending_label is not None and feats is not None:
                if writer is None:
                    feature_names = list(feats.keys())
                    writer = csv.DictWriter(f, fieldnames=feature_names + ["label"])
                    if not file_exists:
                        writer.writeheader()
                        file_exists = True

                row = dict(feats)
                row["label"] = pending_label
                writer.writerow(row)
                f.flush()
                print("Saved:", row)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()