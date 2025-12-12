import json
import joblib
import numpy as np

print("infer_runtime.py started")
class PostureModel:
    def __init__(self, model_path="models/posture_rf.joblib", meta_path="models/meta.json"):
        self.model = joblib.load(model_path)
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        self.features = self.meta["features"]
        self.label_names = {int(k): v for k, v in self.meta["label_names"].items()}
        self.label_to_id = {v: k for k, v in self.label_names.items()}

        # sklearn predict_proba returns probabilities aligned to model.classes_
        self.class_ids = [int(c) for c in getattr(self.model, "classes_", [])]
        self.id_to_index = {cid: i for i, cid in enumerate(self.class_ids)}

    def predict(self, feature_dict):
        x = np.array([[feature_dict[f] for f in self.features]], dtype=np.float32)
        proba = self.model.predict_proba(x)[0]

        best_idx = int(np.argmax(proba))
        # Use the actual class label from sklearn (not the index)
        cls = int(self.model.classes_[best_idx]) if hasattr(self.model, "classes_") else best_idx

        return {
            "class_id": cls,
            "label": self.label_names.get(cls, str(cls)),
            "confidence": float(proba[best_idx]),
            "proba": proba.tolist(),
        }


def _safe_angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """Angle between two 2D vectors in degrees."""
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    nu = float(np.linalg.norm(u) + 1e-8)
    nv = float(np.linalg.norm(v) + 1e-8)
    cosang = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _compute_features_from_landmarks(landmarks) -> dict:
    """Compute the exact feature keys expected by the trained model."""
    import mediapipe as mp

    PL = mp.solutions.pose.PoseLandmark

    def pt(i: int) -> np.ndarray:
        lm = landmarks[i]
        return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

    ls = pt(PL.LEFT_SHOULDER.value)
    rs = pt(PL.RIGHT_SHOULDER.value)
    lh = pt(PL.LEFT_HIP.value)
    rh = pt(PL.RIGHT_HIP.value)
    nose = pt(PL.NOSE.value)

    mid_sh = 0.5 * (ls + rs)
    mid_hip = 0.5 * (lh + rh)

    torso_vec = (mid_sh - mid_hip)
    head_vec = (nose - mid_sh)

    # Keep torso_angle in radians (matches the range you collected ~0..pi)
    # Angle from vertical (image coords: +y is down), measured in the x-y plane.
    torso_vec_xy = torso_vec[:2]
    torso_angle = float(np.abs(np.arctan2(torso_vec_xy[0], -torso_vec_xy[1])))

    # Neck angle in degrees between torso direction and head direction (x-y plane)
    neck_angle = _safe_angle_deg(torso_vec[:2], head_vec[:2])

    # Tilts: signed y-differences (small decimals like your dataset)
    shoulder_tilt = float(ls[1] - rs[1])
    hip_tilt = float(lh[1] - rh[1])

    # Head forward: relative depth (z). Your dataset shows negative values.
    head_forward_z = float(nose[2] - mid_sh[2])

    # Head-to-shoulder: distance in x-y (normalized coords)
    head_to_shoulder = float(np.linalg.norm((nose - mid_sh)[:2]))

    return {
        "torso_angle": torso_angle,
        "neck_angle": neck_angle,
        "shoulder_tilt": shoulder_tilt,
        "hip_tilt": hip_tilt,
        "head_forward_z": head_forward_z,
        "head_to_shoulder": head_to_shoulder,
    }


if __name__ == "__main__":
    import cv2
    import mediapipe as mp

    print("Starting live inference… (press 'q' to quit)")

    pm = PostureModel()

    # You are showing a mirrored (selfie) view using cv2.flip(..., 1).
    # In a mirrored view, left/right labels often look swapped to the user.
    MIRROR_VIEW = True
    SWAP_LEFT_RIGHT_LABELS = True  # set False if you want screen-left/screen-right instead

    # On macOS, AVFoundation is the correct backend.
    # Your scan showed index 0 works; others fail.
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open camera index 0. Close Zoom/Meet/FaceTime and ensure Terminal has Camera permission."
        )

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    win = "PosturePal (Live)"
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read a frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks is not None:
            feats = _compute_features_from_landmarks(res.pose_landmarks.landmark)
            out = pm.predict(feats)

            if MIRROR_VIEW and SWAP_LEFT_RIGHT_LABELS and out["label"] in ("Lean Left", "Lean Right"):
                swapped_label = "Lean Right" if out["label"] == "Lean Left" else "Lean Left"
                swapped_id = pm.label_to_id.get(swapped_label, out["class_id"])
                swapped_idx = pm.id_to_index.get(swapped_id, None)
                if swapped_idx is not None and swapped_idx < len(out["proba"]):
                    out["class_id"] = swapped_id
                    out["label"] = swapped_label
                    out["confidence"] = float(out["proba"][swapped_idx])
                else:
                    # Fallback: swap label only
                    out["label"] = swapped_label

            text = f"{out['label']}  conf={out['confidence']:.2f}"
        else:
            text = "No pose detected"

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    pose.close()
    cap.release()
    cv2.destroyAllWindows()