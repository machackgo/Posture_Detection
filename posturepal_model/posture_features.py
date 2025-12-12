import math
import numpy as np

def _pt(lm):
    # lm: mediapipe landmark with x,y,z,visibility
    return np.array([lm.x, lm.y, getattr(lm, "z", 0.0)], dtype=np.float32)

def _angle_deg(v1, v2, eps=1e-8):
    n1 = np.linalg.norm(v1) + eps
    n2 = np.linalg.norm(v2) + eps
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))

def extract_features(pose_landmarks, mp_pose):
    """
    Returns: dict of numeric features or None if insufficient confidence.
    """
    if pose_landmarks is None:
        return None

    lm = pose_landmarks.landmark

    # Required keypoints
    LSH = _pt(lm[mp_pose.PoseLandmark.LEFT_SHOULDER])
    RSH = _pt(lm[mp_pose.PoseLandmark.RIGHT_SHOULDER])
    LHP = _pt(lm[mp_pose.PoseLandmark.LEFT_HIP])
    RHP = _pt(lm[mp_pose.PoseLandmark.RIGHT_HIP])
    NOSE = _pt(lm[mp_pose.PoseLandmark.NOSE])

    # Midpoints
    MSH = (LSH + RSH) / 2.0
    MHP = (LHP + RHP) / 2.0

    torso_vec = MSH - MHP
    neck_vec = NOSE - MSH

    # Reference vertical axis in image coords: y grows downward, so "up" is negative y
    vertical_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    torso_angle = _angle_deg(torso_vec[:2], vertical_up[:2])  # 2D
    neck_angle = _angle_deg(neck_vec[:2], torso_vec[:2])      # 2D

    shoulder_width = float(np.linalg.norm((LSH - RSH)[:2]) + 1e-8)
    torso_len = float(np.linalg.norm(torso_vec[:2]) + 1e-8)

    # Tilt: positive if right shoulder is lower than left (y is larger)
    shoulder_tilt = float((RSH[1] - LSH[1]) / shoulder_width)
    hip_tilt = float((RHP[1] - LHP[1]) / (float(np.linalg.norm((LHP - RHP)[:2]) + 1e-8)))

    # Head forward proxy: nose z relative to shoulders (works better than pure 2D for “too close”)
    head_forward_z = float(NOSE[2] - MSH[2])

    head_to_shoulder = float(np.linalg.norm((NOSE - MSH)[:2]) / torso_len)

    return {
        "torso_angle": torso_angle,
        "neck_angle": neck_angle,
        "shoulder_tilt": shoulder_tilt,
        "hip_tilt": hip_tilt,
        "head_forward_z": head_forward_z,
        "head_to_shoulder": head_to_shoulder,
    }