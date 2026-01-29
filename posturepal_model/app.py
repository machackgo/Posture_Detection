import gradio as gr
import numpy as np
from PIL import Image

import mediapipe as mp

# Import your existing code (no rewriting model)
from infer_runtime import PostureModel, _compute_features_from_landmarks

pm = PostureModel()

# MediaPipe helpers
_mp_pose = mp.solutions.pose
_mp_drawing = mp.solutions.drawing_utils


def _to_rgb_numpy(img):
    """Accepts PIL or numpy and returns RGB uint8 numpy array."""
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    arr = np.array(img)
    # Gradio webcam/upload usually provides RGB already.
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, :3].astype(np.uint8)
    return arr.astype(np.uint8)


def predict_from_image(img):
    """
    img: from Gradio (webcam snapshot or upload)
    returns: (prediction_text, confidence, annotated_image)
    """
    rgb = _to_rgb_numpy(img)
    if rgb is None:
        return "No image received (if using webcam: click the camera button to capture, then Submit)", 0.0, None

    # MediaPipe Pose on a single image
    pose = _mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
    )

    res = pose.process(rgb)
    pose.close()

    annotated = rgb.copy()

    if res.pose_landmarks is None:
        return "No pose detected (make sure upper body is visible + good lighting)", 0.0, annotated

    # Draw landmarks for visual confirmation
    _mp_drawing.draw_landmarks(
        annotated,
        res.pose_landmarks,
        _mp_pose.POSE_CONNECTIONS,
    )

    feats = _compute_features_from_landmarks(res.pose_landmarks.landmark)
    out = pm.predict(feats)

    label = out.get("label", "Unknown")
    conf = float(out.get("confidence", 0.0))

    return label, conf, annotated


with gr.Blocks(title="PosturePal (Local) — Posture Detection") as demo:
    gr.Markdown("# PosturePal (Local) — Posture Detection\nRuns MediaPipe + RandomForest locally (same machine as the app).")

    with gr.Row():
        img_in = gr.Image(
            label="Upload OR take a webcam snapshot (upper body visible)",
            type="numpy",
            sources=["upload", "webcam"],
        )
        with gr.Column():
            pred_out = gr.Textbox(label="Prediction")
            conf_out = gr.Number(label="Confidence")

    annotated_out = gr.Image(label="Pose overlay (debug)", type="numpy")

    btn = gr.Button("Submit")
    btn.click(fn=predict_from_image, inputs=[img_in], outputs=[pred_out, conf_out, annotated_out])

    gr.Markdown(
        "**Tip:** If you choose **Webcam**, click the small camera/shutter button to capture a frame, then press **Submit**."
    )


if __name__ == "__main__":
    demo.launch()