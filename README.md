# Posture_Detection (PosturePal) 🧍‍♂️📷

**PosturePal** is an end-to-end posture detection project that performs **live webcam inference** to detect posture issues (e.g., leaning) and provide feedback in real time.  
This repo contains the trained model artifacts and the inference pipeline used for the demo.

---

## Project Overview

Many people develop poor posture while studying/working for long hours. This project aims to detect posture changes using a webcam feed and a lightweight ML pipeline.

**Workflow:**
1. Capture webcam frames
2. Extract features from the frame (vision-based feature extraction)
3. Run the trained model for posture classification
4. Display feedback based on the predicted posture

---

## Features

- ✅ Real-time webcam posture detection
- ✅ Posture classification (e.g., neutral vs leaning)
- ✅ Live feedback when posture changes
- ✅ Simple Python-based setup

---

## Tech Stack

- **Python**
- **OpenCV** (camera + frame processing)
- **Machine Learning** (classification model)
- *(Optional: NumPy / scikit-learn depending on implementation)*

---

## Repository Structure

Posture_Detection/
└─ posturepal_model/
├─ (model files / weights)
├─ (inference scripts)
└─ (supporting utilities)

---

## How to Run (General)

> Exact filenames can differ based on your setup, but the typical workflow is:

1) Create and activate a virtual environment
bash
python3 -m venv .venv
source .venv/bin/activate

2.	Install dependencies
pip install -U pip
pip install opencv-python numpy
# (install other dependencies if your project requires them)

3.	Run the main script (webcam inference)
python <your_main_file>.py

Notes / Future Improvements
	•	Add more posture classes (slouch, forward head posture, etc.)
	•	Improve robustness across different lighting and camera angles
	•	Add a UI overlay with better visual guidance
	•	Create a lightweight mobile/edge version

Author

Mohammed Mubashir Uddin Faraz
GitHub: https://github.com/machackgo
