import json
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "data/posture_dataset.csv"

if not os.path.exists(DATA_PATH) or os.path.getsize(DATA_PATH) == 0:
    raise SystemExit(
        "Dataset not found or empty: data/posture_dataset.csv\n"
        "Run: python collect_data.py\n"
        "Then CLICK the camera window and press g/s/h/l/r to save rows. Press q to quit."
    )

LABEL_NAMES = {
    0: "Good",
    1: "Slight Slouch",
    2: "Heavy Slouch",
    3: "Lean Left",
    4: "Lean Right",
}

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["label"])
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        max_depth=None
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
    print("\nReport:\n", classification_report(y_test, pred, target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/posture_rf.joblib")

    meta = {
        "features": list(X.columns),
        "label_names": LABEL_NAMES,
        "model_type": "RandomForestClassifier"
    }
    with open("models/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved: models/posture_rf.joblib and models/meta.json")

if __name__ == "__main__":
    main()