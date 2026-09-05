from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
import joblib

from generate_dataset import ensure_dataset

DATASET_PATH = ensure_dataset()
MODEL_PATH = Path("resume_classifier.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run 'python train_model.py' first."
    )

df = pd.read_csv(DATASET_PATH)

_, X_test, _, y_test = train_test_split(
    df["resume_text"],
    df["category"],
    test_size=0.25,
    random_state=42,
    stratify=df["category"],
)

model = joblib.load(MODEL_PATH)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))

ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.title("Resume Classification Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion matrix as confusion_matrix.png")
