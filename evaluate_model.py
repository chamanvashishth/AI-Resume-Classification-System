from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

from generate_dataset import ensure_dataset
from model_utils import BASE_DIR, build_model

dataset_path = ensure_dataset()
df = pd.read_csv(dataset_path)

X = df["resume_text"].astype(str)
y = df["category"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

model = build_model()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions, zero_division=0))

output_path = BASE_DIR / "confusion_matrix.png"

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    xticks_rotation=20,
)
plt.title("Resume Classification Confusion Matrix")
plt.tight_layout()
plt.savefig(output_path, dpi=150)
plt.close()

print(f"Saved confusion matrix as {output_path}")
