import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

from generate_dataset import ensure_dataset
from model_utils import build_model

dataset_path = ensure_dataset()
df = pd.read_csv(dataset_path)

X_train, X_test, y_train, y_test = train_test_split(
    df["resume_text"],
    df["category"],
    test_size=0.25,
    random_state=42,
    stratify=df["category"],
)

model = build_model()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))

ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.title("Resume Classification Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion matrix as confusion_matrix.png")
