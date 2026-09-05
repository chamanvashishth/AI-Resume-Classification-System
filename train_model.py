from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from generate_dataset import ensure_dataset
from model_utils import MODEL_PATH, build_model

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

print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# Retrain on the full demonstration dataset before saving for deployment.
model.fit(df["resume_text"], df["category"])
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved as {MODEL_PATH}")

sample_resume = """
Machine Learning student skilled in Python, SQL, pandas, statistics,
scikit-learn and predictive modeling. Built classification projects
and analyzed datasets to solve real-world problems.
"""

prediction = model.predict([sample_resume])[0]
probabilities = model.predict_proba([sample_resume])[0]

print("\nSample Resume Prediction:", prediction)
for label, probability in sorted(
    zip(model.classes_, probabilities),
    key=lambda item: -item[1],
):
    print(f"{label}: {probability:.2%}")
