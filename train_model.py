import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from generate_dataset import ensure_dataset
from model_utils import build_model, save_model

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

print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nClassification Report:\n")
print(classification_report(y_test, predictions, zero_division=0))

# Retrain on all available demonstration data before saving.
model.fit(X, y)
model_path = save_model(model)

print(f"\nModel saved as {model_path}")

sample_resume = """
Machine Learning student skilled in Python, SQL, pandas, statistics,
scikit-learn and predictive modeling. Built classification projects
and analyzed datasets to solve real-world problems.
"""

prediction = model.predict([sample_resume])[0]
confidence = float(model.predict_proba([sample_resume]).max())

print("\nSample Resume Prediction:", prediction)
print("Confidence:", f"{confidence:.2%}")
