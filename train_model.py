from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DATASET_PATH = Path("resume_dataset.csv")

if not DATASET_PATH.exists():
    from generate_dataset import ensure_dataset

    ensure_dataset(DATASET_PATH)

df = pd.read_csv(DATASET_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    df["resume_text"],
    df["category"],
    test_size=0.25,
    random_state=42,
    stratify=df["category"],
)

model = Pipeline(
    [
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=2000)),
    ]
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

joblib.dump(model, "resume_classifier.joblib")
print("\nModel saved as resume_classifier.joblib")

sample_resume = """
Machine Learning student skilled in Python, SQL, pandas, statistics,
scikit-learn and predictive modeling. Built classification projects
and analyzed datasets to solve real-world problems.
"""

prediction = model.predict([sample_resume])[0]
probabilities = model.predict_proba([sample_resume])[0]

print("\nSample Resume Prediction:", prediction)
for label, probability in sorted(
    zip(model.classes_, probabilities), key=lambda item: -item[1]
):
    print(f"{label}: {probability:.2%}")
