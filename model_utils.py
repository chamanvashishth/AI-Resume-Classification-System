"""Shared utilities for training and loading the resume classifier."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from generate_dataset import ensure_dataset

MODEL_PATH = Path("resume_classifier.joblib")


def build_model() -> Pipeline:
    """Create the TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                ),
            ),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )


def train_model() -> Pipeline:
    """Train the classifier on the bundled demonstration dataset."""
    dataset_path = ensure_dataset()
    df = pd.read_csv(dataset_path)

    required_columns = {"resume_text", "category"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain 'resume_text' and 'category' columns."
        )

    model = build_model()
    model.fit(df["resume_text"], df["category"])

    joblib.dump(model, MODEL_PATH)
    return model


def load_or_train_model() -> Pipeline:
    """Load a trained model, or train one automatically if it is missing."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    return train_model()
