"""Utilities for building, training, loading, and saving the resume classifier."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from generate_dataset import ensure_dataset

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "resume_classifier.joblib"


def build_model() -> Pipeline:
    """Create the TF-IDF + Logistic Regression classification pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=2000),
            ),
        ]
    )


def train_model() -> Pipeline:
    """Train a new classifier using the demonstration dataset."""
    dataset_path = ensure_dataset()
    df = pd.read_csv(dataset_path)

    required_columns = {"resume_text", "category"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain 'resume_text' and 'category' columns."
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    model = build_model()
    model.fit(df["resume_text"].astype(str), df["category"].astype(str))
    return model


def save_model(model: Pipeline) -> Path:
    """Save a trained model for local or command-line use."""
    joblib.dump(model, MODEL_PATH)
    return MODEL_PATH


def load_or_train_model(persist: bool = False) -> Pipeline:
    """
    Load a valid saved model when available.

    If the model is missing or cannot be loaded, train a fresh model instead.
    The Streamlit app uses persist=False so deployment does not depend on
    write access to the application directory.
    """
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            # A corrupted or incompatible serialized model should not prevent
            # the application from starting.
            pass

    model = train_model()

    if persist:
        try:
            save_model(model)
        except OSError:
            # The model can still be used in memory when storage is unavailable.
            pass

    return model
