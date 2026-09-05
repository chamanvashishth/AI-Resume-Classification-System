# AI Resume Classification System

<div align="center">

## Machine Learning-Based Resume Classification

A beginner-friendly end-to-end NLP project that analyzes resume text and predicts the job domain it most closely matches.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)](https://streamlit.io/)

</div>

---

## Overview

Resumes contain information about skills, education, projects, and experience. When many resumes need to be organized into broad job domains, manually sorting them can become repetitive.

This project demonstrates a complete Machine Learning workflow for **resume text classification**.

The system:

1. Takes resume text as input.
2. Converts the text into numerical features using **TF-IDF**.
3. Uses **Logistic Regression** to learn patterns from labeled examples.
4. Predicts the most likely job category.
5. Displays the prediction and confidence score.

### Supported Categories

- Data Science
- Software Engineering
- Web Development
- Finance

---

## System Architecture

```text
                    Resume Text
                         │
                         ▼
                ┌─────────────────┐
                │ TF-IDF          │
                │ Vectorization   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Logistic        │
                │ Regression      │
                └────────┬────────┘
                         │
                         ▼
                Predicted Category
                         │
                         ▼
                 Confidence Score
```

---

## Machine Learning Workflow

### 1. Resume Text

The model receives plain text from a resume.

Example:

```text
Machine Learning student skilled in Python, SQL, pandas,
statistics, scikit-learn and predictive modeling.
```

### 2. TF-IDF Vectorization

Machine Learning models work with numbers rather than raw text.

**TF-IDF (Term Frequency–Inverse Document Frequency)** converts important words and phrases into numerical features.

For example:

- Python, pandas, and machine learning may signal Data Science.
- React and JavaScript may signal Web Development.
- Accounting and investment may signal Finance.

The project considers both individual words and two-word phrases:

```python
TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
```

### 3. Train-Test Split

The dataset is divided into:

- **Training data** — used to learn patterns.
- **Testing data** — used to evaluate the model.

The project uses a 75/25 split with stratification so category proportions are preserved.

### 4. Logistic Regression

Logistic Regression is used as the classification model.

It learns relationships between TF-IDF features and known categories. For a new resume, it estimates probabilities for each category and selects the category with the highest probability.

---

## Project Structure

```text
AI-Resume-Classification-System/
│
├── app.py                    # Streamlit web interface
├── model_utils.py            # Shared model build/train/load utilities
├── train_model.py            # Training and evaluation script
├── predict_resume.py         # Command-line prediction
├── evaluate_model.py         # Classification report + confusion matrix
├── generate_dataset.py       # Recreates the demonstration dataset
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## Deployment Reliability

The app no longer depends on `resume_classifier.joblib` being committed to GitHub.

When the application starts:

```text
Saved model available?
        │
   ┌────┴────┐
   │ Yes     │ No
   ▼         ▼
Load model   Recreate demo dataset
             ↓
             Train model in memory
             ↓
             Cache model for the running app
             ↓
             Classify resume
```

The implementation also handles a corrupted or incompatible saved model by training a fresh model instead.

This makes the Streamlit deployment independent of generated `.joblib` files and avoids the original `FileNotFoundError`.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/chamanvashishth/AI-Resume-Classification-System.git
cd AI-Resume-Classification-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model locally

```bash
python train_model.py
```

This creates:

```text
resume_classifier.joblib
```

### 4. Run command-line prediction

```bash
python predict_resume.py
```

### 5. Run the web application

```bash
streamlit run app.py
```

The web app can run even when the trained model file does not exist because it can train the demonstration model automatically.

---

## Evaluation

For additional evaluation:

```bash
python evaluate_model.py
```

This prints a classification report and generates:

```text
confusion_matrix.png
```

---

## Dataset

The current demonstration dataset contains:

- **140 samples**
- **4 categories**
- Labeled examples for supervised learning

The dataset is recreated from the repository's embedded demonstration data when it is missing.

---

## Important Note About Accuracy

The current local experiment achieved **1.00 accuracy** on the held-out split.

This should **not** be interpreted as proof of production-level performance.

The demonstration dataset is small and structured, which makes the categories easier to separate than genuinely diverse real-world resumes.

A stronger system would require:

- A larger and more representative dataset
- More job categories
- Independent validation data
- Precision, recall, and F1-score analysis
- Bias and fairness evaluation

---

## Limitations

This project is a learning-focused prototype.

- The dataset is small.
- Only four broad categories are supported.
- Real resumes contain much more variation.
- High accuracy on this dataset may not generalize.
- The system should not be used to automatically accept or reject job candidates.

---

## Future Improvements

- [x] Add a Streamlit interface
- [x] Add automatic model recovery when the saved model is missing
- [x] Add detailed evaluation support
- [ ] Support PDF and DOCX resume uploads
- [ ] Extract text automatically from uploaded resumes
- [ ] Compare multiple classification models
- [ ] Add a larger dataset
- [ ] Add more job categories
- [ ] Deploy the application
- [ ] Explore transformer-based NLP models
- [ ] Evaluate bias and fairness

---

## What I Learned

This project helped me work with:

- Supervised Machine Learning
- Text classification
- NLP fundamentals
- TF-IDF vectorization
- Logistic Regression
- Train-test splitting
- Classification probabilities
- Model persistence with joblib
- Model evaluation
- Streamlit application development

---

## Author

**Chaman Vashishth**

- Portfolio: [chamanvashishth.github.io](https://chamanvashishth.github.io/)
- GitHub: [@chamanvashishth](https://github.com/chamanvashishth)

---

<div align="center">

Built as part of my Machine Learning learning journey.

</div>
