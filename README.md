# AI Resume Classification System

## Problem

Recruiters may receive many resumes for different job roles. Reading and sorting every resume manually can take time.

This project demonstrates a Machine Learning approach that classifies resume text into four job-related categories:

- Data Science
- Software Engineering
- Web Development
- Finance

## How it works

Resume text → TF-IDF → Logistic Regression → Predicted category

TF-IDF converts resume text into numerical features. Logistic Regression learns patterns from the training examples and predicts the category of a new resume.

## Project structure

- `train_model.py` — trains and evaluates the classifier
- `predict_resume.py` — accepts new resume text and predicts its category
- `generate_dataset.py` — recreates the demonstration dataset included with the project
- `requirements.txt` — required Python packages

The training script automatically recreates `resume_dataset.csv` if it is missing.

## Run locally

```bash
git clone https://github.com/chamanvashishth/AI-Resume-Classification-System.git
cd AI-Resume-Classification-System

pip install -r requirements.txt
python train_model.py
python predict_resume.py
```

## Model evaluation

The current demonstration dataset contains 140 samples across four categories. In the local test run, the model achieved 1.00 accuracy on the held-out split.

That number should **not** be interpreted as production performance. The dataset is small and highly structured, so the evaluation is useful for demonstrating the workflow rather than proving real-world reliability.

## Important limitation

The included dataset is intended for learning and prototyping. A practical resume classification system should use a much larger, representative, legally obtained dataset and should be evaluated for class imbalance, bias, fairness, and performance on genuinely unseen resumes.

## Resume description

Built a resume classification system using Python, TF-IDF, and Logistic Regression to categorize resume text into job domains including Data Science, Software Engineering, Web Development, and Finance.