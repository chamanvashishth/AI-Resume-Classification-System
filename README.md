# AI Resume Classification System

<div align="center">

### Machine Learning-based Resume Category Classification

A simple end-to-end NLP project that analyzes resume text and predicts the most relevant job category.

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/Status-Learning%20Project-success.svg)](#important-notes)

</div>

---

## Overview

Recruiters often need to review resumes from candidates applying to different roles. This project explores how **Machine Learning and Natural Language Processing** can help organize resume text by automatically predicting a job category.

The system currently classifies resumes into:

- Data Science
- Software Engineering
- Web Development
- Finance

---

## How It Works

```text
Resume Text
     │
     ▼
TF-IDF Vectorization
     │
     ▼
Logistic Regression
     │
     ▼
Predicted Job Category
```

### Pipeline

1. Resume text is provided as input.
2. **TF-IDF** converts the text into numerical features.
3. A **Logistic Regression** model learns patterns from labeled examples.
4. The trained model predicts the category of a new resume.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **pandas** | Dataset handling |
| **scikit-learn** | Machine learning pipeline |
| **TF-IDF** | Text feature extraction |
| **Logistic Regression** | Resume classification |
| **joblib** | Model serialization |

---

## Project Structure

```text
AI-Resume-Classification-System/
│
├── README.md
├── train_model.py
├── predict_resume.py
├── generate_dataset.py
├── requirements.txt
└── .gitignore
```

### File Description

| File | Description |
|---|---|
| `train_model.py` | Trains and evaluates the classification model |
| `predict_resume.py` | Predicts the category of new resume text |
| `generate_dataset.py` | Recreates the demonstration dataset when needed |
| `requirements.txt` | Lists required Python packages |
| `.gitignore` | Excludes generated and environment files from Git |

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/chamanvashishth/AI-Resume-Classification-System.git
cd AI-Resume-Classification-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train_model.py
```

The script:

- Loads the demonstration dataset
- Creates the dataset automatically if it is missing
- Splits the data into training and testing sets
- Builds the TF-IDF + Logistic Regression pipeline
- Evaluates the model
- Saves the trained model as `resume_classifier.joblib`

### 4. Predict a Resume Category

After training the model:

```bash
python predict_resume.py
```

Paste resume text when prompted, and the system returns:

- Predicted category
- Prediction confidence

---

## Example

### Input

```text
Machine Learning student skilled in Python, SQL, pandas,
statistics, scikit-learn and predictive modeling.
Built classification projects and analyzed datasets.
```

### Output

```text
Predicted Category: Data Science
Confidence: XX.XX%
```

---

## Dataset

The current demonstration dataset contains:

- **140 samples**
- **4 job categories**

It is designed to demonstrate the complete Machine Learning workflow from text preprocessing to prediction.

> The dataset is intentionally small and should be treated as a learning/prototyping dataset rather than a production dataset.

---

## Model Evaluation

In the current local experiment, the model achieved:

```text
Accuracy: 1.00
```

### Important Context

This result should **not** be interpreted as real-world production performance.

The current dataset is relatively small and structured, which can make the classification task easier than working with genuinely diverse resumes. A realistic evaluation would require:

- A larger dataset
- More diverse resume formats
- Additional job categories
- Independent validation data
- Bias and fairness evaluation

---

## Limitations

This project is a learning-focused prototype and has several limitations:

- Limited dataset size
- Only four categories
- Resume text may not reflect real-world diversity
- High accuracy on this dataset may not generalize to unseen resumes
- The system should not be used as an automated hiring decision tool

---

## Future Improvements

- [ ] Add more resume categories
- [ ] Use a larger and more diverse dataset
- [ ] Improve text preprocessing
- [ ] Compare multiple machine learning models
- [ ] Add confusion matrix and detailed visual evaluation
- [ ] Build a Streamlit web interface
- [ ] Add resume file upload support
- [ ] Explore transformer-based NLP models
- [ ] Add fairness and bias evaluation

---

## Key Learning Outcomes

Through this project, I worked with:

- Text classification
- Natural Language Processing basics
- TF-IDF vectorization
- Logistic Regression
- Train-test splitting
- Model evaluation
- Prediction probabilities
- Building an end-to-end Machine Learning workflow

---

## Author

**Chaman Vashishth**

- Portfolio: [chamanvashishth.github.io](https://chamanvashishth.github.io/)
- GitHub: [@chamanvashishth](https://github.com/chamanvashishth)

---

## Important Notes

This repository is created for **learning, experimentation, and demonstrating a Machine Learning workflow**.

Automated resume classification can introduce bias and should not be treated as a replacement for human judgment in real hiring decisions.

---

<div align="center">

⭐ If you found this project useful, consider starring the repository.

**Built as part of my Machine Learning learning journey.**

</div>
