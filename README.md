# AI Resume Classification System

<div align="center">

## Machine Learning-Based Resume Classification

A beginner-friendly end-to-end NLP project that takes resume text as input and predicts the job domain it most closely matches.

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange?logo=scikitlearn)](https://scikit-learn.org/)
[![Project](https://img.shields.io/badge/Project-Learning%20Prototype-success)](#limitations)

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [How the System Works](#how-the-system-works)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation and Usage](#installation-and-usage)
- [Dataset](#dataset)
- [Model Evaluation](#model-evaluation)
- [Example Prediction](#example-prediction)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [What I Learned](#what-i-learned)

---

# Project Overview

Resumes contain useful information about a person's skills, education, tools, projects, and work experience. When there are many resumes, manually grouping them into broad job domains can become repetitive.

This project explores a simple Machine Learning solution for that problem.

The system reads **resume text**, converts the text into numerical features using **TF-IDF**, and then uses a **Logistic Regression classifier** to predict the most relevant category.

The current version supports four categories:

| Category | Examples of Relevant Skills |
|---|---|
| **Data Science** | Python, pandas, SQL, statistics, machine learning |
| **Software Engineering** | C++, Java, algorithms, systems, software development |
| **Web Development** | HTML, CSS, JavaScript, React, frontend/backend |
| **Finance** | Accounting, investment, financial analysis, economics |

The goal of this project is not to automate hiring decisions. It is to demonstrate how **text classification** can be applied to resume data.

---

# Problem Statement

Suppose we have a collection of resumes belonging to different professional domains.

For example:

> Resume A mentions Python, machine learning, pandas, and data analysis.

> Resume B mentions React, JavaScript, HTML, and CSS.

Instead of manually deciding the category every time, we can train a model using previously labeled examples.

The model learns the relationship between:

```text
Words and skills in a resume
            ↓
Patterns learned from training data
            ↓
Predicted job category
```

This project implements that workflow from dataset preparation to prediction.

---

# How the System Works

The complete process can be understood in five steps:

```text
                    ┌─────────────────┐
                    │   Resume Text   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Text Processing │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ TF-IDF Features │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Logistic        │
                    │ Regression      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Predicted       │
                    │ Category        │
                    └─────────────────┘
```

---

# Machine Learning Pipeline

## 1. Resume Text

The input is plain text extracted or provided from a resume.

Example:

```text
Machine Learning student skilled in Python, SQL, pandas,
statistics and scikit-learn. Built classification models
and worked on data analysis projects.
```

At this stage, the computer cannot directly understand the meaning of the text. The text must first be converted into numbers.

---

## 2. TF-IDF Vectorization

The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)**.

In simple terms, TF-IDF helps the model identify which words are important.

For example:

- A word such as **Python** may be useful when identifying Data Science resumes.
- **React** may strongly indicate Web Development.
- **Accounting** may indicate Finance.

TF-IDF converts these words and phrases into numerical values that can be used by a Machine Learning model.

The project uses:

```python
TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
```

Using `ngram_range=(1, 2)` allows the model to consider both:

- Single words, such as `machine`
- Two-word phrases, such as `machine learning`

---

## 3. Train-Test Split

The dataset is divided into two parts:

- **Training data** — used to teach the model.
- **Testing data** — used to evaluate the model on examples it did not train on.

The split is performed using:

```python
train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
```

This means approximately 75% of the data is used for training and 25% for testing.

---

## 4. Logistic Regression

The classifier used in this project is **Logistic Regression**.

Despite its name, Logistic Regression is commonly used for classification tasks.

The model receives the numerical TF-IDF features and learns which patterns are associated with each category.

For a new resume, the model can estimate probabilities such as:

```text
Data Science          → 92%
Software Engineering  →  4%
Web Development       →  2%
Finance               →  2%
```

The category with the highest probability becomes the prediction.

---

## 5. Final Prediction

Once training is complete, the model is saved locally.

The prediction script allows a user to paste new resume text:

```bash
python predict_resume.py
```

The system then returns:

- The predicted category
- The model's confidence score

---

# Technologies Used

| Technology | Role in the Project |
|---|---|
| **Python** | Main programming language |
| **pandas** | Reading and handling the dataset |
| **scikit-learn** | Machine Learning tools and pipeline |
| **TF-IDF** | Converting text into numerical features |
| **Logistic Regression** | Classification algorithm |
| **joblib** | Saving and loading the trained model |

---

# Project Structure

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

## File Explanation

### `train_model.py`

This is the main training script.

It:

1. Loads the dataset.
2. Recreates the demonstration dataset if it is missing.
3. Splits the data into training and testing sets.
4. Builds the TF-IDF + Logistic Regression pipeline.
5. Trains the model.
6. Evaluates its performance.
7. Saves the trained model.

---

### `predict_resume.py`

This script is used after the model has been trained.

The user pastes resume text, and the script returns the predicted category and confidence.

---

### `generate_dataset.py`

This file recreates the demonstration dataset used by the project when the dataset file is not available.

This makes the repository easier to run without requiring manual dataset setup.

---

### `requirements.txt`

Contains the Python libraries required to run the project.

---

# Installation and Usage

## Step 1: Clone the Repository

```bash
git clone https://github.com/chamanvashishth/AI-Resume-Classification-System.git
cd AI-Resume-Classification-System
```

---

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

The required packages are:

- pandas
- scikit-learn
- joblib

---

## Step 3: Train the Model

Run:

```bash
python train_model.py
```

During training, the program will:

```text
Load Dataset
     ↓
Split Data
     ↓
Convert Text Using TF-IDF
     ↓
Train Logistic Regression Model
     ↓
Evaluate Performance
     ↓
Save Trained Model
```

After successful training, a model file named:

```text
resume_classifier.joblib
```

is created.

---

## Step 4: Predict a New Resume

Run:

```bash
python predict_resume.py
```

Paste resume text when prompted.

Example:

```text
Experienced in Python, machine learning, SQL, data analysis,
pandas and predictive modeling.
```

---

# Dataset

The current demonstration dataset contains:

- **140 resume samples**
- **4 categories**
- Labeled examples for supervised learning

The categories are:

```text
Data Science
Software Engineering
Web Development
Finance
```

The dataset is intentionally designed for demonstrating the Machine Learning workflow.

> **Important:** This is a learning and prototype dataset. It should not be considered representative of all real-world resumes.

---

# Model Evaluation

The current local experiment achieved:

```text
Accuracy: 1.00
```

At first glance, this looks perfect. However, this result needs to be interpreted carefully.

## Why This Does Not Automatically Mean the Model Is Perfect

The dataset is:

- Small
- Structured
- Limited to four categories
- Built for demonstration purposes

Because the categories contain relatively distinct vocabulary, the model can separate them more easily than it could separate genuinely diverse real-world resumes.

Therefore:

> **The current accuracy demonstrates that the pipeline works on this dataset. It does not prove production-level performance.**

A stronger evaluation would use:

- More resume samples
- Real-world variation in writing styles
- More overlapping job roles
- Independent validation data
- Class imbalance analysis
- Precision, recall, and F1-score comparisons
- Bias and fairness testing

---

# Example Prediction

### Input

```text
Machine Learning student skilled in Python, SQL, pandas,
statistics, scikit-learn and predictive modeling.
Built classification projects and analyzed datasets.
```

### Processing

```text
Resume Text
     ↓
TF-IDF extracts important terms
     ↓
Logistic Regression evaluates learned patterns
     ↓
Category probabilities are calculated
```

### Expected Output

```text
Predicted Category: Data Science
Confidence: XX.XX%
```

The exact confidence may change depending on the trained model and dataset.

---

# Limitations

This project is currently a **Machine Learning prototype**.

Some important limitations are:

### 1. Small Dataset

A dataset with 140 samples is useful for learning but is not sufficient for a production-grade classification system.

### 2. Limited Categories

Real job markets contain many overlapping roles. Four categories simplify the classification problem.

### 3. Resume Diversity

Real resumes vary significantly in:

- Writing style
- Formatting
- Experience level
- Skills
- Terminology

The demonstration dataset cannot capture all of this variation.

### 4. Hiring Should Not Be Automated

A model like this should not independently decide whether a candidate should be hired or rejected.

Resume classification can be affected by biased training data and should be used carefully, with appropriate human oversight.

---

# Future Improvements

The next versions of this project could include:

- [ ] A larger and more diverse dataset
- [ ] Additional job categories
- [ ] Better text preprocessing
- [ ] Comparison with Naive Bayes and Support Vector Machines
- [ ] Confusion matrix visualization
- [ ] Precision, recall, and F1-score charts
- [ ] Streamlit web interface
- [ ] Resume file upload support
- [ ] PDF/DOCX text extraction
- [ ] Transformer-based NLP models
- [ ] Bias and fairness evaluation
- [ ] Deployment as a web application

---

# What I Learned

While building this project, I worked with:

- Supervised Machine Learning
- Text classification
- Natural Language Processing basics
- TF-IDF vectorization
- Logistic Regression
- Train-test splitting
- Model evaluation
- Classification probabilities
- Saving and loading trained models
- Structuring a complete Machine Learning project

---

# Author

**Chaman Vashishth**

- Portfolio: [chamanvashishth.github.io](https://chamanvashishth.github.io/)
- GitHub: [@chamanvashishth](https://github.com/chamanvashishth)

---

# Final Note

This project focuses on understanding how a complete text-classification workflow can be built using Python and scikit-learn.

It is a useful starting point for learning how text data moves from:

```text
Raw Resume Text
      ↓
Numerical Features
      ↓
Machine Learning Model
      ↓
Predicted Category
```

<div align="center">

### ⭐ If you found this project useful, consider starring the repository.

Built as part of my Machine Learning learning journey.

</div>
