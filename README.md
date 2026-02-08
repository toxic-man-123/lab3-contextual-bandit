# Lab 3: Contextual Bandit for News Recommendation

**Name:** Aryan Gosain  
**Roll Number:** U20230108

> **Submission Scope:** This submission covers Sections 5.1 and 5.2 – Data Pre-processing and User Classification (Context Detection).

---

## Overview

This project implements a **Contextual Bandit** system for personalized news article recommendations. This submission focuses on the foundational components: preprocessing user data and building a classifier to detect user contexts.

### Problem Statement
Given user behavioral data, the goal is to:
1. **Preprocess the data** – Handle missing values, encode features, and prepare for modeling
2. **Classify users** into one of three categories (User1, User2, User3) based on their features

---

## How to Run

### Prerequisites
- Python 3.10+
- Jupyter Notebook or JupyterLab

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd lab3-contextual-bandit

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn matplotlib
```

### Running the Notebook

```bash
# Activate the virtual environment
source venv/bin/activate

# Launch Jupyter
jupyter notebook lab3_results_U20230108.ipynb
```

Run all cells from top to bottom to reproduce the results.

---

## Approach

### 5.1 Data Pre-processing

**Datasets Used:**
- `train_users.csv` – 2,000 labeled user records with 33 features
- `test_users.csv` – 2,000 unlabeled user records with 32 features
- `news_articles.csv` – 209,527 news articles with 6 features

**Pre-processing Steps:**
1. **Duplicate Removal** – Removed duplicate records from all datasets
2. **Missing Value Handling** – Filled missing values in the `age` column (698 missing) using median imputation
3. **Feature Encoding:**
   - **Numeric features (28):** Standardized using `StandardScaler`
   - **Categorical features (3):** One-hot encoded (`browser_version`, `region_code`, `subscriber`)
4. **Final Feature Matrix:** 130 features after encoding

### 5.2 User Classification (Context Detection)

A **Logistic Regression** classifier was trained to predict user categories:

| Configuration | Value |
|--------------|-------|
| Train/Validation Split | 80/20 |
| Stratification | Yes (maintained class proportions) |
| Max Iterations | 2000 |
| Random State | 42 |

---

## Results

### Classification Performance on Validation Set

| User Category | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| user_1 | 0.8378 | 0.8732 | 0.8552 | 142 |
| user_2 | 1.0000 | 0.8169 | 0.8992 | 142 |
| user_3 | 0.8529 | 1.0000 | 0.9206 | 116 |
| **Overall Accuracy** | | | **0.8900** | 400 |

### Test Set Context Distribution

After retraining on the full training set, the classifier predicted the following distribution on `test_users.csv`:

| Context | User Category | Count |
|---------|--------------|-------|
| 0 | User1 | 609 |
| 1 | User2 | 666 |
| 2 | User3 | 725 |

---

## Insights

1. **High Classification Accuracy:** Logistic Regression achieved **89% accuracy** on the validation set, so it shows that user categories are well-separable based on behavioral features.

2. **Class-wise Performance:**
   - **User2** has perfect precision (1.0) but lower recall (0.82) – the model is conservative in predicting this class
   - **User3** has perfect recall (1.0) – all User3 instances are correctly identified
   - **User1** shows balanced precision-recall trade-off

3. **Missing Data Handling:** The `age` feature had ~35% missing values; median imputation was chosen over mean to be robust against potential outliers.

4. **Feature Expansion:** One-hot encoding expanded the feature space from 31 to 130 dimensions, capturing categorical nuances effectively.

---

## Repository Structure

```
lab3-contextual-bandit/
├── README.md                          # This file
├── lab3_results_U20230108.ipynb       # Main notebook with all code and results
├── assignment.pdf                     # Lab assignment specification
├── data/
│   ├── train_users.csv                # Labeled training user data
│   ├── test_users.csv                 # Unlabeled test user data
│   └── news_articles.csv              # News articles dataset
└── venv/                              # Python virtual environment
```

---

*Last Updated: February 2026*
