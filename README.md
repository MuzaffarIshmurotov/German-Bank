# Credit Scoring with Logistic Regression 🎯

A complete implementation of **Logistic Regression** for credit scoring and risk assessment using the German Credit Dataset. This project demonstrates the end-to-end workflow from data preparation to model evaluation with professional visualizations.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)

## 📋 Table of Contents

- [Overview](#overview)
- [Algorithm](#algorithm)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Visualizations](#visualizations)
- [Results](#results)
- [Key Concepts](#key-concepts)
- [License](#license)

## 🎯 Overview

This project implements **Logistic Regression** to predict whether a loan applicant will have good or bad credit based on 20 features including age, credit amount, employment status, checking account status, and more.

**Goal:** Build a credit scoring model that can help banks assess loan application risk.

## 🤖 Algorithm: Logistic Regression

### What is Logistic Regression?

Logistic Regression is a **linear classification algorithm** that predicts probabilities using the sigmoid function.

**Key Components:**
1. **Linear Combination:** z = w₁×x₁ + w₂×x₂ + ... + wₙ×xₙ + b
2. **Sigmoid Function:** σ(z) = 1 / (1 + e^(-z)) → converts to probability [0, 1]
3. **Log Loss:** Measures prediction error
4. **Optimization:** Gradient descent to find best weights

### Why Logistic Regression for Credit Scoring?

✅ **Interpretable** - Can see which features increase/decrease credit risk  
✅ **Probability estimates** - Gives confidence scores (e.g., 85% good credit)  
✅ **Fast** - Trains in seconds, predicts instantly  
✅ **Regulatory compliance** - Banks can explain decisions  
✅ **Industry standard** - Used as baseline in real credit scoring  

### When to Use It

- ✅ When interpretability is crucial
- ✅ As a baseline before trying complex models
- ✅ When you need to explain decisions to stakeholders
- ✅ When training data is limited
- ✅ For regulatory compliance (explainable AI)

### Limitations

- ❌ Assumes linear relationship between features and target
- ❌ Cannot capture complex non-linear patterns
- ❌ Sensitive to feature scaling
- ❌ May underperform on highly non-linear data

## 📊 Dataset

**German Credit Dataset** from UCI Machine Learning Repository

- **Source:** 1,000 real loan applications from German banks
- **Features:** 20 (mix of numerical and categorical)
- **Target:** Binary classification
  - 1 = Good Credit (70% of data)
  - 0 = Bad Credit (30% of data)

### Feature Categories

**Numerical Features (7):**
- `age` - Age in years
- `credit_amount` - Loan amount requested
- `duration` - Loan duration in months
- `installment_rate` - Installment rate as % of income
- `residence_since` - Years at current residence
- `existing_credits` - Number of existing credits
- `num_dependents` - Number of people depending on applicant

**Categorical Features (13):**
- `checking_status` - Checking account status
- `credit_history` - Credit payment history
- `purpose` - Purpose of loan (car, furniture, etc.)
- `savings_status` - Savings account status
- `employment` - Employment status
- And 8 more...

### Class Imbalance

⚠️ **Important:** 70% good credit, 30% bad credit (imbalanced)

This is realistic! In real banking, most loans are repaid.

## ✨ Features

What this project includes:

- ✅ Complete data preprocessing pipeline
- ✅ Categorical feature encoding (text → numbers)
- ✅ Train/test split with stratification (maintains 70-30 ratio)
- ✅ Feature scaling with StandardScaler (critical for Logistic Regression!)
- ✅ Model training with scikit-learn
- ✅ Comprehensive evaluation metrics
- ✅ 8 professional visualizations
- ✅ Feature importance analysis
- ✅ Clean, well-commented code
- ✅ Step-by-step explanations

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Clone the Repository
```bash
git clone https://github.com/yourusername/credit-scoring-ml.git
cd credit-scoring-ml
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Requirements File
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 💻 Usage

### Quick Start

Run the complete Logistic Regression model:
```bash
python logistic_regression_simple.py
```

### Create All Visualizations
```bash
python create_all_visualizations.py
```

### Step-by-Step Learning
```bash
# Step 1: Load data
python step1_load_data.py

# Step 2: Prepare features
python step2_prepare_data.py
```

### Example Code

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

# Load data
df = pd.read_csv('german_credit.csv')

# Encode categorical features
categorical_features = ['checking_status', 'credit_history', 'purpose', ...]
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Prepare data
X = df.drop('target', axis=1)
y = df['target']

# Split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Scale features (CRITICAL for Logistic Regression!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
```

## 📁 Project Structure

```
credit-scoring-ml/
│
├── data/
│   └── german_credit.csv              # Dataset
│
├── src/
│   ├── step1_load_data.py             # Data loading tutorial
│   ├── step2_prepare_data.py          # Data preprocessing
│   ├── logistic_regression_simple.py  # Main model (CLEAN VERSION)
│   └── create_all_visualizations.py   # Create all charts
│
├── visualizations/
│   ├── viz1_data_distribution.png     # Age, amount, duration distributions
│   ├── viz2_age_vs_amount.png         # Scatter plot
│   ├── viz3_correlation.png           # Correlation heatmap
│   ├── viz4_confusion_matrix.png      # Prediction matrix
│   ├── viz5_roc_curve.png             # ROC curve with AUC
│   ├── viz6_feature_importance.png    # Top features
│   ├── viz7_probability_dist.png      # Probability distribution
│   └── viz8_performance_summary.png   # All metrics
│
├── explanations/
│   ├── explain_random_seed.py         # Why np.random.seed(42)?
│   ├── explain_no_headers.py          # Why header=None?
│   ├── explain_feature_names.py       # Why feature names?
│   └── explain_try_except.py          # Why try-except?
│
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── LICENSE                             # MIT License
└── .gitignore                          # Files to ignore
```

## 📊 Visualizations

### 1. Data Distribution
Shows distributions of age, credit amount, duration, and class balance.

### 2. Age vs Credit Amount
Scatter plot showing relationship between age and loan amount, colored by credit quality.

### 3. Correlation Heatmap
Feature correlation matrix showing which features are related.

### 4. Confusion Matrix
Visual representation of correct vs incorrect predictions.

### 5. ROC Curve
Shows model's ability to discriminate between good and bad credit.

### 6. Feature Importance
Top 10 most influential features (positive and negative coefficients).

### 7. Probability Distribution
How confident the model is in its predictions.

### 8. Performance Summary
All metrics (Accuracy, Precision, Recall, F1, AUC) in one chart.

## 📈 Results

### Model Performance

| Metric | Train | Test | Description |
|--------|-------|------|-------------|
| **Accuracy** | 0.7186 | 0.7167 | 71.67% correct predictions |
| **Precision** | 0.7182 | 0.7167 | When predicting good, correct 71.67% |
| **Recall** | 1.0000 | 1.0000 | Catches 100% of good credits |
| **F1-Score** | 0.8360 | 0.8350 | Harmonic mean of precision & recall |
| **AUC-ROC** | 0.6070 | 0.5675 | Discrimination power |
| **Gini** | 0.2140 | 0.1350 | Industry standard (2×AUC - 1) |

### Confusion Matrix (Test Set)

```
              Predicted
           Bad    Good
Actual Bad   0     85     ← Missed all bad credits
      Good   0    215     ← Caught all good credits
```

### Key Findings

✅ **Good at predicting good credits** - 100% recall  
❌ **Poor at predicting bad credits** - 0% recall  
⚠️ **Class imbalance problem** - Model predicts "good" for everyone  
📊 **Gini = 0.135** - Below industry standard (wants >0.3)  

### Top 5 Most Important Features

| Rank | Feature | Coefficient | Impact |
|------|---------|-------------|--------|
| 1 | existing_credits | +0.191 | Positive (increases good credit probability) |
| 2 | property_magnitude | +0.122 | Positive |
| 3 | duration | -0.107 | Negative (longer loans = higher risk) |
| 4 | foreign_worker | +0.103 | Positive |
| 5 | savings_status | +0.094 | Positive |

**Interpretation:**
- **Positive coefficient** = Feature increases probability of good credit
- **Negative coefficient** = Feature decreases probability of good credit

## 🎓 Key Concepts

### Why Feature Scaling Matters

Logistic Regression is **sensitive to feature magnitude**:

```python
# Before scaling:
credit_amount: 250 - 18,500  (huge range!)
age: 19 - 76                 (small range)
# Result: credit_amount dominates the model!

# After scaling (mean=0, std=1):
credit_amount: -1.5 to 2.5
age: -1.5 to 2.5
# Result: All features have equal importance!
```

### Stratified Split

Maintains class ratio in both train and test sets:

```python
# Without stratify:
train_test_split(X, y)
# Might get: train=68% good, test=73% good (random!)

# With stratify:
train_test_split(X, y, stratify=y)
# Always gets: train=70% good, test=70% good (consistent!)
```

### Gini Coefficient (Industry Standard)

```python
Gini = 2 × AUC - 1
```

**Interpretation:**
- **0.0 - 0.2:** Poor discrimination
- **0.2 - 0.3:** Fair discrimination
- **0.3 - 0.4:** Good discrimination ⭐ (industry standard)
- **0.4 - 0.5:** Excellent discrimination
- **0.5+:** Outstanding discrimination

**Our Result:** Gini = 0.135 (Fair, needs improvement)

### Handling Class Imbalance

**The Problem:**
- 70% good credit, 30% bad credit
- Model learns: "Just predict good for everyone → 70% accuracy!"

**Solutions:**
1. Use better metrics (AUC, not accuracy)
2. Apply SMOTE (synthetic minority oversampling)
3. Use class weights in model
4. Adjust decision threshold
5. Try ensemble methods

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Improve documentation
- Add new visualizations

## 📝 License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2024 Ishmurat

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

## 🙏 Acknowledgments

- **Dataset:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- **Original Data:** Professor Dr. Hans Hofmann, University of Hamburg
- **Libraries:** scikit-learn, pandas, matplotlib, seaborn
- **Inspiration:** Real-world credit risk assessment in banking

## 📚 Learn More

- [Logistic Regression Explained](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Credit Scoring Overview](https://en.wikipedia.org/wiki/Credit_score)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

⭐ **Star this repo** if you found it helpful!

🐛 **Found a bug?** [Open an issue](https://github.com/yourusername/credit-scoring-ml/issues)

💡 **Questions?** Feel free to reach out!

**Made with ❤️ for ML learners and aspiring data scientists**
