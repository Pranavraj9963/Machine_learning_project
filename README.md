# 🏦 Loan Risk Intelligence System

A Machine Learning project I built to predict loan approval using customer financial data — with model explainability using SHAP.

---

## About the Project

I wanted to build something beyond just "train a model and get accuracy" — so this project covers the full pipeline: from generating a realistic banking dataset to explaining *why* the model made a particular prediction (using SHAP).

The idea is simple — given a customer's financial and personal details, predict whether their loan should be approved or rejected, and be able to explain that decision.

---

## Dataset

I generated a custom banking dataset with **25,750 records** and **37 features** — things like income, credit score, existing loans, EMI ratio, savings, credit utilization, etc.

**Target column:** `LoanApproved` (0 = Rejected, 1 = Approved)

---

## What I Did

- Cleaned and preprocessed the raw data
- Did EDA (distributions, correlations, outliers)
- Engineered new features like `DebtToIncomeRatio`, `EMIRatio`, `Networth`
- Selected important features using Mutual Information & Random Forest importance
- Trained and compared 9 ML models
- Tuned the best models using RandomizedSearchCV
- Used **SHAP** to explain model predictions (global + local)

---

## Models & Results

| Model | Accuracy |
|--------|----------|
| AdaBoost | 98.73% |
| Random Forest | 98.51% |
| Decision Tree | 98.37% |
| Extra Trees | 95.39% |
| SVM | 88.59% |
| Logistic Regression | 88.18% |
| KNN | 81.50% |
| Naive Bayes | 74.74% |

AdaBoost gave me the best results, closely followed by Random Forest.

---

## Key Insights (from SHAP)

The features that mattered most for approval decisions:

- ExistingLoans, MissedEMIs, LatePayments
- CreditScore, MonthlyIncome, AnnualIncome
- DebtToIncomeRatio, EMIRatio, SavingRatio

---

## Tech Stack

Python · NumPy · Pandas · Matplotlib · Seaborn · Scikit-learn · SHAP · Jupyter Notebook

---

## Project Structure

```
Loan_Risk_Intelligence/
├── data/
├── notebooks/          # step-by-step notebooks (01 to 10)
├── models/
├── reports/
├── requirements.txt
└── README.md
```

---

## What's Next

- Build a Streamlit app for live predictions
- Add a proper dashboard for loan risk
- Try deep learning models
- Deploy it on the cloud
