# Telco Customer Churn Prediction: Baseline vs. Ensemble Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://neuro-five-telco-customer-churn-prediction-lbcwjbkckrgzeuobaxx.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red?style=for-the-badge&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

---

## Executive Summary

Customer churn prediction is a critical machine learning application across telecom, banking, and SaaS industries. This repository contains an end-to-end Machine Learning pipeline utilizing the **Telco Customer Churn dataset**. 

The project evaluates predictive performance across linear baselines (**Logistic Regression**) and advanced Ensemble Learning methods (**Random Forest** & **XGBoost**). Special emphasis is placed on solving target class imbalance using **SMOTE** and **Cost-Sensitive Class Weighting**, culminating in a **fully deployed interactive web app** on Streamlit Cloud.

---

## Project Objectives

* **Exploratory Data Analysis (EDA):** Identify primary drivers of customer attrition across demographics and services.
* **Data Engineering:** Build clean scaling, encoding, and imputation pipelines preventing data leakage.
* **Imbalance Mitigation:** Resolve class imbalance (73.46% vs 26.54%) to optimize sensitivity (Recall).
* **Model Benchmarking:** Benchmark Baseline Logistic Regression against Bagging (Random Forest) and Boosting (XGBoost).
* **Interpretability & Insights:** Extract feature importances to deliver actionable business retention strategies.
* **Model Deployment:** Export trained artifacts and build an interactive web interface for real-time risk assessment.

---

## Dataset Architecture

* **Dataset Source:** Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
* **Total Instances:** 7,043 customer records
* **Target Distribution:**
  * **Non-Churn (0):** 4,139 samples (**73.46%**)
  * **Churn (1):** 1,495 samples (**26.54%**)

---

## Data Preprocessing Pipeline

1. **Identifier Removal:** Dropped non-predictive `customerID` attributes.
2. **Data Cleaning & Imputation:** Converted whitespace strings in `TotalCharges` into floating-point numbers and imputed missing values with median values.
3. **Target Encoding:** Standardized `Churn` status from binary labels (`Yes`/`No`) to numeric format (`1`/`0`).
4. **Categorical Feature Encoding:** Applied One-Hot Encoding (`pd.get_dummies()`) to non-ordinal categorical variables.
5. **Feature Scaling:** Applied `StandardScaler` to normalize numeric distributions (`tenure`, `MonthlyCharges`, `TotalCharges`) for linear model convergence.
6. **Data Partitioning:** Executed an 80/20 stratified train-test split to preserve target class proportions.

---

## Handling Class Imbalance & The Accuracy Paradox

### Empirical Comparison: Resampling vs Cost-Sensitive Weighting

| Model / Strategy | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Original Imbalanced Data)** | **0.8070** | **0.6584** | 0.5668 | 0.6092 | **0.8416** |
| **SMOTE Oversampling** | 0.7381 | 0.5042 | **0.7968** | **0.6176** | 0.8403 |
| **Class Weighting (`balanced`)** | 0.7388 | 0.5052 | 0.7807 | 0.6134 | 0.8412 |

### Theoretical Analysis
* **The Accuracy Paradox:** In an imbalanced dataset (73.46% Non-Churn), a trivial classifier predicting "No Churn" for all customers yields **73.46% Accuracy** while completely failing to identify at-risk customers.
* **Business Priority:** In customer retention, **False Negatives** (failing to identify a churner) directly result in lost revenue. Thus, **Recall** is prioritized over raw Accuracy.
* **Outcome:** Applying **SMOTE** increased Recall from **56.68% to 79.68%** (capturing nearly 80% of all potential churners) with a net gain in the overall **F1 Score** (`0.6092` $\rightarrow$ `0.6176`).

---

## Model Benchmark: Baseline vs Ensemble Methods

Model evaluation on the unseen 20% test partition across default settings:

| Model | Architecture Type | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear Baseline | **0.8070** | **0.6584** | **0.5668** | **0.6092** | **0.8416** |
| **Random Forest** | Bagging (Parallel Ensemble) | 0.7864 | 0.6237 | 0.4920 | 0.5501 | 0.8251 |
| **XGBoost** | Boosting (Sequential Ensemble) | 0.7984 | 0.6406 | 0.5481 | 0.5908 | 0.8388 |

---

## Algorithmic Deep-Dive: Bagging vs Gradient Boosting

* **Random Forest (Bagging):** Fits multiple deep decision trees independently in parallel on bootstrap samples. Variance is reduced through feature random subsampling and majority voting.
* **XGBoost (Gradient Boosting):** Fits shallow decision trees sequentially. Each new tree explicitly minimizes the residual loss function of prior trees using gradient descent optimization. Built-in $L_1$ (Lasso) and $L_2$ (Ridge) regularization parameters actively prevent overfitting.

---

## Feature Importance Analysis

Comparative feature analysis reveals distinct learning behaviors between model families:

* **Random Forest Focus (Continuous Numerical Attributes):**
  1. `TotalCharges`
  2. `MonthlyCharges`
  3. `tenure`
* **XGBoost Focus (Risk Flags & Contract Metadata):**
  1. `Contract_Month-to-month`
  2. `InternetService_Fiber optic`
  3. `PaymentMethod_Electronic check`

---

## 🚀 Interactive Web Application & Deployment

The optimal model (`Logistic Regression + SMOTE`) was packaged alongside its scaling parameters and feature alignment vectors into serialized joblib artifacts to power an interactive **Streamlit** web application.

* **Live Application:** [Launch Telco Churn Predictor](https://neuro-five-telco-customer-churn-prediction-lbcwjbkckrgzeuobaxx.streamlit.app/)
* **Features:**
  * Real-time churn probability scoring based on custom user inputs.
  * Automated feature encoding and feature alignment pipeline.
  * Actionable risk mitigation indicators for retention teams.

---

## Strategic Business Recommendations

1. **Contract Strategy:** Offer targeted multi-month promotional discounts to transition high-risk **Month-to-month** customers to 1-Year or 2-Year contracts.
2. **Service Improvement:** Conduct service audit on **Fiber Optic** subscriptions, as Fiber Optic users exhibit disproportionately higher churn rates relative to DSL subscribers.
3. **Early Tenure Onboarding:** Design proactive customer success programs during the first **6 to 12 months** of customer tenure, where attrition risk is highest.

---

## Repository Structure

```text
├── dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── app.py                            # Streamlit web interface code
├── model.joblib                      # Serialized trained Logistic Regression model
├── scaler.joblib                     # Serialized StandardScaler instance
├── feature_columns.joblib            # Feature schema for input alignment
├── Telco_Customer_Churn_Ensemble.ipynb # End-to-end model analysis notebook
├── requirements.txt                  # Application dependency specification
└── README.md                         # Project documentation
