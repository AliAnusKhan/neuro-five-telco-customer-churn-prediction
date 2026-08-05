# 📊 Telco Customer Churn Prediction: Baseline vs. Ensemble Learning

## 📌 Project Overview

Customer churn prediction is one of the most critical machine learning applications in telecom, banking, insurance, and SaaS industries. This project analyzes customer behavior using the Telco Customer Churn dataset and builds supervised machine learning models — ranging from a single linear baseline to advanced Ensemble methods (Bagging & Boosting) — to predict whether a customer is likely to leave the company.

---

## 🎯 Objectives

- Perform Exploratory Data Analysis (EDA) on customer retention patterns.
- Preprocess, clean, and encode non-numeric and missing variables.
- Train and compare a Baseline Model (**Logistic Regression**) against Ensemble Models (**Random Forest** & **XGBoost**).
- Analyze side-by-side feature importances to determine key churn drivers.
- Provide theoretical and practical insights into Bagging vs. Gradient Boosting methodologies.

---

## 📂 Dataset

**Dataset:** Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)

The dataset contains customer demographic information, subscribed services, contract details, billing information, and churn status.

- **Target Variable:** `Churn`
  - `0` = No (Retained)
  - `1` = Yes (Churned)

---

## ⚙️ Data Preprocessing Pipeline

- **Column Dropping:** Removed non-informative `customerID` identifiers.
- **Type Conversion & Imputation:** Coerced whitespace-filled `TotalCharges` into floating-point numbers and imputed missing entries with median values.
- **Target Encoding:** Mapped `Churn` labels (`Yes`/`No`) into binary format (`1`/`0`).
- **Feature Transformation:** Applied One-Hot Encoding to categorical variables using `pd.get_dummies()`.
- **Feature Scaling:** Applied `StandardScaler` to normalize numeric distributions for Logistic Regression convergence.
- **Train-Test Split:** Created 80% Train and 20% Test stratified splits to maintain class balance.

---

## 🤖 Machine Learning Models Evaluated

1. **Logistic Regression (Baseline):** Linear model standardizing input features.
2. **Random Forest Classifier (Ensemble - Bagging):** Parallel tree ensemble built on bootstrap samples.
3. **XGBoost Classifier (Ensemble - Boosting):** Sequential gradient boosted decision tree model optimizing residual errors.

---

## 📈 Model Performance Comparison

Evaluating models on the unseen 20% test partition yielded the following results:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | **0.8070** | **0.6584** | **0.5668** | **0.6092** | **0.8416** |
| **Random Forest (Bagging)** | 0.7864 | 0.6237 | 0.4920 | 0.5501 | 0.8251 |
| **XGBoost (Boosting)** | 0.7984 | 0.6406 | 0.5481 | 0.5908 | 0.8388 |

---

## ⚙️ Algorithmic Differences: Random Forest vs. XGBoost

Random Forest and XGBoost differ fundamentally in how they combine decision trees to make predictions. **Random Forest** uses **Bagging (Bootstrap Aggregation)**, where multiple deep decision trees are trained independently and in parallel on random bootstrap subsets of data and features, averaging predictions to reduce variance. In contrast, **XGBoost** uses **Gradient Boosting**, constructing shallow decision trees sequentially where each subsequent tree explicitly minimizes the residual errors of preceding trees using gradient descent optimization. While Random Forest focuses primarily on reducing model variance, XGBoost optimizes both bias and variance through built-in $L_1$ and $L_2$ regularization mechanisms.

---

## ⭐ Important Feature Insights

Side-by-side feature importance comparisons reveal key operational findings:

- **Random Forest:** Gives maximum weight to numerical financial metrics:
  1. `TotalCharges`
  2. `MonthlyCharges`
  3. `tenure`
- **XGBoost:** Gives maximum weight to contract and service risk flags:
  1. `Contract_Month-to-month`
  2. `InternetService_Fiber optic`
  3. `PaymentMethod_Electronic check`

---

## 💼 Business Recommendations

- **Contract Incentives:** Promote long-term (1-Year / 2-Year) contracts with discounts, as month-to-month customers exhibit the highest churn probability.
- **Service Quality:** Investigate Fiber Optic service issues, as Fiber Optic subscribers show disproportionately higher churn rates compared to DSL.
- **Onboarding Support:** Focus retention programs on new customers within their first 6–12 months of tenure.

---

## 🛠️ Technologies Used

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-Learn
- XGBoost
- Google Colab & Jupyter Notebook

---

## 📁 Project Structure

```text
├── Telco_Customer_Churn_Ensemble.ipynb
├── README.md
├── requirements.txt
└── dataset/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
