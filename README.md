# 📊 Telco Customer Churn Prediction

## 📌 Project Overview

Customer churn prediction is one of the most common machine learning applications used in the telecom, banking, insurance, and SaaS industries. This project analyzes customer behavior using the Telco Customer Churn dataset and builds machine learning models to predict whether a customer is likely to leave the company.

---

## 🎯 Objectives

- Perform Exploratory Data Analysis (EDA)
- Identify factors influencing customer churn
- Preprocess and encode the dataset
- Train and compare Logistic Regression and Decision Tree models
- Identify the most important features affecting churn
- Provide business insights based on model predictions

---

## 📂 Dataset

**Dataset:** Telco Customer Churn

The dataset contains customer demographic information, subscribed services, contract details, billing information, and churn status.

Target Variable:

- **Churn**
  - 0 = No
  - 1 = Yes

---

## 🔍 Exploratory Data Analysis (EDA)

The following analyses were performed:

- Dataset overview
- Missing value analysis
- Duplicate value analysis
- Data type inspection
- Numerical vs Categorical feature analysis
- Target variable distribution
- Univariate analysis
- Bivariate analysis
- Churn analysis using important business features

---

## ⚙️ Data Preprocessing

The preprocessing pipeline included:

- Removed unnecessary customerID column
- Converted TotalCharges to numeric format
- Encoded target variable (Churn)
- Applied One-Hot Encoding to categorical variables
- Standardized numerical features
- Split data into training and testing sets

---

## 🤖 Machine Learning Models

Two supervised learning models were trained:

- Logistic Regression
- Decision Tree Classifier

---

## 📈 Model Performance

| Model | Accuracy |
|--------|----------|
| Logistic Regression | **82.11%** |
| Decision Tree | **71.19%** |

Logistic Regression achieved better performance across Accuracy, Precision, Recall, and F1-score.

---

## ⭐ Top 3 Important Features

According to the Decision Tree model:

1. MonthlyCharges
2. tenure
3. TotalCharges

---

## 💼 Business Insights

- Customers with month-to-month contracts are more likely to churn.
- Customers with shorter tenure have a higher probability of leaving.
- Higher monthly charges are associated with increased churn.
- Electronic check users exhibit higher churn rates.
- Long-term contracts significantly improve customer retention.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 📁 Project Structure

```
Telco_Customer_Churn.ipynb
README.md
requirements.txt
dataset/
images/
```

---

## 🚀 Future Improvements

- Hyperparameter tuning
- Random Forest and XGBoost comparison
- ROC-AUC analysis
- Cross-validation
- SMOTE for class imbalance
- Model deployment using Flask or Streamlit

---

## 👨‍💻 Author

**Ali Anus**

Computer Science Student

Machine Learning | Data Analytics | Artificial Intelligence
