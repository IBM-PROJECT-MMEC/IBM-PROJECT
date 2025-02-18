# 📊 Bias Detection and Mitigation in AI Models

## 📌 Project Overview
This project focuses on detecting and mitigating bias in AI training datasets. Our approach includes **exploratory data analysis (EDA), data cleaning, feature engineering, and model selection** to build fair and unbiased machine learning models.

## 📅 Phase 2: Solution Architecture
This phase emphasizes **data exploration, visualization, and feature transformation** before deploying any models.

## 🏫 College Name
**Maratha Mandal Engineering College**

## 👥 Team Members & Contributions
- **Mohammed Sayeel Shigganvi (CAN ID: CAN_33860603)**  
  _Data cleaning strategies, API design, model research, and selection._  
- **Abdussuban Shaboddin Patel (CAN ID: CAN_34000109)**  
  _Functional requirements, tools/platform selection, and evaluation metrics._  
- **Mohammed Shaibaj Shaikh (CAN ID: CAN_33990553)**  
  _Visualization tools, feature engineering, and model evaluation._  
- **Tufailahmed M Bargir (CAN ID: CAN_34002247)**  
  _Reporting frameworks, dashboard design, and final documentation._  

---

## 🚀 Key Objectives
✅ Develop **visualizations** to identify bias trends and imbalances.  
✅ Use **EDA** to gain insights that guide model selection.  
✅ Establish **bias mitigation hypotheses** for future model training.  

---

## 🧹 1. Data Cleaning and Preparation
### 📌 Handling Missing Values
- **Numerical Data:** Imputed using the **median** to handle outliers.
- **Categorical Data:** Assigned a placeholder `"Unknown"` for missing categories.

```python
import pandas as pd

# Load dataset
data = pd.read_csv("training_data.csv")

# Impute numerical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())

# Impute categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].fillna('Unknown')
