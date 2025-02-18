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
- ✅ Develop **visualizations** to identify bias trends and imbalances.  
- ✅ Use **EDA** to gain insights that guide model selection.  
- ✅ Establish **bias mitigation hypotheses** for future model training.  

---

## 🧹 1. Data Cleaning and Preparation

### 📌 Handling Missing Values
- **Numerical Data:** Imputed using the **median** to handle outliers.
- **Categorical Data:** Assigned a placeholder `"Unknown"` for missing categories.

Load dataset
data = pd.read_csv("training_data.csv")

Impute numerical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())

Impute categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].fillna('Unknown')

text

### 📌 Outlier Handling
- **Detection:** Boxplots and Z-score analysis.
- **Treatment:** Winsorization (capping extreme values) & exclusion of corrupted data.


Cap extreme values
data['Feature'] = np.clip(data['Feature'], data['Feature'].quantile(0.01), data['Feature'].quantile(0.99))

text

### 📌 Resolving Duplicates
- Duplicate records were removed.
- Logical inconsistencies in feature values were corrected.


Removing duplicates
data = data.drop_duplicates()

text

---

## 📊 2. Data Visualization

### Tools Used
- 📈 Matplotlib – Static plots
- 📊 Seaborn – Heatmaps and detailed visualizations
- 📉 Plotly – Interactive dashboards

### Key Visualizations
- ✅ Correlation Heatmaps – Show feature relationships and class imbalances.
- ✅ Scatterplots – Detect bias-related anomalies.
- ✅ Boxplots – Visualize feature distributions for imbalance detection.


Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

text

---

## 🤖 3. Model Research & Selection

### Techniques Evaluated
- Decision Trees – Detect key features influencing bias.
- Logistic Regression – Analyze linear bias patterns.
- Clustering Algorithms – Identify biased clusters.

### Final Model Choice
- ✅ Decision Trees – Robust and interpretable.
- ✅ Logistic Regression – Useful for fairness assessment.

---

## 🔧 4. Data Transformation & Feature Engineering

### Feature Scaling
- **Standardization:** Ensures uniform feature influence.
- **Min-Max Scaling:** Normalizes skewed features between 0 and 1.


Standardization
scaler = StandardScaler()
data['Standardized_Feature'] = scaler.fit_transform(data[['Feature']])

Min-Max Scaling
data['Scaled_Feature'] = MinMaxScaler().fit_transform(data[['Feature']])

text

### Encoding Categorical Variables
One-Hot Encoding for interpretability.

One-Hot Encoding
encoded_data = pd.get_dummies(data, columns=['CategoricalFeature'])

text

### Dimensionality Reduction (PCA)
PCA reduces dimensions while retaining 95% variance.

from sklearn.decomposition import PCA

pca = PCA(n_components=5)
data_pca = pca.fit_transform(data.drop(columns=['Target']))

text

---

## 📌 5. Bias Mitigation & Evaluation Metrics
- ✅ Fairness Metrics: Statistical Parity Difference, Equal Opportunity.
- ✅ Model Performance: Precision, Recall, and ROC-AUC.

---

 
