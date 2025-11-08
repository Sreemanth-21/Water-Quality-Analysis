# 💧 Water Quality Analysis & Potability Prediction
### Machine Learning • PyCaret • Random Forest • Streamlit Web App

This project predicts **whether water is safe for drinking** using physicochemical parameters like pH, Hardness, Solids, Sulfates, and more.  
It combines **automated ML (PyCaret)**, **manual ML (Random Forest)** and a **Streamlit web app** for real-time predictions.

---

# ✅ Features

###  Machine Learning
-  Missing value handling, outlier removal (IQR)
-  Balanced dataset using SMOTE
-  Auto ML using PyCaret
-  Manual Random Forest with GridSearchCV
-  Metrics: Accuracy, F1, AUC, Precision, Recall
-  Explainability using SHAP

###  Web Application
-  Built with Streamlit
-  User inputs for all water parameters
-  Predicts Potable /  Not Potable
-  Confidence score + SHAP feature importance

---

# 📊 Dataset

Kaggle Dataset:  
https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability

Attributes:
ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity, Potability

---

#  Workflow Overview

## 1️⃣ Data Preprocessing
- Fill missing values with median  
- Remove outliers (IQR)  
- SMOTE class balancing  
- Train-test split  

## 2️⃣ PyCaret – Model Comparison
- Compare 15+ algorithms  
- Pick best model (Random Forest)  

## 3️⃣ Manual ML Training
- RandomForestClassifier  
- GridSearchCV hyperparameter tuning  
- Performance evaluation  

## 4️⃣ Explainability
- SHAP TreeExplainer  
- Generate shap_summary.png  

## 5️⃣ Streamlit Deployment
- UI for inputs  
- Predictions + probability  
- SHAP feature chart  

---

# 🛠 Installation & Setup

##  1. Clone Repo
```bash
git clone https://github.com/Sreemanth-21/Water-Quality-Analysis

cd WaterQualityAnalysis
```
##  2. Create Virtual Environment
```python
python -m venv venv
```
##  3. Activate Environment

Windows:

```   
venv\Scripts\activate
```
Mac/Linux:

```  
source venv/bin/activate
```
## 4. Install Requirements
```python
pip install -r requirements.txt
```
---

# Train the Model
```python
python train.py
```
Outputs:
- best_pipeline.pkl  
- shap_summary.png  

---

# Run Streamlit App
```python
streamlit run app.py
```
Open:
http://localhost:8501

---

# Project Structure
```
WaterQualityAnalysis/
├── train.py
├── app.py
├── requirements.txt
├── water_potability.csv
├── best_pipeline.pkl
├── shap_summary.png
└── README.md
```



---

#  Results Summary

Best Model: **Random Forest**

✅ Accuracy: ~0.70  
✅ F1 Score: ~0.69  
✅ AUC: ~0.75  

---
