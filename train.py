# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from imblearn.over_sampling import SMOTE
from pycaret.classification import setup, compare_models, pull
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb

# =============================
# 1. Load & Clean Dataset
# =============================
data = pd.read_csv("water_potability.csv")
data = data.rename(columns=lambda x: x.strip())

cols = [
    'ph','Hardness','Solids','Chloramines','Sulfate','Conductivity',
    'Organic_carbon','Trihalomethanes','Turbidity','Potability'
]
data = data[cols]

# Missing value imputation
data['ph'].fillna(data['ph'].median(), inplace=True)
data['Sulfate'].fillna(data['Sulfate'].median(), inplace=True)
data['Trihalomethanes'].fillna(data['Trihalomethanes'].median(), inplace=True)

# =============================
# 2. Outlier Removal (IQR)
# =============================
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

for c in ['ph','Sulfate','Trihalomethanes','Hardness','Solids','Chloramines',
          'Conductivity','Organic_carbon','Turbidity']:
    data = remove_outliers_iqr(data, c)

# =============================
# 3. SMOTE Balancing
# =============================
X = data.drop('Potability', axis=1)
y = data['Potability'].astype(int)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# PyCaret setup dataset
df_for_pycaret = pd.concat([X_res, y_res.reset_index(drop=True)], axis=1)

# =============================
# 4. PyCaret Model Benchmarking
# =============================
s = setup(data=df_for_pycaret, target='Potability', session_id=42, verbose=False)
best_pycaret = compare_models()
leaderboard = pull()
best_name = type(best_pycaret).__name__

# =============================
# 5. Manual Model Selection
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# Choose model based on PyCaret winner
if 'RandomForest' in best_name:
    base_model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [8, 12, None],
        'min_samples_split': [2, 5]
    }

elif 'XGB' in best_name:
    base_model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 8],
        'learning_rate': [0.05, 0.1]
    }

elif 'LGBM' in best_name or 'LightGBM' in best_name:
    base_model = lgb.LGBMClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'learning_rate': [0.05, 0.1]
    }

else:
    base_model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [8, 12, None],
        'min_samples_split': [2, 5]
    }

# =============================
# 6. GridSearchCV Tuning
# =============================
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', base_model)
])

grid = GridSearchCV(
    pipe,
    {'clf__' + k: v for k, v in param_grid.items()},
    cv=5,
    scoring='f1',
    n_jobs=-1
)
grid.fit(X_train, y_train)

best_pipeline = grid.best_estimator_

# =============================
# 7. Evaluate Model
# =============================
y_pred = best_pipeline.predict(X_test)
probs = best_pipeline.predict_proba(X_test)[:, 1]

report = classification_report(y_test, y_pred, output_dict=True)
auc = roc_auc_score(y_test, probs)

# Save final model
joblib.dump(best_pipeline, "best_pipeline.pkl")

# =============================
# 8. SHAP Explainability (Final Fix)
# =============================
model_for_shap = best_pipeline.named_steps['clf']
scaler = best_pipeline.named_steps['scaler']

# Use unscaled X_test for visualization, scaled for SHAP calculation
X_test_scaled = scaler.transform(X_test)

# Force SHAP to use "approximate" mode to avoid interaction fallback
explainer = shap.TreeExplainer(model_for_shap, feature_perturbation='tree_path_dependent')

# Compute SHAP values
shap_values = explainer.shap_values(X_test_scaled)

# If list → binary case → take class 1
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
else:
    shap_vals = shap_values

# ✅ Force summary_plot ONLY (NO interaction)
plt.clf()
plt.figure(figsize=(8, 6))
shap.summary_plot(
    shap_vals,
    X_test,
    plot_type="dot",
    show=False,
    max_display=10  # reduces model confusion
)

plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
plt.close()


