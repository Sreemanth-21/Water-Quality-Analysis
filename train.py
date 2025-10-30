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

data = pd.read_csv("water_potability.csv")
data = data.rename(columns=lambda x: x.strip())
cols = ['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity','Potability']
data = data[cols]
data['ph'].fillna(data['ph'].median(), inplace=True)
data['Sulfate'].fillna(data['Sulfate'].median(), inplace=True)
data['Trihalomethanes'].fillna(data['Trihalomethanes'].median(), inplace=True)

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

for c in ['ph','Sulfate','Trihalomethanes','Hardness','Solids','Chloramines','Conductivity','Organic_carbon','Turbidity']:
    data = remove_outliers_iqr(data, c)

X = data.drop('Potability', axis=1)
y = data['Potability'].astype(int)

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

df_for_pycaret = pd.concat([X_res, y_res.reset_index(drop=True)], axis=1)

s = setup(data=df_for_pycaret, target='Potability', session_id=42, verbose=False)

best_pycaret = compare_models()
leaderboard = pull()
best_name = type(best_pycaret).__name__

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

if 'RandomForest' in best_name or 'RandomForestClassifier' in best_name:
    base_model = RandomForestClassifier(random_state=42)
    param_grid = {'n_estimators':[100,200], 'max_depth':[8,12,None], 'min_samples_split':[2,5]}
elif 'XGB' in best_name or 'XGBClassifier' in best_name:
    base_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    param_grid = {'n_estimators':[100,200], 'max_depth':[3,6,8], 'learning_rate':[0.05,0.1]}
elif 'LGBM' in best_name or 'LGBMClassifier' in best_name or 'LightGBM' in best_name:
    base_model = lgb.LGBMClassifier(random_state=42)
    param_grid = {'n_estimators':[100,200], 'max_depth':[5,10,None], 'learning_rate':[0.05,0.1]}
else:
    base_model = RandomForestClassifier(random_state=42)
    param_grid = {'n_estimators':[100,200], 'max_depth':[8,12,None], 'min_samples_split':[2,5]}

pipe = Pipeline([('scaler', StandardScaler()), ('clf', base_model)])
grid = GridSearchCV(pipe, {'clf__' + k: v for k,v in param_grid.items()}, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)
best_pipeline = grid.best_estimator_

y_pred = best_pipeline.predict(X_test)
probs = best_pipeline.predict_proba(X_test)[:,1] if hasattr(best_pipeline, 'predict_proba') else None
report = classification_report(y_test, y_pred, output_dict=True)
auc = roc_auc_score(y_test, probs) if probs is not None else None

joblib.dump(best_pipeline, "best_pipeline.pkl")

model_for_shap = best_pipeline.named_steps['clf']
X_test_scaled = best_pipeline.named_steps['scaler'].transform(X_test)
if hasattr(model_for_shap, 'feature_importances_') or isinstance(model_for_shap, (xgb.XGBClassifier, lgb.LGBMClassifier, RandomForestClassifier)):
    explainer = shap.TreeExplainer(model_for_shap)
    shap_values = explainer.shap_values(X_test_scaled)
    plt.figure(figsize=(8,6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X_test, show=False)
    else:
        shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150)
    plt.close()
else:
    pass
