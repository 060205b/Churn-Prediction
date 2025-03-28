# feature_eng_and_modeling.py

# ....................................................................................
# 1. Import Libraries
# ....................................................................................
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    classification_report
)

# ....................................................................................
# 2. Load and Prepare Data
# ....................................................................................
df = pd.read_csv('./data_for_predictions.csv')
df.drop(columns=["Unnamed: 0"], inplace=True)

y = df['churn']
X = df.drop(columns=['id', 'churn'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ....................................................................................
# 3. Train the Random Forest Classifier
# ....................................................................................
rf = RandomForestClassifier(
    n_estimators=1000,
    criterion='entropy',
    min_samples_split=10,
    random_state=42
)
rf.fit(X_train, y_train)

# ....................................................................................
# 4. Make Predictions
# ....................................................................................
y_pred_rf = rf.predict(X_test)

# ....................................................................................
# 5. Evaluate Model Performance
# ....................................................................................
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf).ravel()

accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)
f1 = f1_score(y_test, y_pred_rf)
roc_auc = roc_auc_score(y_test, y_pred_rf)
conf_matrix = confusion_matrix(y_test, y_pred_rf)

# ....................................................................................
# 6. Output Results
# ....................................................................................
print("\n================= RANDOM FOREST MODEL PERFORMANCE =================\n")

print(f"{'True Positives (TP):':<25} {tp}")
print(f"{'False Positives (FP):':<25} {fp}")
print(f"{'False Negatives (FN):':<25} {fn}")
print(f"{'True Negatives (TN):':<25} {tn}\n")

print(f"{'Accuracy:':<25} {accuracy:.4f}")
print(f"{'Precision:':<25} {precision:.4f}")
print(f"{'Recall:':<25} {recall:.4f}")
print(f"{'F1-Score:':<25} {f1:.4f}")
print(f"{'ROC-AUC Score:':<25} {roc_auc:.4f}")

print("\n----------------- Confusion Matrix -----------------")
print(conf_matrix)

print("\n---------------- Classification Report -------------")
print(classification_report(y_test, y_pred_rf))


================= RANDOM FOREST MODEL PERFORMANCE =================

True Positives (TP):        12
False Positives (FP):       1
False Negatives (FN):       354
True Negatives (TN):        3285

Accuracy:                   0.9028
Precision:                  0.9231
Recall:                     0.0328
F1-Score:                   0.0633
ROC-AUC Score:              0.5162

----------------- Confusion Matrix -----------------
[[3285    1]
 [ 354   12]]

---------------- Classification Report -------------
              precision    recall  f1-score   support

           0       0.90      1.00      0.95      3286
           1       0.92      0.03      0.06       366

    accuracy                           0.90      3652
   macro avg       0.91      0.51      0.50      3652
weighted avg       0.90      0.90      0.86      3652
