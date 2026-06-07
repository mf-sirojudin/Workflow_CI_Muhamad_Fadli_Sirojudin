"""
=================================================
MODELLING WITH MLFLOW AUTOLOG
Customer Churn Prediction

Author : Muhamad Fadli Sirojudin
=================================================
"""

import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# LOAD DATASET
BASE_DIR = Path(__file__).resolve().parent

dataset_path = (
    BASE_DIR
    / "dataset_preprocessing"
    / "telco_churn_processed.csv"
)

df = pd.read_csv(dataset_path)

print("Dataset Shape:", df.shape)

# SPLIT FEATURE DAN TARGET
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# MLFLOW TRACKING
mlflow.set_experiment("Telco_Customer_Churn")

mlflow.sklearn.autolog()

# TRAINING MODEL
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation")
print("-" * 30)
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))