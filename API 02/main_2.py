import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import joblib

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

df = pd.read_csv("X_FlightOnTime/vuelos_etl_limpio.csv")
print("Dataset loaded successfully")

df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")

df = df.dropna()

# Target: delayed if ARR_DELAY > 15 minutes
df["DELAYED"] = (df["ARR_DELAY"] > 15).astype(int)

def time_to_total_minutes(time_str):
    try:
        h, m, s = map(int, str(time_str).split(":"))
        return h * 60 + m
    except:
        return 0


time_cols = [
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "DEP_TIME",
    "ARR_TIME",
    "WHEELS_OFF",
    "WHEELS_ON"
]

for col in time_cols:
    if col in df.columns:
        df[col] = df[col].apply(time_to_total_minutes)

features = [
    "AIRLINE_CODE",
    "ORIGIN",
    "DEST",
    "DISTANCE",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME"
]

X = df[features]
y = df["DELAYED"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

numeric_features = [
    "DISTANCE",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME"
]

categorical_features = [
    "AIRLINE_CODE",
    "ORIGIN",
    "DEST"
]

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(
        drop="first",
        handle_unknown="ignore"
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs"
))
])

pipeline.fit(X_train, y_train)

proba = pipeline.predict_proba(X_test)[:, 1]

for t in [0.2, 0.3, 0.4, 0.5]:
    y_pred_t = (proba >= t).astype(int)

    print(f"\n===== Threshold = {t} =====")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_t))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_t))


joblib.dump(
    pipeline,
    "X_FlightOnTime/DEPLOYABLE.pkl"
)

print("✅ Deployable model saved successfully")