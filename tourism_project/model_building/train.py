
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "tourism.csv"
OUT_DIR = BASE / "deployment"
OUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
THRESHOLD_GRID = np.round(np.arange(0.20, 0.61, 0.01), 2)

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["Unnamed: 0", "CustomerID"])

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.20, stratify=y_trainval,
    random_state=RANDOM_STATE
)

def make_pipeline():
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])
    model = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])

# Validation stage: choose a probability threshold using validation F1.
validation_model = make_pipeline()
validation_model.fit(X_train, y_train)
val_prob = validation_model.predict_proba(X_val)[:, 1]

threshold_rows = []
for threshold in THRESHOLD_GRID:
    pred = (val_prob >= threshold).astype(int)
    threshold_rows.append({
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_val, pred),
        "precision": precision_score(y_val, pred, zero_division=0),
        "recall": recall_score(y_val, pred, zero_division=0),
        "f1": f1_score(y_val, pred, zero_division=0)
    })

threshold_df = pd.DataFrame(threshold_rows)
best_row = threshold_df.loc[threshold_df["f1"].idxmax()]
decision_threshold = float(best_row["threshold"])

# Final model: refit on train + validation data.
final_model = make_pipeline()
final_model.fit(X_trainval, y_trainval)

test_prob = final_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= decision_threshold).astype(int)

metrics = {
    "accuracy": float(accuracy_score(y_test, test_pred)),
    "precision": float(precision_score(y_test, test_pred, zero_division=0)),
    "recall": float(recall_score(y_test, test_pred, zero_division=0)),
    "f1": float(f1_score(y_test, test_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, test_prob)),
    "decision_threshold": decision_threshold
}

artifact = {
    "model": final_model,
    "threshold": decision_threshold,
    "metrics": metrics,
    "feature_columns": X.columns.tolist(),
    "categorical_columns": cat_cols,
    "numeric_columns": num_cols
}
joblib.dump(artifact, OUT_DIR / "model.joblib")

# Metadata for the Streamlit UI.
metadata = {
    "numeric": {
        col: {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median())
        } for col in num_cols
    },
    "categories": {
        col: sorted(df[col].dropna().astype(str).unique().tolist())
        for col in cat_cols
    }
}
(OUT_DIR / "feature_metadata.json").write_text(json.dumps(metadata, indent=2))

threshold_df.to_csv(BASE / "model_building" / "threshold_search.csv", index=False)
(BASE / "model_building" / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

# Optional MLflow integration. The local notebook remains self-contained if MLflow is absent.
try:
    import mlflow
    mlflow.set_experiment("visit-with-us-tourism")
    with mlflow.start_run(run_name="random-forest-final"):
        mlflow.log_params({
            "n_estimators": 400,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE
        })
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1", metrics["f1"])
        mlflow.log_metric("roc_auc", metrics["roc_auc"])
        mlflow.log_metric("decision_threshold", metrics["decision_threshold"])
except Exception as exc:
    (BASE / "model_building" / "mlflow_status.txt").write_text(
        "MLflow logging was skipped because MLflow is not installed in this runtime.\n"
        f"Reason: {exc}\n"
    )

print(json.dumps(metrics, indent=2))
