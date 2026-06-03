"""
Customer Churn Prediction — Production ML Pipeline (Olist)

Approach:
- Engineered feature set: 7 features built from price, freight, and delivery signals
- RandomForestClassifier with class_weight='balanced' for class imbalance
- RandomizedSearchCV hyperparameter tuning (n_estimators, max_depth, min_samples_*)
- SHAP TreeExplainer for explainability
- MLflow experiment tracking (params, metrics, artifacts, model)

Note on dataset:
- The included `data/sample_data.csv` is a 1K-row sample for CI/demo.
- On the full ~115K-row cleaned Olist dataset (cleaned_olist_master.csv),
  this pipeline achieves ROC-AUC ~0.72 (baseline) to ~0.77 (tuned, hold-out).
- On the 1K sample, expect ROC-AUC ~0.65-0.70.
"""
import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "sample_data.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}")
mlflow.set_experiment("olist_churn_rf")

FEATURES = ["price", "freight_value", "delivery_time",
            "total_cost", "freight_ratio", "is_expensive", "is_delayed"]
TARGET = "low_review"

# Toggle: full RandomizedSearchCV (slow, full dataset) or quick fit (sample)
DO_TUNE = os.environ.get("DO_TUNE", "0") == "1"


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")
    return df


def engineer_features(df):
    """Build the 7-feature set used in the notebook study."""
    df = df.copy()
    df[TARGET] = df["review_score"].apply(lambda x: 1 if x <= 2 else 0)

    # Engineered features
    df["total_cost"] = df["price"] + df["freight_value"]
    df["freight_ratio"] = df["freight_value"] / (df["price"] + 1)
    df["is_expensive"] = (df["price"] > df["price"].median()).astype(int)
    if "is_late" in df.columns:
        df["is_delayed"] = df["is_late"].astype(int)
    else:
        df["is_delayed"] = (df["delivery_time"] > df["delivery_time"].median()).astype(int)

    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES]
    y = df[TARGET]
    print(f"Class balance — churn (1): {y.mean():.2%}, non-churn (0): {1-y.mean():.2%}")
    print(f"Using {len(FEATURES)} features on {len(df)} rows")
    return X, y


def train_model(X_train, y_train):
    """Train RandomForest with class imbalance handling.
    If DO_TUNE=1, runs RandomizedSearchCV; else uses notebook-best params."""
    if DO_TUNE:
        print("\nRunning RandomizedSearchCV (this will take a while)...")
        base = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
        param_dist = {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [None, 2, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": ["balanced"],
        }
        search = RandomizedSearchCV(
            base, param_distributions=param_dist,
            n_iter=20, cv=5, scoring="roc_auc",
            random_state=42, n_jobs=-1, verbose=1,
        )
        search.fit(X_train, y_train)
        print(f"Best CV AUC: {search.best_score_:.4f}")
        print(f"Best params: {search.best_params_}")
        return search.best_estimator_, search.best_params_

    # Notebook-best params (from RandomizedSearchCV result)
    params = {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model, params


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    cm = confusion_matrix(y_test, y_pred)
    print("\n=== Evaluation on hold-out test set ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    return metrics, cm


def explain_with_shap(model, X_test):
    print("\nComputing SHAP values for explainability...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # For binary classifier, take positive-class SHAP values
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    plt.figure(figsize=(9, 5))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    bar_path = os.path.join(ARTIFACTS_DIR, "shap_feature_importance.png")
    plt.savefig(bar_path, dpi=120, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    beeswarm_path = os.path.join(ARTIFACTS_DIR, "shap_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"SHAP plots saved to {ARTIFACTS_DIR}")
    return bar_path, beeswarm_path


def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"]); ax.set_yticklabels(["No Churn", "Churn"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
    plt.colorbar(im); plt.tight_layout()
    path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def save_model(model, features):
    model_path = os.path.join(MODEL_DIR, "olist_pipeline.pkl")
    joblib.dump({"model": model, "features": features}, model_path)
    print(f"Model saved to {model_path}")
    return model_path


def run_pipeline():
    with mlflow.start_run() as run:
        print(f"MLflow run_id: {run.info.run_id}")

        df = load_data()
        X, y = engineer_features(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("features", ",".join(FEATURES))
        mlflow.log_param("tuned", DO_TUNE)

        model, params = train_model(X_train, y_train)
        for k, v in params.items():
            mlflow.log_param(f"rf_{k}", v)

        metrics, cm = evaluate_model(model, X_test, y_test)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        cm_path = plot_confusion_matrix(cm)
        mlflow.log_artifact(cm_path)

        bar_path, beeswarm_path = explain_with_shap(model, X_test)
        mlflow.log_artifact(bar_path)
        mlflow.log_artifact(beeswarm_path)

        model_path = save_model(model, FEATURES)
        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(model, "rf_model")

        print(f"\n=== Run complete. ROC-AUC = {metrics['roc_auc']:.4f} ===")
        return metrics


if __name__ == "__main__":
    run_pipeline()
