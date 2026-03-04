# This script adds an Isolation Forest (IF) as a second detection layer and
# compares a combined decision rule against XGBoost alone.
# The combined rule marks a transaction as fraud if:
#   (XGBoost predicts fraud) OR (Isolation Forest flags it as an anomaly).
#
# Steps:
# 1) Load XGBoost model (and tuned threshold if available) and the prepared splits
# 2) Train Isolation Forest only on non-fraud training data (normal-only training)
# 3) Generate predictions from both models on the test set
# 4) Combine decisions with OR logic
# 5) Print precision, recall, and false positive count for:
#      - XGBoost alone
#      - Combined (XGBoost OR Isolation Forest)
# 6) Save the Isolation Forest model to models/isolation_forest.pkl
#
# How to run:
#   python BWT_Crimson/notebooks/combine_isolation_xgb.py

from pathlib import Path  # For file paths
import sys                # To exit with helpful messages
import joblib             # For loading/saving models and thresholds
import numpy as np        # For numeric operations
import pandas as pd       # For loading CSV data
from sklearn.ensemble import IsolationForest  # The anomaly detector
from sklearn.metrics import precision_score, recall_score, confusion_matrix


def print_metrics(name: str, y_true, y_pred):
    # Precision: when we say "fraud", how often are we right?
    prec = precision_score(y_true, y_pred, zero_division=0)
    # Recall: how many of the actual frauds did we find?
    rec = recall_score(y_true, y_pred, zero_division=0)
    # False positives: non-fraud wrongly flagged as fraud
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"{name} -> precision={prec:.4f}, recall={rec:.4f}, false_positives={fp}")
    return {"precision": prec, "recall": rec, "fp": int(fp)}


def main():
    # 1) Resolve paths
    script_path = Path(__file__).resolve()
    notebooks_dir = script_path.parent
    project_root = notebooks_dir.parent
    data_dir = project_root / "data"
    models_dir = project_root / "models"

    # Files we need
    x_train_path = data_dir / "X_train.csv"
    y_train_path = data_dir / "y_train.csv"
    x_test_path = data_dir / "X_test.csv"
    y_test_path = data_dir / "y_test.csv"
    xgb_model_path = models_dir / "xgboost_model.pkl"
    threshold_path = models_dir / "threshold.pkl"
    if_model_path = models_dir / "isolation_forest.pkl"

    # Check expected inputs exist
    for p in [x_train_path, y_train_path, x_test_path, y_test_path, xgb_model_path]:
        if not p.exists():
            print(f"Missing required file: {p}")
            print("Run the previous steps (prepare, train XGBoost, tune threshold) first.")
            sys.exit(1)

    # Load splits
    print(f"Loading: {x_train_path}")
    X_train = pd.read_csv(x_train_path)
    print(f"Loading: {y_train_path}")
    y_train = pd.read_csv(y_train_path).squeeze("columns").astype(int)
    print(f"Loading: {x_test_path}")
    X_test = pd.read_csv(x_test_path)
    print(f"Loading: {y_test_path}")
    y_test = pd.read_csv(y_test_path).squeeze("columns").astype(int)

    # Load XGBoost model
    print(f"Loading XGBoost model: {xgb_model_path}")
    xgb = joblib.load(xgb_model_path)

    # Ensure test columns match training columns used by XGBoost
    X_test = X_test[X_train.columns]

    # Load tuned threshold if available, otherwise default to 0.5
    if threshold_path.exists():
        best = joblib.load(threshold_path)
        threshold = float(best.get("threshold", 0.5))
        print(f"Using tuned threshold from file: {threshold:.2f}")
    else:
        threshold = 0.5
        print("No threshold.pkl found; using default threshold 0.50")

    # 2) Train Isolation Forest on non-fraud training data only
    # Isolation Forest tries to learn what "normal" looks like, so we fit on y=0 rows.
    X_train_normals = X_train[y_train == 0]
    print(f"Training Isolation Forest on {len(X_train_normals)} non-fraud training rows...")

    # contamination approximates the expected proportion of anomalies in data.
    # Fraud rate ~2-3%, so we use 0.03 as a reasonable starting point.
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.03,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    iso.fit(X_train_normals)
    print("Isolation Forest training complete.")

    # Save the Isolation Forest model
    joblib.dump(iso, if_model_path)
    print(f"Isolation Forest model saved to: {if_model_path}")

    # 3) Get predictions on the test set
    # XGBoost probabilities -> threshold to get class labels
    print("Scoring test set with XGBoost...")
    proba = xgb.predict_proba(X_test)[:, 1]
    y_pred_xgb = (proba >= threshold).astype(int)

    # Isolation Forest predict: -1 = anomaly (potential fraud), 1 = normal
    print("Scoring test set with Isolation Forest...")
    iso_labels = iso.predict(X_test)  # -1 anomaly, 1 normal
    y_pred_iso = (iso_labels == -1).astype(int)

    # 4) Combine with OR logic: flagged if either model says "fraud"
    y_pred_combined = np.where((y_pred_xgb == 1) | (y_pred_iso == 1), 1, 0)

    # 5) Print comparison metrics
    print("\nEvaluation on test set:")
    metrics_xgb = print_metrics("XGBoost only", y_test, y_pred_xgb)
    metrics_combined = print_metrics("Combined (XGB OR IF)", y_test, y_pred_combined)

    # Highlight simple comparison
    print("\nSummary comparison:")
    print(f"  XGBoost    -> precision={metrics_xgb['precision']:.4f}, recall={metrics_xgb['recall']:.4f}, FP={metrics_xgb['fp']}")
    print(f"  Combined   -> precision={metrics_combined['precision']:.4f}, recall={metrics_combined['recall']:.4f}, FP={metrics_combined['fp']}")


if __name__ == "__main__":
    main()

