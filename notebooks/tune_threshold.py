# This script tunes the classification threshold for the trained XGBoost model.
# We try thresholds from 0.30 to 0.70 in steps of 0.05 and, for each threshold,
# we compute precision, recall, F1, and the number of false positives.
# We then plot a precision–recall curve and save the best threshold for later use.
#
# Why threshold tuning?
# Many classifiers output probabilities. The default threshold of 0.5 may not be
# ideal for imbalanced problems like fraud detection. Raising the threshold
# usually increases precision (fewer false alarms) at the cost of recall
# (misses more fraud). We'll search for a threshold that keeps recall >= 60%
# while improving precision.
#
# What this script does:
# 1) Loads the trained model and the X_test/y_test splits
# 2) Gets fraud probabilities for the test set
# 3) Evaluates precision/recall/F1/false positives for thresholds 0.30..0.70
# 4) Plots and saves a precision–recall curve (notebooks/precision_recall_curve.png)
# 5) Picks the best threshold with recall >= 0.60, maximizing F1 (tie-break on precision)
# 6) Saves the chosen threshold to models/threshold.pkl
#
# How to run:
#   python BWT_Crimson/notebooks/tune_threshold.py

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix


def main():
    # 1) Figure out paths relative to this script
    script_path = Path(__file__).resolve()
    notebooks_dir = script_path.parent
    project_root = notebooks_dir.parent
    data_dir = project_root / "data"
    models_dir = project_root / "models"

    # Input/output paths
    model_path = models_dir / "xgboost_model.pkl"
    x_test_path = data_dir / "X_test.csv"
    y_test_path = data_dir / "y_test.csv"
    pr_curve_path = notebooks_dir / "precision_recall_curve.png"
    threshold_out_path = models_dir / "threshold.pkl"

    # Check required files exist
    for p in [model_path, x_test_path, y_test_path]:
        if not p.exists():
            print(f"Missing required file: {p}")
            print("Make sure you have trained the model and prepared the test split.")
            sys.exit(1)

    # 2) Load model and test data
    print(f"Loading model: {model_path}")
    model = joblib.load(model_path)

    print(f"Loading X_test: {x_test_path}")
    X_test = pd.read_csv(x_test_path)
    print(f"Loading y_test: {y_test_path}")
    y_test = pd.read_csv(y_test_path).squeeze("columns").astype(int)

    # 3) Get predicted fraud probabilities (probability of class 1)
    # predict_proba returns probabilities for [class 0, class 1]; we take the second column.
    print("Computing predicted probabilities for the test set...")
    proba = model.predict_proba(X_test)[:, 1]

    # 4) Evaluate thresholds 0.30..0.70 (step 0.05)
    thresholds = np.round(np.arange(0.30, 0.701, 0.05), 2)
    results = []
    print("\nThreshold sweep (precision, recall, F1, false positives):")
    for th in thresholds:
        y_pred = (proba >= th).astype(int)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        results.append({"threshold": float(th), "precision": prec, "recall": rec, "f1": f1, "fp": int(fp)})
        print(f"  th={th:.2f}  precision={prec:.4f}  recall={rec:.4f}  F1={f1:.4f}  FP={fp}")

    # 5) Plot and save a precision–recall curve using the probabilities
    print("\nPlotting precision–recall curve...")
    precs, recs, _ = precision_recall_curve(y_test, proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recs, precs, label="XGBoost")
    plt.xlabel("Recall (True Positive Rate)")
    plt.ylabel("Precision (Positive Predictive Value)")
    plt.title("Precision–Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pr_curve_path)
    plt.close()
    print(f"Precision–recall curve saved to: {pr_curve_path}")

    # 6) Choose the best threshold with recall >= 0.60, maximizing F1 then precision
    candidates = [r for r in results if r["recall"] >= 0.60]
    if not candidates:
        print("No threshold met the recall >= 0.60 requirement; selecting the highest F1 regardless.")
        candidates = results

    # Sort by (-F1, -precision, threshold) to pick the highest F1, then highest precision, then lower threshold
    candidates.sort(key=lambda r: (-r["f1"], -r["precision"], r["threshold"]))
    best = candidates[0]
    best_th = best["threshold"]

    print("\nBest threshold selection:")
    print(f"  threshold={best_th:.2f}  precision={best['precision']:.4f}  recall={best['recall']:.4f}  F1={best['f1']:.4f}  FP={best['fp']}")

    # 7) Save the best threshold for use in inference
    joblib.dump({"threshold": best_th, "metrics": best}, threshold_out_path)
    print(f"Saved best threshold to: {threshold_out_path}")


if __name__ == "__main__":
    main()

