# This script trains an XGBoost classifier using prepared train/test splits.
# It explains each step in simple language and saves both the model and
# a confusion matrix image for easy evaluation.
#
# What this script does:
# 1) Loads X_train.csv, X_test.csv, y_train.csv, y_test.csv from the data/ folder
# 2) Calculates scale_pos_weight = (non-fraud count) / (fraud count) using y_train
# 3) Trains an XGBoost classifier with specified hyperparameters
# 4) Evaluates on the test set and prints accuracy, precision, recall, and F1 score
# 5) Saves a confusion matrix heatmap to notebooks/confusion_matrix.png
# 6) Saves the trained model to models/xgboost_model.pkl
#
# How to run (from anywhere):
#   python BWT_Crimson/notebooks/train_xgboost.py

from pathlib import Path  # Easy path handling across OSes
import sys                # For exiting with helpful messages
import joblib             # For saving the trained model to a file
import pandas as pd       # For loading CSV data
import numpy as np        # For numeric operations
import matplotlib.pyplot as plt  # For plotting the confusion matrix
import seaborn as sns            # For a nice-looking heatmap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier  # The XGBoost classifier


def main():
    # 1) Figure out paths relative to this script.
    script_path = Path(__file__).resolve()
    notebooks_dir = script_path.parent
    project_root = notebooks_dir.parent
    data_dir = project_root / "data"
    models_dir = project_root / "models"

    # Make sure the models directory exists (it has a .gitkeep, but we ensure at runtime).
    models_dir.mkdir(parents=True, exist_ok=True)

    # Input files for training and testing.
    x_train_path = data_dir / "X_train.csv"
    x_test_path = data_dir / "X_test.csv"
    y_train_path = data_dir / "y_train.csv"
    y_test_path = data_dir / "y_test.csv"

    # Output files.
    model_path = models_dir / "xgboost_model.pkl"
    cm_fig_path = notebooks_dir / "confusion_matrix.png"

    # Check that all input files are present.
    for p in [x_train_path, x_test_path, y_train_path, y_test_path]:
        if not p.exists():
            print(f"Missing expected file: {p}")
            print("Please run the data preparation step first.")
            sys.exit(1)

    # 2) Load the CSVs.
    # X_* are feature tables, y_* are single-column targets (isFraud).
    print(f"Loading: {x_train_path}")
    X_train = pd.read_csv(x_train_path)
    print(f"Loading: {y_train_path}")
    y_train = pd.read_csv(y_train_path).squeeze("columns").astype(int)

    print(f"Loading: {x_test_path}")
    X_test = pd.read_csv(x_test_path)
    print(f"Loading: {y_test_path}")
    y_test = pd.read_csv(y_test_path).squeeze("columns").astype(int)

    # Ensure columns in test match train order and names (a common pitfall).
    X_test = X_test[X_train.columns]

    print("Shapes:")
    print("  X_train:", X_train.shape)
    print("  y_train:", y_train.shape)
    print("  X_test :", X_test.shape)
    print("  y_test :", y_test.shape)

    # 3) Calculate scale_pos_weight to help with class imbalance.
    # Fraud (1) is usually rare; this weight helps the model pay more attention to it.
    fraud_count = int((y_train == 1).sum())
    nonfraud_count = int((y_train == 0).sum())
    if fraud_count == 0:
        print("Warning: No fraud cases in y_train; using scale_pos_weight = 1.0")
        spw = 1.0
    else:
        spw = nonfraud_count / fraud_count
    print(f"scale_pos_weight (non-fraud / fraud): {spw:.4f}")

    # 4) Create and train the XGBoost classifier.
    # Parameters:
    # - n_estimators: number of trees
    # - max_depth: depth of each tree (controls model complexity)
    # - learning_rate: how fast the model learns (smaller = more gradual)
    # - subsample: fraction of training rows used by each tree (helps generalization)
    # - scale_pos_weight: class imbalance handling (more weight for fraud class)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss",
        tree_method="hist"  # Faster and memory-efficient; good default
    )

    print("Training XGBoost model...")
    model.fit(X_train, y_train)
    print("Training complete.")

    # 5) Evaluate on the test set.
    print("Evaluating on the test set...")
    y_pred = model.predict(X_test)

    # Accuracy: overall percentage of correct predictions.
    acc = accuracy_score(y_test, y_pred)

    # Precision: when the model predicts fraud, how often is it correct?
    prec = precision_score(y_test, y_pred, zero_division=0)

    # Recall: how many of the actual frauds did we catch?
    rec = recall_score(y_test, y_pred, zero_division=0)

    # F1: balance between precision and recall (harmonic mean).
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("Metrics:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")

    # 6) Plot and save a confusion matrix heatmap.
    # Confusion matrix shows:
    # - True Negatives (top-left): non-fraud correctly predicted as non-fraud
    # - False Positives (top-right): non-fraud incorrectly predicted as fraud
    # - False Negatives (bottom-left): fraud incorrectly predicted as non-fraud
    # - True Positives (bottom-right): fraud correctly predicted as fraud
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.title("Confusion Matrix - XGBoost")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(cm_fig_path)
    plt.close()
    print(f"Confusion matrix saved to: {cm_fig_path}")

    # 7) Save the trained model to disk for reuse in APIs or batch scoring.
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()

