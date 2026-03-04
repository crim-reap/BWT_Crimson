# This script prepares data for model training from the featured dataset.
# It separates features (X) and the target (y=isFraud), splits into
# training and test sets, computes scale_pos_weight, prints useful stats,
# and saves the splits as CSV files. Comments are written for beginners.
#
# What this script does:
# 1) Loads data/featured_fraud_data.csv
# 2) Separates X (all columns except isFraud) and y (isFraud)
# 3) Splits into 80% train and 20% test with a fixed random seed (42)
# 4) Computes scale_pos_weight = (# non-fraud) / (# fraud)
# 5) Prints shapes and fraud percentages in train and test
# 6) Saves X_train, X_test, y_train, y_test to the data/ folder
#
# How to run (from anywhere):
#   python BWT_Crimson/notebooks/prepare_data.py

from pathlib import Path   # For easy file path handling
import sys                 # To exit early with a message if something is missing
import numpy as np         # For numeric operations and shuffling
import pandas as pd        # For data loading and saving


def main():
    # 1) Figure out where the data is.
    script_path = Path(__file__).resolve()
    notebooks_dir = script_path.parent
    project_root = notebooks_dir.parent
    data_dir = project_root / "data"

    # Input/Output files.
    featured_path = data_dir / "featured_fraud_data.csv"
    x_train_path = data_dir / "X_train.csv"
    x_test_path = data_dir / "X_test.csv"
    y_train_path = data_dir / "y_train.csv"
    y_test_path = data_dir / "y_test.csv"

    # Check the input file exists.
    if not featured_path.exists():
        print(f"Could not find file: {featured_path}")
        print("Please run the feature engineering script first to generate featured_fraud_data.csv.")
        sys.exit(1)

    # 2) Load the dataset.
    print(f"Loading: {featured_path}")
    df = pd.read_csv(featured_path)
    print("Full dataset shape:", df.shape)

    # 3) Separate features (X) and target (y).
    # 'isFraud' should be a column with values 0 (not fraud) or 1 (fraud).
    if "isFraud" not in df.columns:
        print("The column 'isFraud' was not found in the dataset.")
        sys.exit(1)

    y = df["isFraud"].astype(int)  # Make sure it's integers 0/1
    X = df.drop(columns=["isFraud"])

    print("Features (X) shape:", X.shape)
    print("Target (y) shape:", y.shape)

    # 4) Split into 80% training and 20% test sets using a fixed random state.
    # We shuffle indices with a seed for reproducibility.
    rng = np.random.RandomState(42)
    indices = np.arange(len(df))
    rng.shuffle(indices)

    split_idx = int(0.8 * len(indices))  # 80% for training
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    # 5) Compute scale_pos_weight = (# non-fraud) / (# fraud).
    # This is useful for algorithms like XGBoost to handle class imbalance.
    num_fraud = int((y == 1).sum())
    num_non_fraud = int((y == 0).sum())
    if num_fraud == 0:
        scale_pos_weight = float("inf")  # Avoid division by zero; indicates no fraud labels
    else:
        scale_pos_weight = num_non_fraud / num_fraud

    # 6) Print shapes and fraud percentages in train and test.
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    train_fraud_pct = 100.0 * (y_train.mean() if len(y_train) > 0 else 0.0)
    test_fraud_pct = 100.0 * (y_test.mean() if len(y_test) > 0 else 0.0)
    print(f"Train fraud percentage: {train_fraud_pct:.2f}%")
    print(f"Test fraud percentage:   {test_fraud_pct:.2f}%")
    print(f"scale_pos_weight:        {scale_pos_weight:.4f}")

    # 7) Save the splits to CSV files (without index numbers).
    print(f"Saving X_train to: {x_train_path}")
    X_train.to_csv(x_train_path, index=False)

    print(f"Saving X_test to: {x_test_path}")
    X_test.to_csv(x_test_path, index=False)

    print(f"Saving y_train to: {y_train_path}")
    y_train.to_csv(y_train_path, index=False, header=["isFraud"])

    print(f"Saving y_test to: {y_test_path}")
    y_test.to_csv(y_test_path, index=False, header=["isFraud"])

    print("All files saved successfully.")


if __name__ == "__main__":
    main()

