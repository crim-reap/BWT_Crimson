# This script loads the cleaned dataset, creates new useful features,
# and saves the result to a new CSV. Every step is explained in simple terms.
#
# What this script does (using first 50,000 rows for speed):
# 1) Loads data/cleaned_fraud_data.csv
# 2) Creates:
#    - transaction_hour: hour of day derived from TransactionDT (0–23)
#    - is_high_amount: 1 if TransactionAmt > 90th percentile, else 0
#    - amount_deviation: absolute difference from the mean TransactionAmt
#    - is_free_email: 1 if P_emaildomain contains gmail/yahoo/hotmail (when available as text)
#    - card_mismatch: 1 if (card4, card6) combo appears < 10 times, else 0
# 3) Saves to data/featured_fraud_data.csv
#
# How to run (from the project root or any location):
#   python BWT_Crimson/notebooks/engineer_features.py

from pathlib import Path   # For easy file path handling
import sys                 # To exit with a message if something is missing
import pandas as pd        # For data loading and feature engineering
import numpy as np         # For numeric operations


def main():
    # 1) Figure out where the data is.
    script_path = Path(__file__).resolve()
    notebooks_dir = script_path.parent
    project_root = notebooks_dir.parent
    data_dir = project_root / "data"

    # Input cleaned dataset and output featured dataset paths.
    cleaned_path = data_dir / "cleaned_fraud_data.csv"
    featured_path = data_dir / "featured_fraud_data.csv"

    # Check the input file exists.
    if not cleaned_path.exists():
        print(f"Could not find file: {cleaned_path}")
        print("Please run the cleaning script first to generate cleaned_fraud_data.csv.")
        sys.exit(1)

    # Load only the first 50,000 rows to keep it fast and memory-friendly.
    print(f"Loading first 50,000 rows from: {cleaned_path}")
    df = pd.read_csv(cleaned_path, nrows=50_000)
    print("Initial shape:", df.shape)

    # 2) Create transaction_hour from TransactionDT.
    # In this dataset, TransactionDT is a running time index measured in seconds.
    # We extract the hour of day by taking:
    #   - seconds_in_day = TransactionDT % 86400 (number of seconds in a day)
    #   - hour = seconds_in_day // 3600  (integer division by 3600 seconds/hour)
    if "TransactionDT" in df.columns:
        # Ensure we work with numeric values and preserve missing as NaN.
        td = pd.to_numeric(df["TransactionDT"], errors="coerce")
        seconds_in_day = (td % 86_400)
        # Use -1 for missing hours to keep the column as integer type.
        df["transaction_hour"] = (seconds_in_day // 3_600).fillna(-1).astype("int64")
        print("Added feature: transaction_hour")
    else:
        df["transaction_hour"] = -1
        print("TransactionDT not found; transaction_hour filled with missing values.")

    # 3) Create is_high_amount based on 90th percentile of TransactionAmt.
    if "TransactionAmt" in df.columns:
        amt = pd.to_numeric(df["TransactionAmt"], errors="coerce")
        p90 = amt.quantile(0.90)
        df["is_high_amount"] = (amt > p90).fillna(False).astype("int64")
        print(f"Added feature: is_high_amount (threshold ~ {p90:.2f})")
    else:
        df["is_high_amount"] = 0
        print("TransactionAmt not found; is_high_amount filled with missing values.")

    # 4) Create amount_deviation as the absolute difference from the mean TransactionAmt.
    if "TransactionAmt" in df.columns:
        mean_amt = amt.mean()
        df["amount_deviation"] = (amt - mean_amt).abs()
        print(f"Added feature: amount_deviation (mean ~ {mean_amt:.2f})")
    else:
        df["amount_deviation"] = pd.NA
        print("TransactionAmt not found; amount_deviation filled with missing values.")

    # 5) Create is_free_email: 1 if P_emaildomain contains gmail, yahoo, or hotmail.
    # Note: If the cleaned data encoded P_emaildomain as numbers, we cannot detect text domains.
    # In that case, we set is_free_email = 0 and print a helpful message.
    free_domains = ("gmail", "yahoo", "hotmail")
    if "P_emaildomain" in df.columns:
        if df["P_emaildomain"].dtype == "object":
            dom = df["P_emaildomain"].astype(str).str.lower()
            pattern = "|".join(free_domains)
            df["is_free_email"] = dom.str.contains(pattern, na=False).astype("int64")
            print("Added feature: is_free_email (text-based domain check).")
        else:
            # Likely label-encoded already; default to 0 since we cannot map back.
            df["is_free_email"] = 0
            print("P_emaildomain appears numeric (encoded); is_free_email set to 0.")
    else:
        df["is_free_email"] = 0
        print("P_emaildomain not found; is_free_email filled with missing values.")

    # 6) Create card_mismatch: 1 if (card4, card6) combo appears < 10 times.
    if "card4" in df.columns and "card6" in df.columns:
        combo_counts = df.groupby(["card4", "card6"]).size()
        # Map each row's (card4, card6) to its frequency, then check if < 10.
        df["card_mismatch"] = (
            df.set_index(["card4", "card6"]).index.map(combo_counts).fillna(0) < 10
        ).astype("int64")
        print("Added feature: card_mismatch (rare card4-card6 combinations).")
    else:
        df["card_mismatch"] = 0
        print("card4 and/or card6 not found; card_mismatch filled with missing values.")

    # 7) Save the resulting DataFrame with the new features.
    print("Final shape with new features:", df.shape)
    print(f"Saving featured data to: {featured_path}")
    df.to_csv(featured_path, index=False)
    print("Save complete.")


if __name__ == "__main__":
    main()
