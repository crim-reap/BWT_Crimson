# This script loads two CSV files from the project's data folder,
# merges them on the 'TransactionID' column, prints basic information
# to help you understand the data, and then saves the merged result.
# The comments explain every step in simple terms for beginners.
#
# How to run (from the project root or any location):
#   python BWT_Crimson/notebooks/explore_data.py
#
# The script automatically finds the 'data' folder relative to this file.

from pathlib import Path  # Path helps us work with file paths in a simple way
import pandas as pd       # pandas helps us load, merge, and analyze tabular data


def main():
    # 1) Find the directories and files we will use.
    # __file__ is the path to this Python script.
    # .resolve() turns it into an absolute path (no relative pieces like ..).
    script_path = Path(__file__).resolve()

    # The 'notebooks' folder is this script's parent directory.
    notebooks_dir = script_path.parent

    # The project root is one level up from 'notebooks'.
    project_root = notebooks_dir.parent

    # The data folder lives at '<project_root>/data'.
    data_dir = project_root / "data"

    # Build full paths to the input CSV files.
    transaction_path = data_dir / "train_transaction.csv"
    identity_path = data_dir / "train_identity.csv"

    # Build full path for the output CSV we will create.
    output_path = data_dir / "merged_fraud_data.csv"

    # 2) Load the CSV files into pandas DataFrames.
    # A DataFrame is like a table in memory that we can analyze.
    # Make sure the CSV files exist in the 'data' folder before running this.
    print(f"Loading: {transaction_path}")
    train_transaction = pd.read_csv(transaction_path)

    print(f"Loading: {identity_path}")
    train_identity = pd.read_csv(identity_path)

    # 3) Merge the two DataFrames on the 'TransactionID' column.
    # A "left" join keeps all rows from the left table (train_transaction)
    # and matches rows from the right table (train_identity) when possible.
    print("Merging on 'TransactionID' with a left join...")
    merged = pd.merge(
        train_transaction,
        train_identity,
        on="TransactionID",
        how="left"
    )

    # 4) Print the shape (rows, columns) of the merged data.
    # This tells us how big the table is after merging.
    print("\nMerged dataset shape (rows, columns):")
    print(merged.shape)

    # 5) Show the first 5 rows to get a quick look at the data.
    print("\nFirst 5 rows of the merged dataset:")
    print(merged.head(5))

    # 6) Count how many missing values each column has.
    # isna() returns True where values are missing, and sum() counts them.
    print("\nMissing values per column:")
    missing_counts = merged.isna().sum()
    print(missing_counts)

    # 7) Calculate and print the percentage of transactions that are fraud.
    # The 'isFraud' column should be 1 for fraud and 0 for not fraud.
    # Taking the mean of a 0/1 column gives the fraction of 1s.
    print("\nPercentage of transactions that are fraud:")
    fraud_percentage = merged["isFraud"].mean() * 100
    print(f"{fraud_percentage:.2f}%")

    # 8) Save the merged DataFrame to a new CSV file.
    # index=False prevents pandas from writing row numbers into the file.
    print(f"\nSaving merged dataset to: {output_path}")
    merged.to_csv(output_path, index=False)
    print("Save complete.")


if __name__ == "__main__":
    main()

