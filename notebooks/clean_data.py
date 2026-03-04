# This script loads the merged dataset, cleans it, and saves a new file.
# Each step has beginner-friendly comments to explain what is happening.
#
# What this script does:
# 1) Loads the first 50,000 rows of data/merged_fraud_data.csv to keep it fast.
# 2) Drops columns where more than 40% of the values are missing.
# 3) Fills missing numeric values with the median of each numeric column.
# 4) Fills missing categorical values with the string 'unknown'.
# 5) Label-encodes all categorical columns so they become numeric.
# 6) Saves the cleaned result to data/cleaned_fraud_data.csv.
#
# How to run (from the project root or any location):
#   python BWT_Crimson/notebooks/clean_data.py

from pathlib import Path  # For easy and reliable path handling
import sys                # For exiting early with a message if needed
import pandas as pd       # For data loading and cleaning


def main():
    # 1) Work out where we are and where the data is located.
    # __file__ is the path to this script. We resolve it to an absolute path.
    script_path = Path(__file__).resolve()

    # The 'notebooks' folder is the parent directory of this script.
    notebooks_dir = script_path.parent

    # The project root is one level up from 'notebooks'.
    project_root = notebooks_dir.parent

    # The data folder is at '<project_root>/data'.
    data_dir = project_root / "data"

    # We will load the merged dataset created earlier.
    merged_path = data_dir / "merged_fraud_data.csv"

    # We will save the cleaned dataset to this path.
    cleaned_path = data_dir / "cleaned_fraud_data.csv"

    # Check that the merged CSV exists before proceeding.
    if not merged_path.exists():
        print(f"Could not find file: {merged_path}")
        print("Make sure you have run the merge script and created merged_fraud_data.csv.")
        sys.exit(1)

    # 2) Load only the first 50,000 rows to keep things quick and light on memory.
    print(f"Loading first 50,000 rows from: {merged_path}")
    df = pd.read_csv(merged_path, nrows=50_000)

    # Show the starting shape of the data (rows, columns).
    print("Initial shape:", df.shape)

    # 3) Drop columns where more than 40% of the values are missing.
    # We compute the fraction of missing values for each column.
    missing_fraction = df.isna().mean()

    # Select columns where the fraction of missing values is greater than 0.40 (40%).
    cols_to_drop = missing_fraction[missing_fraction > 0.40].index.tolist()

    # Drop those columns from the dataframe.
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} columns with >40% missing values.")
        df = df.drop(columns=cols_to_drop)
    else:
        print("No columns have >40% missing values. Nothing dropped in this step.")

    # 4) Separate numeric and categorical columns.
    # Numeric columns are those with number types (int, float).
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Categorical columns are usually 'object' or 'category' dtypes (strings, categories).
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")

    # 5) Fill missing numeric values with the median of each numeric column.
    # The median is a common choice because it is not affected by extreme outliers.
    if numeric_cols:
        numeric_medians = df[numeric_cols].median(numeric_only=True)
        df[numeric_cols] = df[numeric_cols].fillna(numeric_medians)
        print("Filled missing numeric values with column medians.")

    # 6) Fill missing categorical values with the string 'unknown'.
    if categorical_cols:
        # First, replace missing values with 'unknown'.
        df[categorical_cols] = df[categorical_cols].where(df[categorical_cols].notna(), "unknown")
        # Ensure all categorical columns are strings before label encoding.
        df[categorical_cols] = df[categorical_cols].astype(str)
        print("Filled missing categorical values with 'unknown' and converted to string.")

    # 7) Label-encode all categorical columns so they become numeric.
    # Here we use pandas.factorize, which assigns a unique integer to each category.
    # This avoids needing extra libraries and works well for basic encoding.
    for col in categorical_cols:
        codes, uniques = pd.factorize(df[col], sort=False)
        df[col] = codes
    if categorical_cols:
        print("Label-encoded all categorical columns.")

    # Show the final shape after cleaning.
    print("Final shape:", df.shape)

    # 8) Save the cleaned dataset to a new CSV file.
    print(f"Saving cleaned data to: {cleaned_path}")
    df.to_csv(cleaned_path, index=False)
    print("Save complete.")


if __name__ == "__main__":
    main()
