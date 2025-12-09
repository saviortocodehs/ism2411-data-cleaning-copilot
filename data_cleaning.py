"""
src/data_cleaning.py

Purpose:
    Load a messy sales CSV (data/raw/sales_data_raw.csv), apply a sequence of
    cleaning steps, and write a cleaned CSV to data/processed/sales_data_clean.csv.

Cleaning steps include:
    - Standardize column names (lowercase, underscores)
    - Strip leading/trailing whitespace from string columns like product and category
    - Handle missing prices and quantities (fill with median)
    - Remove rows with invalid values (negative price or negative quantity)
    - Provide helpful logging (prints) for quick verification

Instructions for the assignment:
    - At least two functions below should be generated with GitHub Copilot and
      then meaningfully modified. See comments above those functions about how
      to trigger Copilot and what to ask for.
"""

import pandas as pd
import numpy as np
from typing import Union

# ------------------------------
# Copilot-assisted function #1
# ------------------------------
# What this function should do:
#   Load a CSV file into a pandas DataFrame.
#   - Accept a file path string.
#   - Use pandas.read_csv with low_memory=False to avoid dtype warnings.
#   - Return a DataFrame.
# How to trigger Copilot:
#   Place your caret on the next line and start typing "def load_data("
#   or write a short comment like "### copilot: implement load_data(file_path)"
#   then accept Copilot's suggestion and modify if needed.
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV path into a DataFrame.

    Args:
        file_path: path to CSV file (expected encoding utf-8)

    Returns:
        pd.DataFrame with raw data loaded
    """
    # Copilot generated (modified): Added error handling and logging
    try:
        df = pd.read_csv(file_path, low_memory=False, encoding='utf-8')
        print(f"Successfully loaded {len(df)} rows from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise


# ------------------------------
# Copilot-assisted function #2
# ------------------------------
# What this function should do:
#   Standardize column names (lowercase, replace spaces and special chars with underscores),
#   and trim whitespace from string columns that represent product/category names.
# How to trigger Copilot:
#   Place caret on the next line and start typing "def clean_column_names(" or
#   write a short comment "### copilot: implement clean_column_names(df)" and accept suggestion.
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names and trim whitespace from string columns.

    Steps:
    - Lowercase all column names
    - Replace spaces and dots with underscores
    - Remove characters that are not alphanumeric or underscore
    - Strip leading/trailing whitespace from object (string) columns
    """
    # Copilot generated (modified): Added tracking of renamed columns and improved string cleaning
    original_columns = df.columns.tolist()
    new_columns = []
    rename_map = {}
    
    for col in df.columns:
        # lowercase and strip
        c = col.strip().lower()
        # replace spaces, dots, and dashes with underscore
        c = c.replace(" ", "_").replace(".", "_").replace("-", "_")
        # keep only alphanumeric and underscores
        c = "".join(ch for ch in c if ch.isalnum() or ch == "_")
        new_columns.append(c)
        if c != col:
            rename_map[col] = c
    
    df.columns = new_columns
    
    # Log column name changes
    if rename_map:
        print(f"Renamed {len(rename_map)} columns for standardization")

    # Strip whitespace from string/object columns (product, category, etc.)
    # Modified: Improved handling of string columns with better NaN preservation
    for col in df.select_dtypes(include=["object"]).columns:
        # Strip leading/trailing whitespace from non-null values
        mask = df[col].notna()
        df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip()

    return df


def convert_numeric_columns(df: pd.DataFrame, numeric_cols: Union[list, None] = None) -> pd.DataFrame:
    """
    Convert likely numeric columns to numeric dtype, coercing errors to NaN.

    Why:
        Some numeric columns might contain stray characters like '$' or commas;
        coercing to numeric ensures we can compute medians and filter invalid rows.

    Args:
        df: DataFrame
        numeric_cols: optional list of column names to convert; if None, tries common names.

    Returns:
        DataFrame with numeric conversions applied.
    """
    if numeric_cols is None:
        # Common candidate names used in sales datasets
        candidates = ["price", "unit_price", "quantity", "qty", "amount", "total"]
        numeric_cols = [c for c in df.columns if c in candidates]
    for col in numeric_cols:
        if col in df.columns:
            # remove currency symbols and commas then convert
            df[col] = df[col].astype(str).str.replace(r"[\$,]", "", regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ------------------------------
# Copilot-assisted function #3
# ------------------------------
# (This third function can also be generated via Copilot if you like.)
# What this function should do:
#   Fill or drop missing prices and quantities. Be consistent: here we fill with median.
#   Document the choice in a comment.
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values for price and quantity.

    Policy chosen:
        - Fill missing numerical values (price, quantity) with the column median.
        - Why: filling with the median is robust to outliers and keeps rows for downstream analysis.
        - If a column is entirely missing, leave as-is (user can decide later).

    Returns:
        DataFrame with missing price/quantity handled.
    """
    # Copilot generated (modified): Added detailed tracking and validation
    # Identify candidate numeric columns
    candidates = []
    for name in ["price", "unit_price", "quantity", "qty", "amount", "total"]:
        if name in df.columns:
            candidates.append(name)

    # Ensure conversions to numeric happened
    df = convert_numeric_columns(df, candidates)

    total_filled = 0
    for col in candidates:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count == 0:
                # No missing values in this column
                continue
            
            non_null_count = df[col].notna().sum()
            if non_null_count == 0:
                # If entire column is empty, we won't invent a value; leave as-is.
                print(f"Warning: column '{col}' contains no numeric values; leaving as-is.")
                continue
            
            # Fill with median (robust to outliers)
            median_val = df[col].median(skipna=True)
            df[col] = df[col].fillna(median_val)
            total_filled += missing_count
            print(f"Filled {missing_count} missing values in '{col}' with median = {median_val}")
    
    print(f"Total missing values filled: {total_filled}")
    return df


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with clearly invalid values:
      - negative price
      - negative quantity

    Why:
      Negative prices or quantities are most likely data entry mistakes and will break aggregations.

    Returns:
      DataFrame with invalid rows removed. Also prints how many rows removed.
    """
    orig_count = len(df)

    # Determine which columns to check
    price_cols = [c for c in df.columns if "price" in c or c in ("amount", "total")]
    qty_cols = [c for c in df.columns if "qty" in c or c == "quantity"]

    mask = pd.Series(True, index=df.index)

    for p in price_cols:
        # rows with price < 0 are invalid
        mask = mask & (df[p].isna() | (df[p] >= 0))
    for q in qty_cols:
        mask = mask & (df[q].isna() | (df[q] >= 0))

    df_clean = df[mask].copy()
    removed = orig_count - len(df_clean)
    print(f"Removed {removed} rows with negative price/quantity (from {orig_count} total rows).")
    return df_clean


# A small helper to show a quick summary to the user
def quick_summary(df: pd.DataFrame, name: str = "data"):
    """
    Print a short summary for quick debug/inspection.
    """
    print(f"--- Quick summary for {name} ---")
    print(f"Rows: {len(df)}  Columns: {len(df.columns)}")
    print("Columns:", list(df.columns))
    print(df.head().to_string(index=False))
    print("-------------------------------\n")


if __name__ == "__main__":
    raw_path = "data/raw/sales_data_raw.csv"
    cleaned_path = "data/processed/sales_data_clean.csv"

    # 1) Load raw data
    # Why: start from the instructor-provided CSV in data/raw/.
    df_raw = load_data(raw_path)
    quick_summary(df_raw, "raw")

    # 2) Standardize column names and trim strings
    # Why: consistent column naming makes later code robust and reproducible.
    df = clean_column_names(df_raw)
    quick_summary(df, "after_clean_column_names")

    # 3) Handle missing numeric values (prices/quantities)
    # Why: consistent handling of missing numeric values is required by the assignment.
    df = handle_missing_values(df)
    quick_summary(df, "after_handle_missing_values")

    # 4) Remove invalid rows (negative price/quantity)
    # Why: negative values represent errors for sales datasets and must be removed.
    df = remove_invalid_rows(df)
    quick_summary(df, "final_clean")

    # 5) Write the cleaned CSV
    df.to_csv(cleaned_path, index=False)
    print(f"Cleaning complete. Clean file written to: {cleaned_path}")
