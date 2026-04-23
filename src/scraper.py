"""
scraper.py
----------
Loads PBS CPI Annexure-1 Excel files from a local folder.

USAGE:
    Place your monthly Excel files in data/annexures/ with naming: YYYY_MM.xlsx
    e.g. 2022_01.xlsx, 2022_02.xlsx, ..., 2024_12.xlsx

    from src.scraper import load_all_months, inspect_file
    df = load_all_months("data/annexures/")
    df.to_csv("data/cpi_data.csv", index=False)
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# INSPECTION HELPER — run this first on a new file
# ─────────────────────────────────────────────────────────────

def inspect_file(filepath):
    """
    Print raw structure of an Excel file so you can identify
    which row contains city headers and how items are laid out.
    Always call this before running the full pipeline on a new batch.
    """
    print(f"\n{'='*60}")
    print(f"Inspecting: {filepath}")
    print('='*60)

    raw = pd.read_excel(filepath, header=None)
    print(f"Shape (rows x cols): {raw.shape}")
    print(f"\nFirst 12 rows (raw):")
    print(raw.head(12).to_string())
    print(f"\nLast 5 rows:")
    print(raw.tail(5).to_string())


# ─────────────────────────────────────────────────────────────
# FIND HEADER ROW
# ─────────────────────────────────────────────────────────────

# Major Pakistani cities that appear in PBS Urban CPI
_CITY_KEYWORDS = [
    'karachi', 'lahore', 'islamabad', 'rawalpindi', 'faisalabad',
    'multan', 'peshawar', 'quetta', 'hyderabad', 'sialkot',
    'gujranwala', 'bahawalpur', 'sargodha', 'sukkur', 'larkana'
]

def _find_header_row(raw_df):
    """
    Scan rows top-down to find the row that contains city names.
    Returns the integer index of that row, or raises ValueError.
    """
    for i, row in raw_df.iterrows():
        row_lower = row.astype(str).str.strip().str.lower()
        matches = sum(row_lower.str.contains(city, na=False).any()
                      for city in _CITY_KEYWORDS)
        if matches >= 3:          # at least 3 known cities found
            return i
    raise ValueError(
        "Could not auto-detect the header row. "
        "Run inspect_file() to view the raw structure and set header_row manually."
    )


# ─────────────────────────────────────────────────────────────
# LOAD ONE ANNEXURE FILE
# ─────────────────────────────────────────────────────────────

def load_annexure(filepath, header_row=None):
    """
    Load one PBS CPI Annexure-1 Excel file.

    Parameters
    ----------
    filepath   : path to the .xlsx file
    header_row : integer row index for the city-name header row.
                 If None, it is detected automatically.

    Returns
    -------
    items_df : DataFrame  — rows = items, columns = ['item','category'] + cities
    cities   : list of city name strings
    """
    raw = pd.read_excel(filepath, header=None)

    if header_row is None:
        header_row = _find_header_row(raw)
        print(f"  Auto-detected header row: {header_row}")

    # Re-read with proper header
    df = pd.read_excel(filepath, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    # ── Identify city columns ──────────────────────────────
    # First column is the item name column; rest are cities (skip unnamed/NaN cols)
    item_col = df.columns[0]
    city_cols = []
    for c in df.columns[1:]:
        c_stripped = c.strip()
        # Skip blank, 'Unnamed', or purely numeric column names
        if (c_stripped == '' or
                c_stripped.lower().startswith('unnamed') or
                c_stripped == 'nan'):
            continue
        city_cols.append(c_stripped)

    # Rename first column cleanly
    df = df.rename(columns={item_col: 'item_raw'})

    # ── Walk rows: detect category headers vs item rows ────
    current_category = 'Unknown'
    records = []

    for _, row in df.iterrows():
        item_name = str(row['item_raw']).strip()
        if item_name in ('nan', '', 'NaN', 'None'):
            continue

        # If all city-value cells are NaN → this is a category heading row
        city_vals = pd.to_numeric(
            row[city_cols].astype(str).str.replace(',', ''),
            errors='coerce'
        )
        if city_vals.isna().all():
            current_category = item_name
            continue

        # Build price record
        record = {'item': item_name, 'category': current_category}
        for city in city_cols:
            try:
                val = float(str(row[city]).replace(',', ''))
                record[city] = val if val > 0 else np.nan
            except (ValueError, TypeError):
                record[city] = np.nan

        records.append(record)

    items_df = pd.DataFrame(records)

    # Normalize city column names: strip whitespace, title-case
    rename_map = {c: c.strip().title() for c in city_cols}
    items_df = items_df.rename(columns=rename_map)
    cities = [rename_map[c] for c in city_cols]

    return items_df, cities


# ─────────────────────────────────────────────────────────────
# LOAD ALL MONTHS → LONG FORMAT
# ─────────────────────────────────────────────────────────────

def load_all_months(data_dir, header_row=None):
    """
    Load every YYYY_MM.xlsx file in data_dir and return a single
    long-format DataFrame.

    Parameters
    ----------
    data_dir   : folder containing YYYY_MM.xlsx files
    header_row : if all files share the same header row, pass it here
                 to skip auto-detection (faster)

    Returns
    -------
    DataFrame with columns: [year, month, item, category, city, price]
    """
    data_path = Path(data_dir)
    xlsx_files = sorted(data_path.glob("*.xlsx"))

    if not xlsx_files:
        raise FileNotFoundError(
            f"No .xlsx files found in '{data_dir}'.\n"
            "Name files as YYYY_MM.xlsx, e.g. 2022_01.xlsx"
        )

    print(f"Found {len(xlsx_files)} Excel files in '{data_dir}'")
    all_records = []

    for filepath in xlsx_files:
        stem = filepath.stem          # e.g. "2022_01"
        parts = stem.split("_")

        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            print(f"  Skipping '{filepath.name}' — expected YYYY_MM.xlsx format")
            continue

        year, month = int(parts[0]), int(parts[1])
        print(f"  Loading {year}-{month:02d} ...", end=" ")

        try:
            items_df, cities = load_annexure(filepath, header_row=header_row)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        # Wide → long
        for _, row in items_df.iterrows():
            for city in cities:
                price = row.get(city, np.nan)
                if pd.notna(price) and price > 0:
                    all_records.append({
                        'year':     year,
                        'month':    month,
                        'item':     row['item'],
                        'category': row['category'],
                        'city':     city,
                        'price':    price
                    })

        print(f"OK — {len(items_df)} items, {len(cities)} cities")

    df_long = pd.DataFrame(all_records)

    print(f"\n{'─'*50}")
    print(f"Total records : {len(df_long):,}")
    print(f"Unique items  : {df_long['item'].nunique()}")
    print(f"Unique cities : {df_long['city'].nunique()}")
    print(f"Years covered : {sorted(df_long['year'].unique())}")
    print(f"{'─'*50}\n")

    return df_long


# ─────────────────────────────────────────────────────────────
# NORMALIZE ITEM NAMES (fixes minor spelling diffs across months)
# ─────────────────────────────────────────────────────────────

def normalize_item_names(df):
    """
    Strip whitespace and lowercase item names so that
    'Wheat Flour ' and 'wheat flour' are treated as the same item.
    """
    df = df.copy()
    df['item'] = df['item'].str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
    df['category'] = df['category'].str.strip()
    return df


# ─────────────────────────────────────────────────────────────
# ENTRY POINT — quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        # python src/scraper.py data/annexures/2022_01.xlsx
        inspect_file(sys.argv[1])
    else:
        # python src/scraper.py  → load everything
        df = load_all_months("data/annexures/")
        df = normalize_item_names(df)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/cpi_data.csv", index=False)
        print("Saved → data/cpi_data.csv")
        print(df.head(10).to_string())