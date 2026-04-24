"""
scraper.py
----------
Loads ALL Excel files from data/annexures/ and produces
a single long-format CSV: data/cpi_data.csv

Confirmed file structure (from inspecting real files):
  Row 0  : Title row  e.g. "Average Monthly Prices of 51 Essential Items..."
  Row 1  : Header     col0=S.No, col1=Description, col2=Unit, col3..19=Cities
  Row 2+ : Data       51 item rows
  Col 20+: Average prices & % change — ignored

File naming convention: YYYY_MM.xlsx  e.g. 2023_03.xlsx

Usage:
    python src/scraper.py                          # load all → data/cpi_data.csv
    python src/scraper.py data/annexures/2023_03.xlsx  # inspect one file
"""

import os
import re
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# CITY NAME LOOKUP
# Adobe/online converters preserve the PDF line-break hyphens
# e.g. 'Islam-\nabad' → 'Islamabad'
# ─────────────────────────────────────────────────────────────

_CITY_FIXES = {
    'Islam-\nabad':    'Islamabad',
    'Rawal-\npindi':   'Rawalpindi',
    'Gujran-\nwala':   'Gujranwala',
    'Faisal-\nabad':   'Faisalabad',
    'Sar-\ngodha':     'Sargodha',
    'Baha-\nwalpur':   'Bahawalpur',
    'Hyder-\nabad':    'Hyderabad',
    'Pesha-\nwar':     'Peshawar',
    'Khuz-\ndar':      'Khuzdar',
    'Muzaf-\nfarabad': 'Muzaffarabad',
    'Muz-\naffarabad': 'Muzaffarabad',
    'D.I.\nKhan':      'D.I. Khan',
    'D.G.\nKhan':      'D.G. Khan',
    'Nawa-\nbshah':    'Nawabshah',
    'Jac-\nobabad':    'Jacobabad',
    'Mir-\npur':       'Mirpur',
    'Tur-\nbat':       'Turbat',
    'M.B.\nDin':       'Mandi Bahauddin',
    'Shei-\nkhupura':  'Sheikhupura',
    # newer annexure files use clean names already, handled by fallback below
}

# Patterns that identify non-city columns (averages, % change, dates)
_NON_CITY = re.compile(
    r'average|%change|%\s*change|feb|mar|jan|apr|may|jun|'
    r'jul|aug|sep|oct|nov|dec|\d{4}|over|prices?|change|s\.?\s*no',
    re.IGNORECASE
)


def _clean_city(raw):
    """
    Convert a raw column header to a clean city name.
    Returns None if the column is not a city (e.g. averages, % change).
    """
    if raw is None or str(raw).strip() in ('nan', '', 'NaN'):
        return None
    raw = str(raw).strip()

    # Direct lookup for hyphenated names
    if raw in _CITY_FIXES:
        return _CITY_FIXES[raw]

    # Generic fix: remove line-break hyphens introduced by PDF conversion
    cleaned = re.sub(r'-\n', '', raw).replace('\n', ' ').strip()

    # Drop if it matches non-city patterns
    if _NON_CITY.search(cleaned):
        return None

    # Drop if too short, purely numeric, or clearly not a place name
    if len(cleaned) < 2 or re.match(r'^\d+$', cleaned):
        return None

    return cleaned.title()


# ─────────────────────────────────────────────────────────────
# VALIDATE FILE
# ─────────────────────────────────────────────────────────────

def _is_real_xlsx(filepath):
    """Real XLSX files are ZIP archives starting with PK\x03\x04."""
    try:
        with open(filepath, 'rb') as f:
            return f.read(4) == b'PK\x03\x04'
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# PARSE ONE EXCEL FILE
# ─────────────────────────────────────────────────────────────

def parse_excel(filepath):
    """
    Parse one PBS SPI price Excel file.

    Expected layout:
      Row 0  → title (skipped)
      Row 1  → headers: S.No | Description | Unit | City1 | City2 | ...
      Row 2+ → data:    1    | Wheat Flour  | 20Kg | price | price | ...

    Returns
    -------
    items_df : DataFrame  columns = ['item', 'category'] + city names
    cities   : list of city name strings
    """
    try:
        raw = pd.read_excel(filepath, header=None, engine='openpyxl')
    except Exception as e:
        print(f"    Cannot read: {e}")
        return None, None

    if raw.shape[0] < 3 or raw.shape[1] < 6:
        print(f"    File too small: {raw.shape}")
        return None, None

    # ── Locate the header row ─────────────────────────────────
    # It contains 'Description' in one of the first 5 rows
    header_row_idx = None
    for i in range(min(5, len(raw))):
        row_vals = raw.iloc[i].astype(str).str.strip().str.lower()
        if row_vals.str.contains('description').any():
            header_row_idx = i
            break

    # Fallback: use row 1 if Description not found (older file format)
    if header_row_idx is None:
        header_row_idx = 1

    header = raw.iloc[header_row_idx].tolist()

    # ── Map column index → city name ──────────────────────────
    # Skip cols 0 (S.No), 1 (Description), 2 (Unit)
    city_col_map = {}   # { col_index: city_name }
    for col_i, cell in enumerate(header):
        if col_i < 3:
            continue
        city = _clean_city(cell)
        if city:
            city_col_map[col_i] = city

    if len(city_col_map) < 3:
        print(f"    Only {len(city_col_map)} city columns found — check file format")
        # Debug: print what was in the header row
        print(f"    Header row content: {[str(h)[:15] for h in header[:12]]}")
        return None, None

    cities = list(city_col_map.values())

    # ── Parse item rows ───────────────────────────────────────
    records = []
    data_start = header_row_idx + 1

    for idx in range(data_start, len(raw)):
        row  = raw.iloc[idx].tolist()
        sno  = str(row[0]).strip() if len(row) > 0 else ''
        desc = str(row[1]).strip() if len(row) > 1 else ''
        unit = str(row[2]).strip() if len(row) > 2 else ''

        # Only process numbered rows (1, 2, ... 51)
        if not sno.isdigit():
            continue
        if not desc or desc == 'nan':
            continue

        # Some converters split long descriptions — unit col starts with ')'
        if unit.startswith(')'):
            desc = (desc + unit).strip()

        item_name = re.sub(r'\s+', ' ', desc).strip().lower()

        rec = {'item': item_name, 'category': 'SPI Items'}
        for col_i, city in city_col_map.items():
            if col_i < len(row):
                raw_val = str(row[col_i]).replace(',', '').strip()
                try:
                    price = float(raw_val)
                    rec[city] = price if price > 0 else np.nan
                except ValueError:
                    rec[city] = np.nan
            else:
                rec[city] = np.nan

        records.append(rec)

    if not records:
        print(f"    No item rows extracted")
        return None, None

    items_df = pd.DataFrame(records)
    return items_df, cities


# ─────────────────────────────────────────────────────────────
# WIDE → LONG FORMAT
# ─────────────────────────────────────────────────────────────

def _wide_to_long(items_df, cities, year, month):
    """Convert item × city wide DataFrame to long-format rows."""
    rows = []
    for _, row in items_df.iterrows():
        item = str(row.get('item', '')).strip().lower()
        cat  = str(row.get('category', 'SPI Items')).strip()
        if not item or item == 'nan':
            continue
        for city in cities:
            price = row.get(city, np.nan)
            if pd.notna(price) and price > 0:
                rows.append({
                    'year':     year,
                    'month':    month,
                    'item':     item,
                    'category': cat,
                    'city':     city,
                    'price':    float(price)
                })
    return rows


# ─────────────────────────────────────────────────────────────
# AUDIT FOLDER
# ─────────────────────────────────────────────────────────────

def audit_data_folder(annexures_dir="data/annexures"):
    """
    Scan data/annexures/ and report which files are valid,
    fake, or missing. Call this before loading.
    """
    folder = Path(annexures_dir)
    print("\n" + "="*60)
    print(f"  AUDIT: {folder}")
    print("="*60)

    if not folder.exists():
        print(f"  ✗ Folder not found: {folder}")
        print(f"    Create it and place YYYY_MM.xlsx files inside.")
        print("="*60)
        return [], []

    files = sorted(folder.glob("*.xlsx")) + sorted(folder.glob("*.xls"))
    if not files:
        print(f"  ✗ No Excel files found in {folder}")
        print("="*60)
        return [], []

    valid, invalid = [], []

    for fp in files:
        stem = fp.stem
        m    = re.match(r'^(\d{4})_(\d{2})$', stem)

        if not m:
            print(f"  ✗ SKIP   {fp.name}  — name must be YYYY_MM.xlsx")
            continue

        year, month = int(m.group(1)), int(m.group(2))
        kb = fp.stat().st_size // 1024

        if _is_real_xlsx(fp):
            print(f"  ✓ OK     {fp.name}  ({kb} KB)")
            valid.append((year, month, fp))
        else:
            print(f"  ✗ FAKE   {fp.name}  ({kb} KB)"
                  f"  ← not a real Excel file, delete and re-download")
            invalid.append(fp)

    # Coverage summary
    if valid:
        years_present = sorted(set(y for y, _, _ in valid))
        covered = {(y, mo) for y, mo, _ in valid}
        all_expected = {(y, mo) for y in years_present for mo in range(1, 13)}
        missing = sorted(all_expected - covered)

        print(f"\n  Valid  : {len(valid)}")
        print(f"  Fake   : {len(invalid)}")
        if missing:
            s = ", ".join(f"{y}-{mo:02d}" for y, mo in missing[:12])
            if len(missing) > 12:
                s += f" ... (+{len(missing)-12} more)"
            print(f"  Missing: {len(missing)} months — {s}")

    print("="*60 + "\n")
    return valid, invalid


# ─────────────────────────────────────────────────────────────
# MAIN LOADER
# ─────────────────────────────────────────────────────────────

def load_all_data(annexures_dir="data/annexures"):
    """
    Load every valid YYYY_MM.xlsx from annexures_dir.

    Returns
    -------
    DataFrame with columns: [year, month, item, category, city, price]
    """
    valid_files, _ = audit_data_folder(annexures_dir)

    if not valid_files:
        raise RuntimeError(
            f"No valid Excel files found in '{annexures_dir}'.\n"
            "Name files as YYYY_MM.xlsx  e.g. 2023_03.xlsx"
        )

    all_records = []

    for year, month, fp in sorted(valid_files):
        print(f"  {year}-{month:02d}  {fp.name} ...", end=" ", flush=True)

        items_df, cities = parse_excel(fp)

        if items_df is None or items_df.empty:
            print("FAILED")
            continue

        rows = _wide_to_long(items_df, cities, year, month)
        all_records.extend(rows)
        print(f"OK  ({len(items_df)} items × {len(cities)} cities"
              f" = {len(rows)} records)")

    if not all_records:
        raise RuntimeError(
            "No records loaded from any file.\n"
            "Run:  python src/scraper.py data/annexures/YYYY_MM.xlsx\n"
            "to inspect a single file and debug."
        )

    df = pd.DataFrame(all_records)

    # Normalise text fields
    df['item']     = (df['item'].str.strip()
                               .str.lower()
                               .str.replace(r'\s+', ' ', regex=True))
    df['city']     = df['city'].str.strip()
    df['category'] = df['category'].str.strip()

    # Summary
    print(f"\n{'─'*55}")
    print(f"  Records : {len(df):,}")
    print(f"  Items   : {df['item'].nunique()}")
    print(f"  Cities  : {df['city'].nunique()}")
    print(f"  Years   : {sorted(df['year'].unique())}")
    for yr in sorted(df['year'].unique()):
        months = sorted(df[df['year'] == yr]['month'].unique())
        print(f"            {yr}: {len(months)} months → {months}")
    print(f"{'─'*55}\n")

    return df


# ─────────────────────────────────────────────────────────────
# INSPECT A SINGLE FILE  (debugging helper)
# ─────────────────────────────────────────────────────────────

def inspect_file(filepath):
    """
    Print the raw structure of one Excel file.
    Use this when a file fails to parse — it shows you exactly
    what the converter produced so you can spot layout differences.
    """
    fp = Path(filepath)
    kb = fp.stat().st_size // 1024
    print(f"\n{'='*60}")
    print(f"  File  : {fp.name}  ({kb} KB)")

    if not _is_real_xlsx(fp):
        print("  ✗ NOT a real Excel file (probably HTML saved as .xlsx)")
        with open(fp, 'r', errors='replace') as f:
            print(f"  First 200 chars: {f.read(200)}")
        print('='*60)
        return

    print("  ✓ Valid Excel file")
    raw = pd.read_excel(fp, header=None, engine='openpyxl')
    print(f"  Shape : {raw.shape}")
    print(f"\n  First 5 rows (first 8 cols):")
    for i in range(min(5, len(raw))):
        vals = [str(v)[:18] for v in raw.iloc[i].tolist()[:8]]
        print(f"    Row {i}: {vals}")

    # Show what parse_excel sees
    print(f"\n  Parsing...")
    items_df, cities = parse_excel(fp)
    if items_df is not None:
        print(f"  ✓ Items  : {len(items_df)}")
        print(f"  ✓ Cities : {cities}")
        print(f"\n  Sample (first 5 items, first 3 cities):")
        cols = ['item'] + (cities[:3] if cities else [])
        print(items_df[cols].head().to_string(index=False))
    else:
        print("  ✗ Parsing failed")
    print('='*60)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Inspect a single file: python src/scraper.py data/annexures/2023_03.xlsx
        inspect_file(sys.argv[1])
    else:
        # Load everything: python src/scraper.py
        df = load_all_data("data/annexures")
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/cpi_data.csv", index=False)
        print(f"Saved → data/cpi_data.csv")
        print(df.head(10).to_string(index=False))