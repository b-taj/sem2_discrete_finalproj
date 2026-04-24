"""
scraper.py  —  FINAL VERSION
==============================
Handles both formats confirmed working:
  - data/pdfs/*.pdf         (PBS monthly SPI price tables)
  - data/annexures/*.xlsx   (PBS annexure Excel files)

File naming: YYYY_MM.pdf / YYYY_MM.xlsx
e.g. 2023_03.pdf, 2025_07.xlsx

Usage:
    python src/scraper.py                        # load everything → data/cpi_data.csv
    python src/scraper.py data/pdfs/2023_03.pdf  # inspect one file
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
# FILE VALIDATION
# ─────────────────────────────────────────────────────────────

def is_real_pdf(filepath):
    try:
        with open(filepath, "rb") as f:
            return f.read(5) == b"%PDF-"
    except Exception:
        return False

def is_real_xlsx(filepath):
    try:
        with open(filepath, "rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# CITY NAME CLEANUP
# ─────────────────────────────────────────────────────────────
# PBS PDFs wrap city names across two lines with a hyphen.
# e.g. 'Islam-\nabad' → 'Islamabad'

_CITY_FIXES = {
    'Islam-\nabad':     'Islamabad',
    'Rawal-\npindi':    'Rawalpindi',
    'Gujran-\nwala':    'Gujranwala',
    'Faisal-\nabad':    'Faisalabad',
    'Sar-\ngodha':      'Sargodha',
    'Baha-\nwalpur':    'Bahawalpur',
    'Hyder-\nabad':     'Hyderabad',
    'Pesha-\nwar':      'Peshawar',
    'Khuz-\ndar':       'Khuzdar',
    'Muzaf-\nfarabad':  'Muzaffarabad',
    'Muz-\naffarabad':  'Muzaffarabad',
    'D.I.\nKhan':       'D.I. Khan',
    'D.G.\nKhan':       'D.G. Khan',
    'Nawa-\nbshah':     'Nawabshah',
    'Jac-\nobabad':     'Jacobabad',
    'Mir-\npur':        'Mirpur',
    'Gilgit\nCity':     'Gilgit',
    'Tur-\nbat':        'Turbat',
    'M.B.\nDin':        'Mandi Bahauddin',
    'Shei-\nkhupura':   'Sheikhupura',
    'Sahiwal\nCity':    'Sahiwal',
    'Dera\nIsmail':     'D.I. Khan',
    'Dera\nGhazi':      'D.G. Khan',
}

# Non-city trailing columns to drop
_NON_CITY_PATTERNS = re.compile(
    r'average|%change|feb|mar|jan|apr|may|jun|jul|aug|sep|oct|nov|dec'
    r'|\d{4}|over|prices?|change',
    re.IGNORECASE
)

def clean_city_name(raw):
    """Convert raw column header to clean city name, or None if not a city."""
    if not raw:
        return None
    raw = str(raw).strip()
    if raw in _CITY_FIXES:
        return _CITY_FIXES[raw]
    # Generic: remove line-break hyphens and newlines
    cleaned = re.sub(r'-\n', '', raw).replace('\n', ' ').strip()
    # Drop if it matches non-city patterns
    if _NON_CITY_PATTERNS.search(cleaned):
        return None
    # Drop if too short or purely numeric
    if len(cleaned) < 3 or re.match(r'^\d+$', cleaned):
        return None
    return cleaned.title()


# ─────────────────────────────────────────────────────────────
# PDF PARSER
# ─────────────────────────────────────────────────────────────
#
# PBS SPI price tables have this structure:
#   col 0 : S. No.  (row number like '1', '2', ...)
#   col 1 : Description (item name — may overflow into col 2)
#   col 2 : Unit  (sometimes contains overflow from description)
#   col 3+ : City prices  (until 'Average Prices' columns start)
#
# The table always has exactly 51 item rows + 1 header row.

def parse_pdf(filepath):
    """
    Parse a PBS SPI monthly price PDF.
    Returns (items_df, cities) or (None, None).
    """
    try:
        import pdfplumber
    except ImportError:
        print("    ✗ pdfplumber not installed. Run: pip install pdfplumber")
        return None, None

    records = []
    cities  = None

    try:
        with pdfplumber.open(str(filepath)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if len(table) < 5:
                        continue

                    header = table[0]
                    if not header or len(header) < 6:
                        continue

                    # ── Detect city columns from header ──────────────────
                    city_col_map = {}   # col_index → city_name
                    for col_idx, cell in enumerate(header):
                        if col_idx < 3:   # skip S.No, Description, Unit
                            continue
                        city = clean_city_name(cell)
                        if city:
                            city_col_map[col_idx] = city

                    if len(city_col_map) < 3:
                        continue   # not a price table

                    page_cities = list(city_col_map.values())
                    if cities is None:
                        cities = page_cities

                    # ── Parse data rows ───────────────────────────────────
                    for row in table[1:]:
                        if not row:
                            continue

                        sno  = str(row[0] or '').strip()
                        desc = str(row[1] or '').strip()
                        unit = str(row[2] or '').strip()

                        # Only process numbered item rows
                        if not sno.isdigit():
                            continue
                        if not desc:
                            continue

                        # Fix description overflow into unit column
                        # (happens when item name is long — unit starts with ')')
                        if unit.startswith(')'):
                            desc = (desc + unit).strip()
                        # Clean item name
                        item_name = re.sub(r'\s+', ' ', desc).strip().lower()

                        # Extract prices for each city column
                        rec = {'item': item_name, 'category': 'SPI Items'}
                        for col_idx, city in city_col_map.items():
                            if col_idx < len(row):
                                raw_val = str(row[col_idx] or '').replace(',', '').strip()
                                try:
                                    price = float(raw_val)
                                    rec[city] = price if price > 0 else np.nan
                                except ValueError:
                                    rec[city] = np.nan
                            else:
                                rec[city] = np.nan
                        records.append(rec)

    except Exception as e:
        print(f"    ✗ PDF parse error: {e}")
        return None, None

    if not records:
        return None, None

    df = pd.DataFrame(records)
    return df, cities


# ─────────────────────────────────────────────────────────────
# EXCEL PARSER
# ─────────────────────────────────────────────────────────────
#
# PBS Annexure Excel files have a similar layout but as spreadsheets.
# The city header row is auto-detected by finding 3+ known city names.

_CITY_KEYWORDS = [
    'karachi', 'lahore', 'islamabad', 'rawalpindi', 'faisalabad',
    'multan', 'peshawar', 'quetta', 'hyderabad', 'sialkot',
    'gujranwala', 'sukkur', 'larkana', 'bahawalpur', 'sargodha',
]

def _fix_excel_city(name):
    """Fix hyphenated city names from Excel merged cells."""
    joined = str(name).replace('-', '').replace(' ', '').lower()
    lookup = {
        'islamabad':    'Islamabad',   'rawalpindi':   'Rawalpindi',
        'gujranwala':   'Gujranwala',  'sialkot':      'Sialkot',
        'lahore':       'Lahore',      'faisalabad':   'Faisalabad',
        'sargodha':     'Sargodha',    'multan':       'Multan',
        'bahawalpur':   'Bahawalpur',  'karachi':      'Karachi',
        'hyderabad':    'Hyderabad',   'sukkur':       'Sukkur',
        'larkana':      'Larkana',     'peshawar':     'Peshawar',
        'bannu':        'Bannu',       'quetta':       'Quetta',
        'khuzdar':      'Khuzdar',     'turbat':       'Turbat',
        'zhob':         'Zhob',        'gilgit':       'Gilgit',
        'muzaffarabad': 'Muzaffarabad','mirpur':       'Mirpur',
        'nawabshah':    'Nawabshah',   'jacobabad':    'Jacobabad',
        'sahiwal':      'Sahiwal',     'sheikhupura':  'Sheikhupura',
        'chiniot':      'Chiniot',     'okara':        'Okara',
        'dikhan':       'D.I. Khan',   'dgkhan':       'D.G. Khan',
        'mandibahauddin':'Mandi Bahauddin',
    }
    return lookup.get(joined, str(name).strip().title())

def parse_excel(filepath):
    """
    Parse a PBS CPI Annexure Excel file.
    Returns (items_df, cities) or (None, None).
    """
    try:
        raw = pd.read_excel(filepath, header=None, engine='openpyxl')
    except Exception as e:
        print(f"    ✗ Excel read error: {e}")
        return None, None

    # ── Find header row ───────────────────────────────────────
    header_row = None
    for i, row in raw.iterrows():
        row_lower = row.astype(str).str.strip().str.lower()
        hits = sum(
            row_lower.str.contains(city, na=False).any()
            for city in _CITY_KEYWORDS
        )
        if hits >= 3:
            header_row = i
            break

    if header_row is None:
        print(f"    ✗ Cannot find city header row")
        return None, None

    # Re-read with header
    df = pd.read_excel(filepath, header=header_row, engine='openpyxl')
    df.columns = [str(c).strip() for c in df.columns]
    all_cols   = list(df.columns)

    # ── Identify item column (first column with real text names) ─
    item_col = all_cols[0]
    for col in all_cols[:4]:
        # Check if this column has non-numeric values that look like item names
        sample = df[col].dropna().astype(str).str.strip()
        sample = sample[sample != 'nan']
        non_numeric = sample[~sample.str.match(r'^\d+\.?\d*$')]
        if len(non_numeric) >= 5:
            item_col = col
            break

    # ── Identify city columns ─────────────────────────────────
    city_cols = {}   # original_name → clean_name
    for c in all_cols:
        if c == item_col:
            continue
        c_str = str(c).strip()
        if c_str in ('nan', '', 'NaN') or c_str.lower().startswith('unnamed'):
            continue
        fixed = _fix_excel_city(c_str)
        # Only keep if it looks like a city (not a stat column)
        if not _NON_CITY_PATTERNS.search(c_str):
            city_cols[c] = fixed

    if len(city_cols) < 3:
        print(f"    ✗ Found only {len(city_cols)} city columns")
        return None, None

    # ── Parse rows ────────────────────────────────────────────
    current_cat = 'Unknown'
    records     = []

    for _, row in df.iterrows():
        name = str(row[item_col]).strip()
        if name in ('nan', '', 'NaN', 'None'):
            continue
        # Skip pure row numbers
        if re.match(r'^\d+\.?\d*$', name):
            continue

        # Check if any city column has a numeric value
        numeric_found = False
        for orig_c in city_cols:
            val = str(row.get(orig_c, '')).replace(',', '').strip()
            try:
                float(val)
                numeric_found = True
                break
            except ValueError:
                pass

        if not numeric_found:
            current_cat = name   # it's a category heading
            continue

        rec = {'item': name.lower().strip(), 'category': current_cat}
        for orig_c, clean_c in city_cols.items():
            val = str(row.get(orig_c, '')).replace(',', '').strip()
            try:
                price = float(val)
                rec[clean_c] = price if price > 0 else np.nan
            except ValueError:
                rec[clean_c] = np.nan
        records.append(rec)

    if not records:
        return None, None

    df_out = pd.DataFrame(records)
    cities = list(city_cols.values())
    return df_out, cities


# ─────────────────────────────────────────────────────────────
# AUDIT
# ─────────────────────────────────────────────────────────────

def audit_data_folder(base_dir="data"):
    """Scan folders, validate files, report coverage."""
    print("\n" + "="*65)
    print("  DATA AUDIT")
    print("="*65)

    folders = {
        "PDF":   Path(base_dir) / "pdfs",
        "Excel": Path(base_dir) / "annexures",
    }

    valid_files   = []
    invalid_files = []

    for fmt, folder in folders.items():
        if not folder.exists():
            print(f"\n  [{fmt}] Folder not found: {folder}")
            continue

        files = sorted(folder.glob("*"))
        if not files:
            print(f"\n  [{fmt}] Folder empty")
            continue

        print(f"\n  [{fmt}] {folder}  ({len(files)} files)")

        for fp in files:
            stem   = fp.stem
            suffix = fp.suffix.lower()
            m      = re.match(r'^(\d{4})_(\d{2})$', stem)
            if not m:
                print(f"    ✗ SKIP  {fp.name}  — must be YYYY_MM{suffix}")
                continue

            year, month = int(m.group(1)), int(m.group(2))
            valid = is_real_xlsx(fp) if suffix in ('.xlsx','.xls') else is_real_pdf(fp)

            kb = fp.stat().st_size // 1024
            if valid:
                print(f"    ✓ OK    {fp.name}  ({kb} KB)")
                valid_files.append((year, month, fp, fmt))
            else:
                print(f"    ✗ FAKE  {fp.name}  ({kb} KB)  ← delete and re-download")
                invalid_files.append(fp)

    # Coverage check across years present in valid_files
    if valid_files:
        years_present = sorted(set(y for y, _, _, _ in valid_files))
        all_months = set()
        for y in years_present:
            for mo in range(1, 13):
                all_months.add((y, mo))
        covered = {(y, m) for y, m, _, _ in valid_files}
        missing = sorted(all_months - covered)

        print(f"\n  Valid   : {len(valid_files)}")
        print(f"  Fake    : {len(invalid_files)}")
        print(f"  Missing : {len(missing)} months")
        if missing:
            s = ", ".join(f"{y}-{m:02d}" for y, m in missing[:12])
            if len(missing) > 12:
                s += f" ... (+{len(missing)-12})"
            print(f"            {s}")

    print("="*65 + "\n")
    return valid_files, invalid_files


# ─────────────────────────────────────────────────────────────
# WIDE → LONG
# ─────────────────────────────────────────────────────────────

def wide_to_long(items_df, cities, year, month):
    rows = []
    for _, row in items_df.iterrows():
        item = str(row.get('item', '')).strip().lower()
        cat  = str(row.get('category', 'SPI Items')).strip()
        if not item or item in ('nan', ''):
            continue
        for city in cities:
            price = row.get(city, np.nan)
            if pd.notna(price) and price > 0:
                rows.append({
                    'year': year, 'month': month,
                    'item': item, 'category': cat,
                    'city': city, 'price': float(price)
                })
    return rows


# ─────────────────────────────────────────────────────────────
# MAIN LOADER
# ─────────────────────────────────────────────────────────────

def load_all_data(base_dir="data"):
    """
    Load all valid YYYY_MM.pdf and YYYY_MM.xlsx files.
    Returns long-format DataFrame: [year, month, item, category, city, price]
    """
    valid_files, _ = audit_data_folder(base_dir)

    if not valid_files:
        raise RuntimeError(
            "No valid files found.\n"
            "Place files as data/pdfs/YYYY_MM.pdf or data/annexures/YYYY_MM.xlsx"
        )

    all_records = []

    for year, month, fp, fmt in sorted(valid_files):
        print(f"  {year}-{month:02d} [{fmt:5s}] {fp.name} ...", end=" ", flush=True)

        if fmt == "Excel":
            items_df, cities = parse_excel(fp)
        else:
            items_df, cities = parse_pdf(fp)

        if items_df is None or items_df.empty or not cities:
            print("FAILED")
            continue

        rows = wide_to_long(items_df, cities, year, month)
        all_records.extend(rows)
        print(f"OK  ({len(items_df)} items × {len(cities)} cities = {len(rows)} records)")

    if not all_records:
        raise RuntimeError("No records loaded. Check output above for per-file errors.")

    df = pd.DataFrame(all_records)

    # Normalize
    df['item']     = df['item'].str.strip().str.lower().str.replace(r'\s+', ' ', regex=True)
    df['city']     = df['city'].str.strip()
    df['category'] = df['category'].str.strip()

    print(f"\n{'─'*55}")
    print(f"  Records : {len(df):,}")
    print(f"  Items   : {df['item'].nunique()}")
    print(f"  Cities  : {df['city'].nunique()}")
    print(f"  Years   : {sorted(df['year'].unique())}")
    for yr in sorted(df['year'].unique()):
        mo = sorted(df[df['year'] == yr]['month'].unique())
        print(f"            {yr}: months {mo}")
    print(f"{'─'*55}\n")

    return df


# ─────────────────────────────────────────────────────────────
# INSPECT HELPER
# ─────────────────────────────────────────────────────────────

def inspect_file(filepath):
    """Quick diagnostic on a single file."""
    fp = Path(filepath)
    suffix = fp.suffix.lower()
    kb = fp.stat().st_size // 1024

    print(f"\n{'='*65}")
    print(f"  File  : {fp.name}  ({kb} KB)")

    if suffix == '.pdf':
        if not is_real_pdf(fp):
            print("  ✗ NOT a real PDF — probably HTML"); return
        print("  ✓ Real PDF")
        items_df, cities = parse_pdf(fp)
    elif suffix in ('.xlsx', '.xls'):
        if not is_real_xlsx(fp):
            print("  ✗ NOT a real Excel — probably HTML"); return
        print("  ✓ Real Excel")
        items_df, cities = parse_excel(fp)
    else:
        print(f"  ✗ Unknown format: {suffix}"); return

    if items_df is None:
        print("  ✗ Parsing FAILED")
    else:
        print(f"  ✓ Items  : {len(items_df)}")
        print(f"  ✓ Cities : {cities}")
        print(f"\n  First 5 items:")
        print(items_df[['item','category']].head().to_string(index=False))
        print(f"\n  Sample prices (first item, first 4 cities):")
        if len(items_df) > 0:
            row = items_df.iloc[0]
            for c in (cities or [])[:4]:
                print(f"    {c}: {row.get(c, 'N/A')}")
    print('='*65)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) == 2:
        inspect_file(sys.argv[1])
    else:
        df = load_all_data("data")
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/cpi_data.csv", index=False)
        print(f"Saved → data/cpi_data.csv")
        print(df.head(10).to_string(index=False))