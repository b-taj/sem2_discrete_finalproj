"""
scraper.py — reads PBS CPI monthly review PDFs
Each PDF contains "Annexure A" with item-level CPI indices per city.
"""

import pdfplumber
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path


def extract_annexure_from_pdf(pdf_path):
    """
    Extract city-wise CPI index table from a PBS monthly review PDF.
    The table appears after the heading 'Annexure A' or 'Annexure-1'.
    
    Returns DataFrame: columns = ['item', 'category', city1, city2, ...]
    """
    records = []
    current_category = "Unknown"
    in_annexure = False
    cities = None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            # Detect start of Annexure section
            if re.search(r'Annexure[\s\-]*[A1]', text, re.IGNORECASE):
                in_annexure = True

            if not in_annexure:
                continue

            # Extract table from this page
            table = page.extract_table()
            if not table:
                continue

            for row in table:
                if not row or all(c is None or str(c).strip() == '' for c in row):
                    continue

                row_clean = [str(c).strip() if c else '' for c in row]
                first_cell = row_clean[0]

                # Detect city header row
                if any(city in row_clean for city in
                       ['Karachi', 'Lahore', 'Islamabad', 'Rawalpindi']):
                    cities = [c for c in row_clean if c and c != '']
                    continue

                if cities is None:
                    continue

                # Detect category rows (non-numeric rest of row)
                numeric_vals = [c for c in row_clean[1:] if c]
                all_text = all(not re.match(r'^\d+\.?\d*$', v) for v in numeric_vals)

                if all_text or len(numeric_vals) == 0:
                    if first_cell:
                        current_category = first_cell
                    continue

                # Item row — build record
                record = {'item': first_cell, 'category': current_category}
                for i, city in enumerate(cities[1:], start=1):
                    if i < len(row_clean):
                        try:
                            record[city] = float(row_clean[i].replace(',', ''))
                        except ValueError:
                            record[city] = np.nan
                    else:
                        record[city] = np.nan

                records.append(record)

    return pd.DataFrame(records)


def load_all_months(pdf_dir="data/pdfs"):
    """
    Load all YYYY_MM.pdf files and return long-format DataFrame.
    columns: [year, month, item, category, city, price]
    """
    all_records = []
    pdf_path = Path(pdf_dir)
    files = sorted(pdf_path.glob("*.pdf"))

    if not files:
        raise FileNotFoundError(f"No PDFs found in {pdf_dir}")

    print(f"Found {len(files)} PDF files")

    for filepath in files:
        stem = filepath.stem
        parts = stem.split("_")
        if len(parts) != 2:
            continue
        year, month = int(parts[0]), int(parts[1])
        print(f"  Parsing {year}-{month:02d}...", end=" ")

        try:
            df = extract_annexure_from_pdf(filepath)
            cities = [c for c in df.columns if c not in ('item', 'category')]
            for _, row in df.iterrows():
                for city in cities:
                    price = row.get(city, np.nan)
                    if pd.notna(price) and price > 0:
                        all_records.append({
                            'year': year, 'month': month,
                            'item': str(row['item']).strip().lower(),
                            'category': row['category'],
                            'city': city, 'price': price
                        })
            print(f"OK — {len(df)} items")
        except Exception as e:
            print(f"FAILED: {e}")

    df_long = pd.DataFrame(all_records)
    print(f"\nTotal: {len(df_long):,} records, "
          f"{df_long['item'].nunique()} items, "
          f"{df_long['city'].nunique()} cities")
    return df_long


if __name__ == "__main__":
    df = load_all_months("data/pdfs")
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/cpi_data.csv", index=False)
    print("Saved → data/cpi_data.csv")