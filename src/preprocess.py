"""
preprocess.py
-------------
Reads the cleaned long-format CPI CSV and computes:
  - Price-change vectors  v(y)_{i,c}  for each (item, city, year)
  - A summary of items/cities available per year

OUTPUT
------
    data/price_change_vectors.pkl   — dict keyed (item, city, year) → np.ndarray shape (11,)
    data/metadata.json              — items, cities, years lists
"""

import pandas as pd
import numpy as np
import pickle
import json
import os


# ─────────────────────────────────────────────────────────────
# LOAD CSV
# ─────────────────────────────────────────────────────────────

def load_cpi_data(csv_path="data/cpi_data.csv"):
    """Load the long-format CPI data produced by scraper.py."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"'{csv_path}' not found. Run scraper.py first."
        )
    df = pd.read_csv(csv_path)
    required = {'year', 'month', 'item', 'category', 'city', 'price'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df


# ─────────────────────────────────────────────────────────────
# FILTER COMPLETE SERIES
# ─────────────────────────────────────────────────────────────

def filter_complete_series(df, required_months=12):
    """
    Keep only (item, city, year) combinations that have all
    required_months of data — needed to compute a full Δp vector.
    """
    counts = (
        df.groupby(['item', 'city', 'year'])['month']
        .nunique()
        .reset_index(name='month_count')
    )
    complete = counts[counts['month_count'] >= required_months]
    df_clean = df.merge(complete[['item', 'city', 'year']], on=['item', 'city', 'year'])
    
    dropped = len(df['item'].unique()) - len(df_clean['item'].unique())
    print(f"filter_complete_series: kept {len(df_clean):,} rows "
          f"({dropped} items lost some city-year combos)")
    return df_clean


# ─────────────────────────────────────────────────────────────
# COMPUTE PRICE-CHANGE VECTORS
# ─────────────────────────────────────────────────────────────

def compute_price_change_vectors(df):
    """
    For every (item, city, year) group, compute the monthly
    price-change vector:

        v(y)_{i,c} = (Δp_2, Δp_3, ..., Δp_12)   shape = (11,)

    where  Δp_m = price_month_m  −  price_month_(m-1)

    Returns
    -------
    vectors : dict  { (item, city, year) : np.ndarray shape (11,) }
    """
    vectors = {}
    skipped = 0

    for (item, city, year), group in df.groupby(['item', 'city', 'year']):
        group_sorted = group.sort_values('month')
        prices = group_sorted['price'].values          # up to 12 values

        if len(prices) < 2:
            skipped += 1
            continue

        delta = np.diff(prices)                        # month-to-month differences

        # Pad with NaN if fewer than 11 diffs (shouldn't happen after filter)
        if len(delta) < 11:
            delta = np.concatenate([delta, np.full(11 - len(delta), np.nan)])

        vectors[(item, city, year)] = delta[:11]       # always length 11

    print(f"compute_price_change_vectors: {len(vectors):,} vectors created, "
          f"{skipped} skipped (< 2 months)")
    return vectors


# ─────────────────────────────────────────────────────────────
# BUILD METADATA
# ─────────────────────────────────────────────────────────────

def build_metadata(df, vectors):
    """
    Extract the lists of items, cities, years and category mappings
    that are actually present in the vectors dictionary.
    """
    keys = list(vectors.keys())
    items    = sorted(set(k[0] for k in keys))
    cities   = sorted(set(k[1] for k in keys))
    years    = sorted(set(k[2] for k in keys))

    # item → category mapping (take first occurrence)
    cat_map = (
        df[['item', 'category']]
        .drop_duplicates('item')
        .set_index('item')['category']
        .to_dict()
    )

    metadata = {
        'items':    items,
        'cities':   cities,
        'years':    years,
        'category_map': cat_map
    }

    print(f"Metadata: {len(items)} items, {len(cities)} cities, years={years}")
    return metadata


# ─────────────────────────────────────────────────────────────
# ZERO-VECTOR CHECK
# ─────────────────────────────────────────────────────────────

def remove_zero_vectors(vectors):
    """
    Drop vectors that are all-zero or all-NaN — cosine similarity
    is undefined for them and they add noise to the graph.
    """
    cleaned = {}
    removed = 0
    for key, vec in vectors.items():
        if np.all(vec == 0) or np.all(np.isnan(vec)):
            removed += 1
        else:
            cleaned[key] = vec
    print(f"remove_zero_vectors: removed {removed}, kept {len(cleaned)}")
    return cleaned


# ─────────────────────────────────────────────────────────────
# SAVE / LOAD HELPERS
# ─────────────────────────────────────────────────────────────

def save_vectors(vectors, path="data/price_change_vectors.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(vectors, f)
    print(f"Saved vectors → {path}")


def load_vectors(path="data/price_change_vectors.pkl"):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_metadata(metadata, path="data/metadata.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata → {path}")


def load_metadata(path="data/metadata.json"):
    with open(path, 'r') as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# SUMMARY STATS (optional diagnostic)
# ─────────────────────────────────────────────────────────────

def vector_summary(vectors, metadata):
    """Print basic stats about the price-change vectors per year."""
    print("\n── Vector Summary ──────────────────────────────")
    for year in metadata['years']:
        year_vecs = {k: v for k, v in vectors.items() if k[2] == year}
        norms = [np.linalg.norm(v) for v in year_vecs.values()]
        print(f"  Year {year}: {len(year_vecs):>5} vectors | "
              f"mean ‖v‖ = {np.mean(norms):.3f} | "
              f"max ‖v‖ = {np.max(norms):.3f}")
    print("────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== PREPROCESSING ===\n")

    df = load_cpi_data("data/cpi_data.csv")
    print(f"Loaded {len(df):,} rows, columns: {list(df.columns)}\n")

    df = filter_complete_series(df)

    vectors = compute_price_change_vectors(df)
    vectors = remove_zero_vectors(vectors)

    metadata = build_metadata(df, vectors)
    vector_summary(vectors, metadata)

    save_vectors(vectors)
    save_metadata(metadata)

    print("\n=== DONE ===")