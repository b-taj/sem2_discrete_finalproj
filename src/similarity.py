"""
similarity.py
-------------
Computes:
  1. Cosine similarity between every item pair for each (city, year)
  2. N_y(i,j)  — number of cities where sim >= threshold τ
  3. avg_sim_y(i,j) — average cosine similarity across all cities

OUTPUT
------
    data/similarity_counts_yYEAR.pkl   — N_y dict  { (item_i, item_j): count }
    data/similarity_avg_yYEAR.pkl      — avg-sim dict { (item_i, item_j): float }
"""

import numpy as np
import pickle
import os
from itertools import combinations
from src.preprocess import load_vectors, load_metadata


# ─────────────────────────────────────────────────────────────
# COSINE SIMILARITY (single pair)
# ─────────────────────────────────────────────────────────────

def cosine_sim(v1, v2):
    """
    Cosine similarity between two 1-D numpy arrays.
    Returns 0.0 if either vector has zero norm (undefined similarity).
    NaN values are treated as 0.
    """
    v1 = np.nan_to_num(v1, nan=0.0)
    v2 = np.nan_to_num(v2, nan=0.0)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


# ─────────────────────────────────────────────────────────────
# PER-CITY SIMILARITY MATRIX FOR ONE YEAR
# ─────────────────────────────────────────────────────────────

def city_similarity_matrix(vectors, city, year, items):
    """
    Compute cosine similarity for all item pairs in a given city/year.

    Returns
    -------
    sim_dict : dict  { (item_i, item_j) : float }
               Keys are sorted tuples so (A,B) == (B,A).
    """
    # Collect available vectors for this city & year
    item_vecs = {}
    for item in items:
        key = (item, city, year)
        if key in vectors:
            item_vecs[item] = vectors[key]

    available = list(item_vecs.keys())
    sim_dict = {}

    for item_i, item_j in combinations(available, 2):
        pair = (item_i, item_j) if item_i < item_j else (item_j, item_i)
        sim_dict[pair] = cosine_sim(item_vecs[item_i], item_vecs[item_j])

    return sim_dict


# ─────────────────────────────────────────────────────────────
# AGGREGATE ACROSS ALL CITIES
# ─────────────────────────────────────────────────────────────

def aggregate_across_cities(vectors, cities, items, year, tau=0.5):
    """
    For a given year, compute:
      N_y(i,j)       = number of cities where sim(i,j) >= tau
      avg_sim(i,j)   = (1/|C|) * sum_c sim_c(i,j)

    Parameters
    ----------
    vectors : dict from preprocess.py
    cities  : list of city names
    items   : list of item names
    year    : int
    tau     : float, similarity threshold

    Returns
    -------
    N_y      : dict  { (item_i, item_j) : int }
    avg_sim  : dict  { (item_i, item_j) : float }
    """
    N_y      = {}     # city count where sim >= tau
    sum_sim  = {}     # running sum of similarities
    count_c  = {}     # how many cities had both items

    total_cities = len(cities)
    print(f"  Year {year}: processing {total_cities} cities × "
          f"{len(items)} items ...", end=" ", flush=True)

    for city in cities:
        sim_dict = city_similarity_matrix(vectors, city, year, items)

        for pair, sim in sim_dict.items():
            # Count cities
            sum_sim[pair]  = sum_sim.get(pair, 0.0) + sim
            count_c[pair]  = count_c.get(pair, 0) + 1

            # Threshold check
            if sim >= tau:
                N_y[pair] = N_y.get(pair, 0) + 1

    # Build avg_sim over ALL cities (pairs with no data for a city count as 0)
    avg_sim = {}
    for pair in sum_sim:
        avg_sim[pair] = sum_sim[pair] / total_cities

    print(f"done. {len(N_y)} pairs exceed τ={tau}")
    return N_y, avg_sim


# ─────────────────────────────────────────────────────────────
# COMPUTE FOR ALL YEARS
# ─────────────────────────────────────────────────────────────

def compute_all_years(vectors, metadata, tau=0.5):
    """
    Run aggregate_across_cities for every year in metadata.

    Returns
    -------
    N_all      : dict  { year : N_y_dict }
    avg_all    : dict  { year : avg_sim_dict }
    """
    cities = metadata['cities']
    items  = metadata['items']
    years  = metadata['years']

    N_all   = {}
    avg_all = {}

    print(f"\nComputing similarities (τ={tau}) ...")
    for year in years:
        N_y, avg_sim = aggregate_across_cities(
            vectors, cities, items, year, tau=tau
        )
        N_all[year]   = N_y
        avg_all[year] = avg_sim

    print("Similarity computation complete.\n")
    return N_all, avg_all


# ─────────────────────────────────────────────────────────────
# SENSITIVITY: RECOMPUTE N_y FOR A DIFFERENT τ WITHOUT RELOADING
# ─────────────────────────────────────────────────────────────

def recount_with_tau(vectors, cities, items, year, new_tau):
    """
    Quickly recompute N_y for a single year under a new threshold.
    Useful for threshold sensitivity analysis (Section 11 of project spec).
    """
    N_y = {}
    for city in cities:
        sim_dict = city_similarity_matrix(vectors, city, year, items)
        for pair, sim in sim_dict.items():
            if sim >= new_tau:
                N_y[pair] = N_y.get(pair, 0) + 1
    return N_y


# ─────────────────────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────────────────────

def save_similarities(N_all, avg_all, out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    for year in N_all:
        with open(f"{out_dir}/similarity_counts_y{year}.pkl", 'wb') as f:
            pickle.dump(N_all[year], f)
        with open(f"{out_dir}/similarity_avg_y{year}.pkl", 'wb') as f:
            pickle.dump(avg_all[year], f)
        print(f"Saved similarity data for year {year}")


def load_similarities(years, data_dir="data"):
    N_all   = {}
    avg_all = {}
    for year in years:
        with open(f"{data_dir}/similarity_counts_y{year}.pkl", 'rb') as f:
            N_all[year] = pickle.load(f)
        with open(f"{data_dir}/similarity_avg_y{year}.pkl", 'rb') as f:
            avg_all[year] = pickle.load(f)
    return N_all, avg_all


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== SIMILARITY COMPUTATION ===\n")

    vectors  = load_vectors("data/price_change_vectors.pkl")
    metadata = load_metadata("data/metadata.json")

    TAU = 0.5   # default threshold — change for sensitivity analysis

    N_all, avg_all = compute_all_years(vectors, metadata, tau=TAU)

    save_similarities(N_all, avg_all)

    # Quick stats
    for year in metadata['years']:
        n_pairs = len(N_all[year])
        print(f"Year {year}: {n_pairs} item-pairs with sim >= {TAU} in ≥1 city")

    print("\n=== DONE ===")