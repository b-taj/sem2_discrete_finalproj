"""
analysis.py
-----------
Computes and reports:
  - Degree, closeness, betweenness centrality per year (Section 8)
  - Temporal comparison across years (Section 9)
  - Category connectivity analysis (Section 10)
  - Centrality change under different weights (Section 11)

OUTPUT
------
    data/centrality_yYEAR_VARIANT.csv   — centrality table per year/variant
    data/temporal_analysis.csv          — edge stability summary
    data/category_analysis.csv          — intra / inter category edges
"""

import pandas as pd
import numpy as np
import networkx as nx
import os
import json


# ─────────────────────────────────────────────────────────────
# CENTRALITY COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_centrality(G, weighted=False):
    """
    Compute degree, closeness, and betweenness centrality for graph G.

    Parameters
    ----------
    G        : nx.Graph
    weighted : bool — if True, use 'weight' attribute for centrality

    Returns
    -------
    DataFrame sorted by degree_centrality descending.
    Columns: item, degree_centrality, closeness_centrality,
             betweenness_centrality, degree (raw)
    """
    weight_arg = 'weight' if weighted else None

    degree_cent      = nx.degree_centrality(G)
    closeness_cent   = nx.closeness_centrality(G, distance=weight_arg)
    betweenness_cent = nx.betweenness_centrality(G, weight=weight_arg, normalized=True)

    # Raw degree (useful for ranking ties)
    raw_degree = dict(G.degree())

    rows = []
    for node in G.nodes():
        cat = G.nodes[node].get('category', 'Unknown')
        rows.append({
            'item':                   node,
            'category':               cat,
            'degree':                 raw_degree[node],
            'degree_centrality':      round(degree_cent[node], 6),
            'closeness_centrality':   round(closeness_cent[node], 6),
            'betweenness_centrality': round(betweenness_cent[node], 6),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('degree_centrality', ascending=False).reset_index(drop=True)
    return df


def compute_centrality_all_years(graphs, weighted=False, variant='unweighted'):
    """
    Compute centrality for all years for one graph variant.

    Returns
    -------
    cent_all : dict  { year : DataFrame }
    """
    cent_all = {}
    for year, variants in graphs.items():
        G = variants[variant]
        cent_df = compute_centrality(G, weighted=weighted)
        cent_all[year] = cent_df
        print(f"Year {year} [{variant}] — top 5 by degree centrality:")
        print(cent_df[['item', 'category', 'degree_centrality']].head(5).to_string(index=False))
        print()
    return cent_all


def save_centrality(cent_all, variant, out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    for year, df in cent_all.items():
        path = f"{out_dir}/centrality_y{year}_{variant}.csv"
        df.to_csv(path, index=False)
        print(f"Saved → {path}")


# ─────────────────────────────────────────────────────────────
# HIGHLY CENTRAL ITEMS INTERPRETATION
# ─────────────────────────────────────────────────────────────

def top_central_items(cent_all, top_n=10):
    """
    Print the top-N most central items per year and identify
    items that remain highly central across all years (persistent leaders).
    """
    print("\n── Top Central Items Per Year ────────────────────────")
    top_sets = {}
    for year, df in sorted(cent_all.items()):
        top = df.head(top_n)['item'].tolist()
        top_sets[year] = set(top)
        print(f"\n  Year {year}:")
        for i, row in df.head(top_n).iterrows():
            print(f"    {i+1:>2}. {row['item']:<35} "
                  f"deg={row['degree_centrality']:.4f}  "
                  f"btw={row['betweenness_centrality']:.4f}")

    # Items in top-N across ALL years
    if len(top_sets) > 1:
        persistent = set.intersection(*top_sets.values())
        print(f"\n  Persistent top-{top_n} items across ALL years: {sorted(persistent)}")
    print("──────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────
# TEMPORAL ANALYSIS
# ─────────────────────────────────────────────────────────────

def temporal_edge_analysis(graphs, variant='unweighted'):
    """
    Analyse edge appearance/disappearance across years.
    Fulfils Section 9 of the project spec.

    Returns
    -------
    summary_df : DataFrame with edge stability stats
    """
    years = sorted(graphs.keys())
    edge_sets = {}
    for year in years:
        G = graphs[year][variant]
        edge_sets[year] = set(
            tuple(sorted(e)) for e in G.edges()
        )

    rows = []

    # Pairwise year comparisons
    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i+1]
        e1, e2 = edge_sets[y1], edge_sets[y2]
        rows.append({
            'comparison':    f"{y1}→{y2}",
            'edges_y1':      len(e1),
            'edges_y2':      len(e2),
            'appeared':      len(e2 - e1),        # new edges in y2
            'disappeared':   len(e1 - e2),        # edges lost in y2
            'stable':        len(e1 & e2),        # edges in both
            'jaccard':       round(len(e1 & e2) / len(e1 | e2), 4) if e1 | e2 else 0
        })

    # Edges present in ALL years
    all_years_edges = set.intersection(*edge_sets.values()) if edge_sets else set()
    rows.append({
        'comparison':  'ALL YEARS',
        'edges_y1':    '-',
        'edges_y2':    '-',
        'appeared':    '-',
        'disappeared': '-',
        'stable':      len(all_years_edges),
        'jaccard':     '-'
    })

    summary_df = pd.DataFrame(rows)

    print("\n── Temporal Edge Analysis ─────────────────────────────")
    print(summary_df.to_string(index=False))
    print(f"\n  Persistent edges (all years): {sorted(all_years_edges)[:10]} ...")
    print("──────────────────────────────────────────────────────\n")

    return summary_df, all_years_edges


def temporal_centrality_stability(cent_all, top_n=10):
    """
    Check whether the top-N central items change across years.
    Returns a DataFrame showing rank of each top item per year.
    """
    years = sorted(cent_all.keys())

    # Collect union of top-N items across all years
    all_top = set()
    for df in cent_all.values():
        all_top.update(df.head(top_n)['item'].tolist())

    records = []
    for item in sorted(all_top):
        row = {'item': item}
        for year in years:
            df = cent_all[year].reset_index(drop=True)
            match = df[df['item'] == item]
            row[f'rank_{year}'] = int(match.index[0]) + 1 if len(match) > 0 else None
            row[f'deg_{year}']  = round(match['degree_centrality'].values[0], 4) \
                                   if len(match) > 0 else 0.0
        records.append(row)

    stability_df = pd.DataFrame(records).sort_values(f'rank_{years[0]}')

    print("\n── Centrality Rank Stability ─────────────────────────")
    print(stability_df.to_string(index=False))
    print("──────────────────────────────────────────────────────\n")

    return stability_df


def component_evolution(graphs, variant='unweighted'):
    """
    Track connected components across years (Section 9).
    """
    years = sorted(graphs.keys())
    print("\n── Component Evolution ────────────────────────────────")
    print(f"{'Year':>6}  {'Nodes':>6}  {'Edges':>6}  {'Components':>11}  "
          f"{'Largest':>8}  {'Isolated':>9}")
    print("─" * 55)
    for year in years:
        G = graphs[year][variant]
        components   = list(nx.connected_components(G))
        n_comp       = len(components)
        largest_size = max(len(c) for c in components) if components else 0
        isolated     = sum(1 for c in components if len(c) == 1)
        print(f"{year:>6}  {G.number_of_nodes():>6}  {G.number_of_edges():>6}  "
              f"{n_comp:>11}  {largest_size:>8}  {isolated:>9}")
    print("──────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────
# CATEGORY ANALYSIS (Section 10)
# ─────────────────────────────────────────────────────────────

def category_analysis(graphs, variant='unweighted'):
    """
    For each year, compute:
      - Intra-category edges (both items same category)
      - Inter-category edges (items in different categories)
      - Category-level connectivity (edges between each pair of categories)

    Returns
    -------
    cat_df : DataFrame
    """
    years = sorted(graphs.keys())
    rows  = []

    for year in years:
        G = graphs[year][variant]

        intra = 0
        inter = 0
        cat_pairs = {}

        for u, v in G.edges():
            cat_u = G.nodes[u].get('category', 'Unknown')
            cat_v = G.nodes[v].get('category', 'Unknown')

            if cat_u == cat_v:
                intra += 1
            else:
                inter += 1
                pair = tuple(sorted([cat_u, cat_v]))
                cat_pairs[pair] = cat_pairs.get(pair, 0) + 1

        rows.append({
            'year':         year,
            'total_edges':  G.number_of_edges(),
            'intra_edges':  intra,
            'inter_edges':  inter,
            'pct_intra':    round(100 * intra / max(G.number_of_edges(), 1), 1)
        })

        top_cat_pairs = sorted(cat_pairs.items(), key=lambda x: -x[1])[:5]
        print(f"\nYear {year} — top 5 inter-category connections:")
        for pair, cnt in top_cat_pairs:
            print(f"  {pair[0]:30} <-> {pair[1]:30} : {cnt}")

    cat_df = pd.DataFrame(rows)
    print("\n── Category Summary ──────────────────────────────────")
    print(cat_df.to_string(index=False))
    print("──────────────────────────────────────────────────────\n")

    return cat_df


# ─────────────────────────────────────────────────────────────
# WEIGHT SENSITIVITY (Section 11)
# ─────────────────────────────────────────────────────────────

def compare_weighting_schemes(graphs, year):
    """
    Compare centrality rankings under the three edge weighting schemes
    for a given year. Shows how rankings shift when weights change.
    """
    variants = {
        'unweighted':     (False, 'unweighted'),
        'weighted_count': (True,  'weighted_count'),
        'weighted_avg':   (True,  'weighted_avg'),
    }

    top_dfs = {}
    for name, (weighted, variant_key) in variants.items():
        G   = graphs[year][variant_key]
        df  = compute_centrality(G, weighted=weighted)
        df  = df[['item', 'degree_centrality', 'betweenness_centrality']].head(10)
        df  = df.rename(columns={
            'degree_centrality':      f'deg_{name[:3]}',
            'betweenness_centrality': f'btw_{name[:3]}'
        })
        top_dfs[name] = df.set_index('item')

    combined = pd.concat(top_dfs.values(), axis=1).fillna('-')
    print(f"\n── Weight Sensitivity — Year {year} ──────────────────")
    print(combined.to_string())
    print("──────────────────────────────────────────────────────\n")
    return combined


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.graph_builder import load_graphs
    from src.preprocess   import load_metadata

    print("=== ANALYSIS ===\n")

    K   = 3
    TAU = 0.5

    metadata = load_metadata("data/metadata.json")
    graphs   = load_graphs(K=K, tau=TAU)

    # ── 1. Centrality ───────────────────────────────────────
    cent_all = compute_centrality_all_years(graphs, weighted=False,
                                            variant='unweighted')
    top_central_items(cent_all, top_n=10)
    save_centrality(cent_all, variant='unweighted')

    # ── 2. Temporal ─────────────────────────────────────────
    temporal_df, persistent_edges = temporal_edge_analysis(graphs)
    temporal_df.to_csv("data/temporal_analysis.csv", index=False)

    stability_df = temporal_centrality_stability(cent_all)
    stability_df.to_csv("data/centrality_stability.csv", index=False)

    component_evolution(graphs)

    # ── 3. Category ─────────────────────────────────────────
    cat_df = category_analysis(graphs)
    cat_df.to_csv("data/category_analysis.csv", index=False)

    # ── 4. Weight sensitivity ───────────────────────────────
    for year in metadata['years']:
        compare_weighting_schemes(graphs, year)

    print("=== DONE ===")