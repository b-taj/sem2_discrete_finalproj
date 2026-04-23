"""
visualize.py
------------
All visualisations for the project:
  - Item co-movement network graphs (per year, per weighting scheme)
  - Centrality bar charts
  - Threshold sensitivity plots
  - Temporal edge stability chart
  - Category heatmap

All figures are saved to report/figures/
"""

import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')              # non-interactive backend (safe for all OSes)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

FIGURES_DIR = "report/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# COLOUR PALETTE FOR CATEGORIES
# ─────────────────────────────────────────────────────────────

CATEGORY_COLORS = {
    'Food & Non-Alcoholic Beverages':    '#e74c3c',
    'Alcoholic Beverages & Tobacco':     '#8e44ad',
    'Clothing & Footwear':               '#3498db',
    'Housing, Water, Electricity, Gas':  '#f39c12',
    'Furnishing & Household Equipment':  '#27ae60',
    'Health':                            '#1abc9c',
    'Transport':                         '#e67e22',
    'Communication':                     '#2980b9',
    'Recreation & Culture':              '#d35400',
    'Education':                         '#16a085',
    'Restaurants & Hotels':              '#c0392b',
    'Miscellaneous':                     '#7f8c8d',
    'Unknown':                           '#bdc3c7',
}

def _node_colors(G):
    """Return a list of colours matching G.nodes() order."""
    default = '#bdc3c7'
    return [
        CATEGORY_COLORS.get(G.nodes[n].get('category', 'Unknown'), default)
        for n in G.nodes()
    ]

def _category_legend(ax, categories_present):
    """Add a compact legend mapping category → colour."""
    patches = [
        mpatches.Patch(color=CATEGORY_COLORS.get(c, '#bdc3c7'), label=c)
        for c in sorted(categories_present)
        if c in CATEGORY_COLORS
    ]
    ax.legend(handles=patches, fontsize=6, loc='upper left',
              bbox_to_anchor=(1.01, 1), borderaxespad=0, title='Category')


# ─────────────────────────────────────────────────────────────
# NETWORK GRAPH
# ─────────────────────────────────────────────────────────────

def plot_network(G, title="Item Co-Movement Network",
                 filename="network.png", layout='spring',
                 centrality_size=True):
    """
    Draw the item–item network with:
      - Node colour  → category
      - Node size    → degree centrality (if centrality_size=True)
      - Edge opacity → proportional to weight (if edges have weight)

    Parameters
    ----------
    layout : 'spring' | 'kamada_kawai' | 'circular'
    """
    fig, ax = plt.subplots(figsize=(16, 12))

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, 'Empty graph', ha='center', va='center')
        plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150, bbox_inches='tight')
        plt.close()
        return

    # Layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42, k=0.8)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.circular_layout(G)

    # Node sizes
    deg_cent = nx.degree_centrality(G)
    if centrality_size:
        node_sizes = [max(50, 4000 * deg_cent[n]) for n in G.nodes()]
    else:
        node_sizes = [200] * G.number_of_nodes()

    # Node colours
    node_colors = _node_colors(G)

    # Edge weights for alpha
    if nx.get_edge_attributes(G, 'weight'):
        weights  = [G[u][v].get('weight', 1) for u, v in G.edges()]
        max_w    = max(weights) if weights else 1
        alphas   = [0.2 + 0.6 * (w / max_w) for w in weights]
        edge_colors = [(0.5, 0.5, 0.5, a) for a in alphas]
    else:
        edge_colors = [(0.6, 0.6, 0.6, 0.4)] * G.number_of_edges()

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                           width=0.8, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color=node_colors, alpha=0.9, ax=ax)

    # Labels only for top-20 nodes by degree
    top_nodes = sorted(deg_cent, key=deg_cent.get, reverse=True)[:20]
    label_dict = {n: n for n in top_nodes}
    nx.draw_networkx_labels(G, pos, labels=label_dict,
                            font_size=6, font_weight='bold', ax=ax)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')

    # Category legend
    categories_present = {G.nodes[n].get('category', 'Unknown') for n in G.nodes()}
    _category_legend(ax, categories_present)

    plt.tight_layout()
    out = f"{FIGURES_DIR}/{filename}"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")


def plot_all_year_networks(graphs, variant='unweighted'):
    """Plot one network per year for a given variant."""
    for year, variants in sorted(graphs.items()):
        G = variants[variant]
        plot_network(
            G,
            title=f"Item Co-Movement Network — {year} ({variant})",
            filename=f"network_{year}_{variant}.png"
        )


# ─────────────────────────────────────────────────────────────
# SIDE-BY-SIDE WEIGHTING SCHEME COMPARISON
# ─────────────────────────────────────────────────────────────

def plot_weighting_comparison(graphs, year):
    """
    Draw unweighted, weighted_count, and weighted_avg graphs side by side
    for the same year (required by Section 13 of the demo video spec).
    """
    variant_labels = {
        'unweighted':     'Unweighted',
        'weighted_count': 'Weighted (city count)',
        'weighted_avg':   'Weighted (avg cosine)'
    }

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    fig.suptitle(f"Weighting Scheme Comparison — Year {year}",
                 fontsize=16, fontweight='bold')

    for ax, (variant, label) in zip(axes, variant_labels.items()):
        G   = graphs[year][variant]
        pos = nx.spring_layout(G, seed=42, k=0.8)

        deg_cent   = nx.degree_centrality(G)
        node_sizes = [max(30, 3000 * deg_cent[n]) for n in G.nodes()]
        node_colors = _node_colors(G)

        nx.draw_networkx(
            G, pos, ax=ax,
            node_size=node_sizes,
            node_color=node_colors,
            with_labels=False,
            edge_color='#aaaaaa',
            width=0.6, alpha=0.85
        )
        ax.set_title(f"{label}\n"
                     f"Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}",
                     fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    out = f"{FIGURES_DIR}/weighting_comparison_{year}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────
# CENTRALITY BAR CHART
# ─────────────────────────────────────────────────────────────

def plot_centrality_barchart(cent_df, year, top_n=20,
                              centrality_col='degree_centrality',
                              variant='unweighted'):
    """
    Horizontal bar chart of top-N items by a centrality measure.
    Bars coloured by category.
    """
    df = cent_df.head(top_n).copy()

    colors = [
        CATEGORY_COLORS.get(cat, '#bdc3c7')
        for cat in df['category']
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(df['item'][::-1], df[centrality_col][::-1],
                   color=colors[::-1], edgecolor='white', linewidth=0.5)

    ax.set_xlabel(centrality_col.replace('_', ' ').title(), fontsize=11)
    ax.set_title(f"Top {top_n} Items — {centrality_col.replace('_',' ').title()}\n"
                 f"Year {year}, {variant}", fontsize=12, fontweight='bold')
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    # Value labels
    for bar, val in zip(bars, df[centrality_col][::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=7)

    plt.tight_layout()
    out = f"{FIGURES_DIR}/centrality_{centrality_col}_{year}_{variant}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")


def plot_all_centrality_charts(cent_all, variant='unweighted'):
    for year, df in sorted(cent_all.items()):
        for col in ('degree_centrality', 'closeness_centrality',
                    'betweenness_centrality'):
            plot_centrality_barchart(df, year, centrality_col=col,
                                     variant=variant)


# ─────────────────────────────────────────────────────────────
# THRESHOLD SENSITIVITY PLOT
# ─────────────────────────────────────────────────────────────

def plot_threshold_sensitivity(N_all_per_tau, tau_values, K_values, year,
                                items):
    """
    Line chart showing how edge count changes as K varies
    for different tau values.

    Parameters
    ----------
    N_all_per_tau : dict  { tau : N_y_dict }
    tau_values    : list of tau floats used
    K_values      : list of K ints to sweep
    year          : int
    items         : list of item names
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for tau in tau_values:
        N_y = N_all_per_tau[tau]
        edge_counts = []
        for K in K_values:
            count = sum(1 for v in N_y.values() if v >= K)
            edge_counts.append(count)
        ax.plot(K_values, edge_counts, marker='o', label=f'τ = {tau}')

    ax.set_xlabel('City-count threshold K', fontsize=11)
    ax.set_ylabel('Number of edges in graph', fontsize=11)
    ax.set_title(f'Threshold Sensitivity — Year {year}', fontsize=13,
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(linestyle='--', alpha=0.4)

    out = f"{FIGURES_DIR}/threshold_sensitivity_{year}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────
# TEMPORAL EDGE STABILITY CHART
# ─────────────────────────────────────────────────────────────

def plot_temporal_stability(temporal_df):
    """
    Stacked bar chart showing appeared / disappeared / stable edges
    between consecutive years.
    """
    df = temporal_df[temporal_df['comparison'] != 'ALL YEARS'].copy()
    if df.empty:
        print("No temporal data to plot.")
        return

    comparisons = df['comparison'].tolist()
    appeared    = df['appeared'].astype(int).tolist()
    disappeared = df['disappeared'].astype(int).tolist()
    stable      = df['stable'].astype(int).tolist()

    x = np.arange(len(comparisons))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width, stable,      width, label='Stable',      color='#27ae60')
    ax.bar(x,         appeared,    width, label='Appeared',     color='#3498db')
    ax.bar(x + width, disappeared, width, label='Disappeared',  color='#e74c3c')

    ax.set_xticks(x)
    ax.set_xticklabels(comparisons, fontsize=11)
    ax.set_ylabel('Number of edges', fontsize=11)
    ax.set_title('Temporal Edge Stability Across Years', fontsize=13,
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    out = f"{FIGURES_DIR}/temporal_edge_stability.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────
# CATEGORY HEATMAP
# ─────────────────────────────────────────────────────────────

def plot_category_heatmap(graphs, year, variant='unweighted'):
    """
    Heatmap of edge density between every pair of categories.
    """
    G = graphs[year][variant]
    categories = sorted({
        G.nodes[n].get('category', 'Unknown') for n in G.nodes()
    })

    # Build category × category edge-count matrix
    mat = pd.DataFrame(0, index=categories, columns=categories)
    for u, v in G.edges():
        cu = G.nodes[u].get('category', 'Unknown')
        cv = G.nodes[v].get('category', 'Unknown')
        mat.loc[cu, cv] += 1
        if cu != cv:
            mat.loc[cv, cu] += 1

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(mat.values, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, ax=ax, label='Edge count')

    ax.set_xticks(range(len(categories)))
    ax.set_yticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(categories, fontsize=8)
    ax.set_title(f'Category-Level Edge Density — Year {year}',
                 fontsize=13, fontweight='bold')

    # Annotate cells
    for i in range(len(categories)):
        for j in range(len(categories)):
            val = mat.values[i, j]
            if val > 0:
                ax.text(j, i, str(val), ha='center', va='center',
                        fontsize=7, color='black')

    plt.tight_layout()
    out = f"{FIGURES_DIR}/category_heatmap_{year}_{variant}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────
# CENTRALITY RANK CHANGE (across years)
# ─────────────────────────────────────────────────────────────

def plot_centrality_rank_change(stability_df, years):
    """
    Bump chart showing how the rank of top items changes year by year.
    """
    rank_cols = [f'rank_{y}' for y in sorted(years)]
    df = stability_df.dropna(subset=rank_cols).copy()
    df = df[df[rank_cols[0]] <= 15]           # only items that started in top-15

    fig, ax = plt.subplots(figsize=(10, 7))
    x_pos = list(range(len(years)))

    for _, row in df.iterrows():
        ranks = [row[c] for c in rank_cols]
        ax.plot(x_pos, ranks, marker='o', linewidth=1.5, markersize=6)
        ax.text(x_pos[-1] + 0.1, ranks[-1], row['item'],
                va='center', fontsize=7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(y) for y in sorted(years)], fontsize=11)
    ax.invert_yaxis()
    ax.set_ylabel('Rank (lower = more central)', fontsize=11)
    ax.set_title('Centrality Rank Changes Across Years', fontsize=13,
                 fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    out = f"{FIGURES_DIR}/centrality_rank_change.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.graph_builder import load_graphs
    from src.preprocess    import load_metadata
    from src.analysis      import (compute_centrality_all_years,
                                   temporal_edge_analysis,
                                   temporal_centrality_stability)

    print("=== VISUALISATION ===\n")

    K   = 3
    TAU = 0.5

    metadata = load_metadata("data/metadata.json")
    graphs   = load_graphs(K=K, tau=TAU)
    years    = metadata['years']

    # ── Networks ────────────────────────────────────────────
    plot_all_year_networks(graphs, variant='unweighted')
    for year in years:
        plot_weighting_comparison(graphs, year)

    # ── Centrality ──────────────────────────────────────────
    cent_all = compute_centrality_all_years(
        graphs, weighted=False, variant='unweighted'
    )
    plot_all_centrality_charts(cent_all, variant='unweighted')

    # ── Temporal ────────────────────────────────────────────
    temporal_df, _ = temporal_edge_analysis(graphs)
    plot_temporal_stability(temporal_df)

    stability_df = temporal_centrality_stability(cent_all)
    plot_centrality_rank_change(stability_df, years)

    # ── Category heatmaps ───────────────────────────────────
    for year in years:
        plot_category_heatmap(graphs, year)

    print(f"\nAll figures saved to '{FIGURES_DIR}/'")
    print("=== DONE ===")