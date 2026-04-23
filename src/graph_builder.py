"""
graph_builder.py
----------------
Builds NetworkX graphs for each year using two strategies:

  1. Unweighted  : edge exists if N_y(i,j) >= K
  2. Weighted (count)   : weight = N_y(i,j)           if N_y(i,j) >= K
  3. Weighted (avg-sim) : weight = avg_sim_y(i,j)     if N_y(i,j) >= K

OUTPUT
------
    data/graphs_K{K}_tau{tau}.pkl   — { year: { 'unweighted', 'weighted_count',
                                                 'weighted_avg' } : nx.Graph }
"""

import networkx as nx
import pickle
import os
from src.preprocess import load_metadata
from src.similarity import load_similarities


# ─────────────────────────────────────────────────────────────
# BUILD SINGLE GRAPH
# ─────────────────────────────────────────────────────────────

def build_unweighted_graph(N_y, items, K=3):
    """
    Unweighted graph:  edge (i,j) exists iff N_y(i,j) >= K

    Parameters
    ----------
    N_y   : dict  { (item_i, item_j) : city_count }
    items : list of all item names (to pre-add as nodes)
    K     : int, minimum city-count threshold

    Returns
    -------
    G : nx.Graph
    """
    G = nx.Graph()
    G.add_nodes_from(items)           # ensure isolated items are still nodes

    for (item_i, item_j), count in N_y.items():
        if count >= K:
            G.add_edge(item_i, item_j)

    return G


def build_weighted_count_graph(N_y, items, K=3):
    """
    Weighted graph — weight = N_y(i,j) (number of agreeing cities).
    Higher weight means more cities show similar price movement.
    """
    G = nx.Graph()
    G.add_nodes_from(items)

    for (item_i, item_j), count in N_y.items():
        if count >= K:
            G.add_edge(item_i, item_j, weight=count)

    return G


def build_weighted_avg_graph(N_y, avg_sim, items, K=3):
    """
    Weighted graph — weight = avg cosine similarity across all cities.
    Uses N_y as the threshold gate (edge only if N_y(i,j) >= K),
    but the edge weight reflects overall co-movement strength.
    """
    G = nx.Graph()
    G.add_nodes_from(items)

    for (item_i, item_j), count in N_y.items():
        if count >= K:
            pair = (item_i, item_j) if item_i < item_j else (item_j, item_i)
            weight = avg_sim.get(pair, avg_sim.get((item_j, item_i), 0.0))
            G.add_edge(item_i, item_j, weight=weight)

    return G


# ─────────────────────────────────────────────────────────────
# BUILD ALL GRAPHS FOR ALL YEARS
# ─────────────────────────────────────────────────────────────

def build_all_graphs(N_all, avg_all, metadata, K=3):
    """
    Build all three graph variants for every year.

    Returns
    -------
    graphs : dict  { year : { 'unweighted'     : nx.Graph,
                               'weighted_count' : nx.Graph,
                               'weighted_avg'   : nx.Graph } }
    """
    items = metadata['items']
    years = metadata['years']
    graphs = {}

    for year in years:
        N_y     = N_all[year]
        avg_sim = avg_all[year]

        G_unw  = build_unweighted_graph(N_y, items, K=K)
        G_wc   = build_weighted_count_graph(N_y, items, K=K)
        G_wa   = build_weighted_avg_graph(N_y, avg_sim, items, K=K)

        graphs[year] = {
            'unweighted':     G_unw,
            'weighted_count': G_wc,
            'weighted_avg':   G_wa
        }

        _print_graph_stats(year, G_unw, G_wc, G_wa)

    return graphs


def _print_graph_stats(year, G_unw, G_wc, G_wa):
    print(f"\nYear {year}:")
    print(f"  Unweighted    — nodes: {G_unw.number_of_nodes():>4}, "
          f"edges: {G_unw.number_of_edges():>5}, "
          f"components: {nx.number_connected_components(G_unw)}")
    print(f"  Weighted(cnt) — nodes: {G_wc.number_of_nodes():>4}, "
          f"edges: {G_wc.number_of_edges():>5}")
    print(f"  Weighted(avg) — nodes: {G_wa.number_of_nodes():>4}, "
          f"edges: {G_wa.number_of_edges():>5}")


# ─────────────────────────────────────────────────────────────
# SENSITIVITY ANALYSIS — vary K and tau
# ─────────────────────────────────────────────────────────────

def sensitivity_analysis(N_all, avg_all, metadata,
                          tau_values=(0.3, 0.5, 0.7),
                          K_values=(2, 3, 5)):
    """
    Build graphs across a grid of (tau, K) values and report
    edge counts — fulfils Section 11 of the project spec.

    Note: tau affects N_all (which must be precomputed per tau).
          Here we only vary K on the existing N_all (fixed tau).
          To vary tau properly, rerun similarity.py with each tau.
    """
    items = metadata['items']
    years = metadata['years']

    print("\n── Sensitivity Analysis (varying K) ─────────────────")
    print(f"{'Year':>6} {'K':>4}  {'Nodes':>7}  {'Edges':>7}  {'Components':>12}")
    print("─" * 45)

    for year in years:
        N_y = N_all[year]
        for K in K_values:
            G = build_unweighted_graph(N_y, items, K=K)
            print(f"{year:>6} {K:>4}  {G.number_of_nodes():>7}  "
                  f"{G.number_of_edges():>7}  "
                  f"{nx.number_connected_components(G):>12}")
        print()


# ─────────────────────────────────────────────────────────────
# ADD CATEGORY ATTRIBUTES TO NODES
# ─────────────────────────────────────────────────────────────

def attach_categories(graphs, metadata):
    """
    Add a 'category' attribute to every node in every graph.
    Needed for visualisation colouring and category analysis (Section 10).
    """
    cat_map = metadata['category_map']
    for year, variants in graphs.items():
        for variant_name, G in variants.items():
            for node in G.nodes():
                G.nodes[node]['category'] = cat_map.get(node, 'Unknown')
    return graphs


# ─────────────────────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────────────────────

def save_graphs(graphs, K, tau, out_dir="data"):
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/graphs_K{K}_tau{str(tau).replace('.','')}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(graphs, f)
    print(f"Saved graphs → {path}")
    return path


def load_graphs(K, tau, data_dir="data"):
    path = f"{data_dir}/graphs_K{K}_tau{str(tau).replace('.','')}.pkl"
    with open(path, 'rb') as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== GRAPH BUILDING ===\n")

    K   = 3      # minimum number of agreeing cities
    TAU = 0.5    # must match the tau used in similarity.py

    metadata         = load_metadata("data/metadata.json")
    N_all, avg_all   = load_similarities(metadata['years'])

    graphs = build_all_graphs(N_all, avg_all, metadata, K=K)
    graphs = attach_categories(graphs, metadata)

    sensitivity_analysis(N_all, avg_all, metadata)

    save_graphs(graphs, K=K, tau=TAU)

    print("\n=== DONE ===")