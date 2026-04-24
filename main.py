"""
main.py
-------
Master pipeline for the PBS CPI Item Co-Movement Network project.

USAGE
-----
    # Step 1 only (inspect a single file):
    python main.py --inspect data/annexures/2022_01.xlsx

    # Full pipeline (all steps):
    python main.py

    # Skip scraping if cpi_data.csv already exists:
    python main.py --skip-scrape

    # Change parameters:
    python main.py --tau 0.6 --K 4

    # Run only a specific stage:
    python main.py --stage preprocess
    python main.py --stage similarity
    python main.py --stage graphs
    python main.py --stage analysis
    python main.py --stage visualize
"""

import os
import sys
import argparse
import time


# ─────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="PBS CPI Item Co-Movement Network Pipeline"
    )
    p.add_argument('--inspect',      type=str,   default=None,
                   help='Path to a single Excel file to inspect and exit.')
    p.add_argument('--skip-scrape',  action='store_true',
                   help='Skip scraping step (use existing data/cpi_data.csv).')
    p.add_argument('--tau',          type=float, default=0.5,
                   help='Cosine similarity threshold τ (default: 0.5).')
    p.add_argument('--K',            type=int,   default=3,
                   help='City-count threshold K (default: 3).')
    p.add_argument('--data-dir',     type=str,   default='data/annexures',
                   help='Folder containing YYYY_MM.xlsx files.')
    p.add_argument('--stage',        type=str,   default=None,
                   choices=['scrape','preprocess','similarity',
                             'graphs','analysis','visualize'],
                   help='Run only a specific pipeline stage.')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# STAGE RUNNERS
# ─────────────────────────────────────────────────────────────

def run_scrape(data_dir):
    from src.scraper import load_all_months, normalize_item_names
    print_header("STAGE 1 — DATA LOADING")

    df = load_all_months(data_dir)
    df = normalize_item_names(df)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/cpi_data.csv", index=False)
    print(f"\nSaved → data/cpi_data.csv  ({len(df):,} rows)\n")
    return df


def run_preprocess():
    from src.preprocess import (load_cpi_data, filter_complete_series,
                                compute_price_change_vectors, remove_zero_vectors,
                                build_metadata, vector_summary,
                                save_vectors, save_metadata)
    print_header("STAGE 2 — PREPROCESSING")

    df      = load_cpi_data("data/cpi_data.csv")
    df      = filter_complete_series(df)
    vectors = compute_price_change_vectors(df)
    vectors = remove_zero_vectors(vectors)
    meta    = build_metadata(df, vectors)

    vector_summary(vectors, meta)
    save_vectors(vectors)
    save_metadata(meta)
    return vectors, meta


def run_similarity(tau):
    from src.preprocess import load_vectors, load_metadata
    from src.similarity import compute_all_years, save_similarities
    print_header(f"STAGE 3 — SIMILARITY  (τ = {tau})")

    vectors  = load_vectors()
    metadata = load_metadata()
    N_all, avg_all = compute_all_years(vectors, metadata, tau=tau)
    save_similarities(N_all, avg_all)
    return N_all, avg_all


def run_graphs(K, tau):
    from src.preprocess    import load_metadata
    from src.similarity    import load_similarities
    from src.graph_builder import (build_all_graphs, attach_categories,
                                   sensitivity_analysis, save_graphs)
    print_header(f"STAGE 4 — GRAPH BUILDING  (K = {K})")

    metadata         = load_metadata()
    N_all, avg_all   = load_similarities(load_metadata()['years'])
    graphs           = build_all_graphs(N_all, avg_all, metadata, K=K)
    graphs           = attach_categories(graphs, metadata)

    print("\n── Sensitivity Analysis ──")
    sensitivity_analysis(N_all, avg_all, metadata)

    save_graphs(graphs, K=K, tau=tau)
    return graphs


def run_analysis(K, tau):
    from src.preprocess import load_metadata
    from src.graph_builder import load_graphs
    from src.analysis import (
        compute_centrality_all_years, top_central_items, save_centrality,
        temporal_edge_analysis, temporal_centrality_stability,
        component_evolution, category_analysis, compare_weighting_schemes
    )
    print_header("STAGE 5 — ANALYSIS")

    metadata = load_metadata()
    graphs   = load_graphs(K=K, tau=tau)
    years    = metadata['years']

    # Centrality (all three variants)
    for variant, weighted in [('unweighted', False),
                               ('weighted_count', True),
                               ('weighted_avg', True)]:
        print(f"\n  → Centrality for variant: {variant}")
        cent = compute_centrality_all_years(graphs, weighted=weighted,
                                            variant=variant)
        save_centrality(cent, variant=variant)

    # Use unweighted for temporal / category analysis
    cent_unw = compute_centrality_all_years(graphs, weighted=False,
                                             variant='unweighted')
    top_central_items(cent_unw)

    temporal_df, _ = temporal_edge_analysis(graphs)
    temporal_df.to_csv("data/temporal_analysis.csv", index=False)

    stab = temporal_centrality_stability(cent_unw)
    stab.to_csv("data/centrality_stability.csv", index=False)

    component_evolution(graphs)

    cat_df = category_analysis(graphs)
    cat_df.to_csv("data/category_analysis.csv", index=False)

    for year in years:
        compare_weighting_schemes(graphs, year)

    print("\nAll analysis results saved to data/\n")
    return cent_unw


def run_visualize(K, tau):
    from src.preprocess import load_metadata
    from src.graph_builder import load_graphs
    from src.analysis import (compute_centrality_all_years,
                               temporal_edge_analysis,
                               temporal_centrality_stability)
    from src.visualize import (
        plot_all_year_networks, plot_weighting_comparison,
        plot_all_centrality_charts, plot_temporal_stability,
        plot_centrality_rank_change, plot_category_heatmap
    )
    print_header("STAGE 6 — VISUALISATION")

    metadata = load_metadata()
    graphs   = load_graphs(K=K, tau=tau)
    years    = metadata['years']

    plot_all_year_networks(graphs, variant='unweighted')

    for year in years:
        plot_weighting_comparison(graphs, year)
        plot_category_heatmap(graphs, year)

    cent_all = compute_centrality_all_years(graphs, weighted=False,
                                             variant='unweighted')
    plot_all_centrality_charts(cent_all)

    temporal_df, _ = temporal_edge_analysis(graphs)
    plot_temporal_stability(temporal_df)

    stab = temporal_centrality_stability(cent_all)
    plot_centrality_rank_change(stab, years)

    print("\nAll figures saved to report/figures/\n")


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def print_header(title):
    w = 60
    print("\n" + "═" * w)
    print(f"  {title}")
    print("═" * w)


def elapsed(start):
    s = time.time() - start
    return f"{int(s // 60)}m {s % 60:.1f}s"


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Inspect mode ────────────────────────────────────────
    if args.inspect:
        from src.scraper import inspect_file
        inspect_file(args.inspect)
        return

    tau = args.tau
    K   = args.K

    t0 = time.time()

    # ── Single-stage mode ───────────────────────────────────
    if args.stage:
        {
            'scrape':      lambda: run_scrape(args.data_dir),
            'preprocess':  lambda: run_preprocess(),
            'similarity':  lambda: run_similarity(tau),
            'graphs':      lambda: run_graphs(K, tau),
            'analysis':    lambda: run_analysis(K, tau),
            'visualize':   lambda: run_visualize(K, tau),
        }[args.stage]()
        print(f"\nStage '{args.stage}' completed in {elapsed(t0)}")
        return

    # ── Full pipeline ───────────────────────────────────────
    print_header("PBS CPI Item Co-Movement Network — Full Pipeline")
    print(f"  Parameters:  τ = {tau}   K = {K}")
    print(f"  Data dir  :  {args.data_dir}")

    if not args.skip_scrape:
        run_scrape(args.data_dir)
    else:
        print("\n[Skipping scrape — using existing data/cpi_data.csv]")
        if not os.path.exists("data/cpi_data.csv"):
            print("ERROR: data/cpi_data.csv not found. "
                  "Remove --skip-scrape or run the scrape stage first.")
            sys.exit(1)

    run_preprocess()
    run_similarity(tau)
    run_graphs(K, tau)
    run_analysis(K, tau)
    run_visualize(K, tau)

    print_header(f"PIPELINE COMPLETE — total time: {elapsed(t0)}")
    print("""
  Output summary:
    data/cpi_data.csv                  — cleaned long-format CPI data
    data/price_change_vectors.pkl      — Δp vectors per (item, city, year)
    data/metadata.json                 — items, cities, years, categories
    data/similarity_counts_yYEAR.pkl   — N_y per year
    data/similarity_avg_yYEAR.pkl      — avg similarity per year
    data/graphs_K{K}_tau{tau}.pkl      — NetworkX graphs per year
    data/centrality_*.csv              — centrality tables
    data/temporal_analysis.csv         — edge stability
    data/category_analysis.csv         — intra/inter category edges
    report/figures/                    — all PNG visualisations
""")


if __name__ == "__main__":
    # Add project root to path so 'src.' imports work
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()