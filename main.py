"""
main.py
-------
Master pipeline for the CPI Item Co-Movement Network project.

USAGE
-----
# Full pipeline
python main.py

# Skip scraping (use existing data/cpi_data.csv)
python main.py --skip-scrape

# Change parameters
python main.py --tau 0.6 --K 4

# Run only a specific stage
python main.py --stage preprocess
python main.py --stage similarity
python main.py --stage graphs
python main.py --stage analysis
python main.py --stage visualize

# (Optional) Inspect single PDF
python main.py --inspect data/pdfs/2022_01.pdf
"""

import os
import sys
import argparse
import time


# ─────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CPI Item Co-Movement Network Pipeline"
    )

    p.add_argument('--inspect', type=str, default=None,
                   help='Inspect a single PDF file and exit.')

    p.add_argument('--skip-scrape', action='store_true',
                   help='Skip scraping step (use existing data/cpi_data.csv).')

    p.add_argument('--tau', type=float, default=0.5,
                   help='Cosine similarity threshold τ.')

    p.add_argument('--K', type=int, default=3,
                   help='City-count threshold K.')

    p.add_argument('--data-dir', type=str, default='data/pdfs',
                   help='Folder containing PDF files.')

    p.add_argument('--stage', type=str, default=None,
                   choices=['scrape', 'preprocess', 'similarity',
                            'graphs', 'analysis', 'visualize'],
                   help='Run only one pipeline stage.')

    return p.parse_args()


# ─────────────────────────────────────────────
# STAGE RUNNERS
# ─────────────────────────────────────────────

def run_scrape(data_dir):
    from src.scraper import load_all_months

    print_header("STAGE 1 — PDF LOADING & SCRAPING")

    df = load_all_months(data_dir)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/cpi_data.csv", index=False)

    print(f"\nSaved → data/cpi_data.csv ({len(df):,} rows)\n")
    return df


def run_preprocess():
    from src.preprocess import (
        load_cpi_data, filter_complete_series,
        compute_price_change_vectors, remove_zero_vectors,
        build_metadata, vector_summary,
        save_vectors, save_metadata
    )

    print_header("STAGE 2 — PREPROCESSING")

    df = load_cpi_data("data/cpi_data.csv")
    df = filter_complete_series(df)

    vectors = compute_price_change_vectors(df)
    vectors = remove_zero_vectors(vectors)

    meta = build_metadata(df, vectors)

    vector_summary(vectors, meta)

    save_vectors(vectors)
    save_metadata(meta)

    return vectors, meta


def run_similarity(tau):
    from src.preprocess import load_vectors, load_metadata
    from src.similarity import compute_all_years, save_similarities

    print_header(f"STAGE 3 — SIMILARITY (τ={tau})")

    vectors = load_vectors()
    metadata = load_metadata()

    N_all, avg_all = compute_all_years(vectors, metadata, tau=tau)

    save_similarities(N_all, avg_all)

    return N_all, avg_all


def run_graphs(K, tau):
    from src.preprocess import load_metadata
    from src.similarity import load_similarities
    from src.graph_builder import (
        build_all_graphs, attach_categories,
        sensitivity_analysis, save_graphs
    )

    print_header(f"STAGE 4 — GRAPH BUILDING (K={K})")

    metadata = load_metadata()

    N_all, avg_all = load_similarities(metadata['years'])

    graphs = build_all_graphs(N_all, avg_all, metadata, K=K)
    graphs = attach_categories(graphs, metadata)

    sensitivity_analysis(N_all, avg_all, metadata)

    save_graphs(graphs, K=K, tau=tau)

    return graphs


def run_analysis(K, tau):
    from src.preprocess import load_metadata
    from src.graph_builder import load_graphs
    from src.analysis import (
        compute_centrality_all_years,
        top_central_items,
        save_centrality,
        temporal_edge_analysis,
        temporal_centrality_stability,
        component_evolution,
        category_analysis,
        compare_weighting_schemes
    )

    print_header("STAGE 5 — ANALYSIS")

    metadata = load_metadata()
    graphs = load_graphs(K=K, tau=tau)
    years = metadata['years']

    cent_unweighted = compute_centrality_all_years(
        graphs, weighted=False, variant='unweighted'
    )

    save_centrality(cent_unweighted, variant='unweighted')

    top_central_items(cent_unweighted)

    temporal_df, _ = temporal_edge_analysis(graphs)
    temporal_df.to_csv("data/temporal_analysis.csv", index=False)

    stab = temporal_centrality_stability(cent_unweighted)
    stab.to_csv("data/centrality_stability.csv", index=False)

    component_evolution(graphs)

    cat_df = category_analysis(graphs)
    cat_df.to_csv("data/category_analysis.csv", index=False)

    for year in years:
        compare_weighting_schemes(graphs, year)

    return cent_unweighted


def run_visualize(K, tau):
    from src.preprocess import load_metadata
    from src.graph_builder import load_graphs
    from src.analysis import (
        compute_centrality_all_years,
        temporal_edge_analysis,
        temporal_centrality_stability
    )
    from src.visualize import (
        plot_all_year_networks,
        plot_weighting_comparison,
        plot_all_centrality_charts,
        plot_temporal_stability,
        plot_centrality_rank_change,
        plot_category_heatmap
    )

    print_header("STAGE 6 — VISUALIZATION")

    metadata = load_metadata()
    graphs = load_graphs(K=K, tau=tau)
    years = metadata['years']

    plot_all_year_networks(graphs)

    for year in years:
        plot_weighting_comparison(graphs, year)
        plot_category_heatmap(graphs, year)

    cent = compute_centrality_all_years(graphs, weighted=False)

    plot_all_centrality_charts(cent)

    temporal_df, _ = temporal_edge_analysis(graphs)

    plot_temporal_stability(temporal_df)

    stab = temporal_centrality_stability(cent)

    plot_centrality_rank_change(stab, years)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def print_header(title):
    print("\n" + "═" * 60)
    print(title)
    print("═" * 60)


def elapsed(t0):
    t = time.time() - t0
    return f"{int(t//60)}m {t%60:.1f}s"


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # OPTIONAL: inspect single PDF
    if args.inspect:
        from src.scraper import extract_annexure_from_pdf
        print_header("INSPECT MODE")

        df = extract_annexure_from_pdf(args.inspect)
        print(df.head())
        return

    t0 = time.time()

    if args.stage:
        {
            'scrape': lambda: run_scrape(args.data_dir),
            'preprocess': run_preprocess,
            'similarity': lambda: run_similarity(args.tau),
            'graphs': lambda: run_graphs(args.K, args.tau),
            'analysis': lambda: run_analysis(args.K, args.tau),
            'visualize': lambda: run_visualize(args.K, args.tau),
        }[args.stage]()

        print(f"\nStage '{args.stage}' done in {elapsed(t0)}")
        return

    print_header("FULL PIPELINE STARTED")

    if not args.skip_scrape:
        run_scrape(args.data_dir)
    else:
        print("Skipping scrape... using existing CSV")

    run_preprocess()
    run_similarity(args.tau)
    run_graphs(args.K, args.tau)
    run_analysis(args.K, args.tau)
    run_visualize(args.K, args.tau)

    print_header(f"DONE in {elapsed(t0)}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()