"""
Main Pipeline — Cultural Menu Optimizer
"""

import argparse
import json
import os
import sys
import time
import subprocess
import pandas as pd

#Fix paths so all imports work regardless of where you run from
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

#Step 0: Check dependencies
def check_dependencies():
    missing = []
    for pkg in ["sklearn", "seaborn", "matplotlib"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg.replace("sklearn", "scikit-learn"))
    if missing:
        print(f"Missing packages: {missing}")
        print("   Run: pip install scikit-learn seaborn matplotlib pandas")
        sys.exit(1)
    try:
        import xgboost
    except ImportError:
        pass

check_dependencies()

# Step 1: Generate data files
print("Generating data files...")
subprocess.run([sys.executable, os.path.join(ROOT, "data", "generate_data.py")], check=True)

#Now safe to import project modules
from models.train import MenuRecommender, evaluate_model
from utils.visualization import (
    plot_feature_importance,
    plot_country_clusters_pca,
    plot_dietary_heatmap,
    plot_recommendation_results,
    plot_evaluation_results,
    plot_neighbor_similarity,
)


def run_pipeline(target_countries_filter=None, top_n=14, make_plots=True):
    t0 = time.time()
    print("\n" + "=" * 60)
    print("  Cultural Menu Optimizer - Full Pipeline")
    print("=" * 60 + "\n")

    print("Loading data...")
    df_train  = pd.read_csv("data/training_countries.csv", index_col="country")
    df_target = pd.read_csv("data/target_countries.csv",   index_col="country")

    with open("data/menu_items.json")    as f: menu_items    = json.load(f)
    with open("data/country_menus.json") as f: country_menus = json.load(f)

    if target_countries_filter:
        valid = [c for c in target_countries_filter if c in df_target.index]
        if not valid:
            print(f"None of {target_countries_filter} found in target countries")
            sys.exit(1)
        df_target = df_target.loc[valid]

    print(f"   Training countries : {len(df_train)}")
    print(f"   Target countries   : {len(df_target)}")
    print(f"   Menu items         : {len(menu_items)}")

    print("\nTraining models...")
    recommender = MenuRecommender()
    recommender.fit(df_train, menu_items, country_menus, verbose=True)
    recommender.save("models/recommender.pkl")

    print("\nEvaluating on holdout countries...")
    holdout = ["India", "Japan", "Saudi Arabia", "Malaysia", "Turkey"]
    eval_results = evaluate_model(recommender, holdout)

    print(f"\n   {'Country':<15} {'Jaccard':>8} {'Precision':>10} {'Recall':>8}")
    print("   " + "-" * 45)
    for country, metrics in eval_results.items():
        print(f"   {country:<15} {metrics['jaccard']:>8.3f} "
              f"{metrics['precision']:>10.3f} {metrics['recall']:>8.3f}")

    avg_jaccard = sum(m["jaccard"] for m in eval_results.values()) / len(eval_results)
    print(f"\n   Avg Jaccard Similarity: {avg_jaccard:.3f}")

    print(f"\nGenerating recommendations for {len(df_target)} target countries...")
    recommendations = recommender.recommend(df_target, top_n=top_n)

    for country, recs in recommendations.items():
        print(f"\n  -- {country} --")
        print(f"     Cluster  : {recs['cluster_name']} (conf: {recs['cluster_confidence']:.2f})")
        c = recs['constraints']
        print(f"     Beef taboo: {c['beef_taboo']:.2f} | Pork taboo: {c['pork_taboo']:.2f} | Halal: {c['halal_required']}")
        print(f"     Top 5 items:")
        for item in recs["recommended_menu"][:5]:
            veg_tag = " (veg)" if item["vegetarian"] else ""
            print(f"       {item['confidence_pct']:5.1f}%  {item['name']}{veg_tag}")

    if make_plots:
        print("\nGenerating visualizations...")
        os.makedirs("outputs", exist_ok=True)
        plot_feature_importance(recommender.classifier.feature_importance())
        plot_country_clusters_pca(df_train)
        plot_dietary_heatmap(df_train)
        plot_evaluation_results(eval_results)
        for country in list(df_target.index)[:6]:
            plot_recommendation_results(recommendations, country)
            plot_neighbor_similarity(recommendations, country)

    print("\nExporting results...")
    os.makedirs("outputs", exist_ok=True)

    with open("outputs/recommendations.json", "w") as f:
        json.dump(recommendations, f, indent=2, default=str)
    print("   Saved: outputs/recommendations.json")

    rows = []
    for country, recs in recommendations.items():
        for rank, item in enumerate(recs["recommended_menu"], 1):
            rows.append({
                "country": country, "rank": rank,
                "item_id": item["id"], "item_name": item["name"],
                "confidence_pct": item["confidence_pct"],
                "protein": item["protein"], "vegetarian": item["vegetarian"],
                "spice_level": item["spice"], "cluster": recs["cluster_name"],
                "halal_required": recs["constraints"]["halal_required"],
            })
    pd.DataFrame(rows).to_csv("outputs/recommendations.csv", index=False)
    print("   Saved: outputs/recommendations.csv")

    eval_df = pd.DataFrame(eval_results).T
    eval_df.index.name = "country"
    eval_df[["jaccard", "precision", "recall"]].to_csv("outputs/evaluation_results.csv")
    print("   Saved: outputs/evaluation_results.csv")

    elapsed = time.time() - t0
    print(f"\nPipeline complete in {elapsed:.1f}s — all outputs in ./outputs/")
    print("=" * 60 + "\n")

    return recommendations, eval_results, recommender


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cultural Menu Optimizer Pipeline")
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--top-n", type=int, default=14)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        target_countries_filter=args.countries,
        top_n=args.top_n,
        make_plots=not args.no_plots
    )