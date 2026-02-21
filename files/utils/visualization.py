"""
Evaluation, Visualization, and EDA for Cultural Menu Optimizer
Generates plots for:
- Feature importance
- Country clusters (PCA/UMAP)
- Similarity heatmaps
- Recommendation confidence charts
- Holdout evaluation results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import os
import json

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_engineering import get_feature_matrix, compute_dietary_constraints

# ── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

CLUSTER_COLORS = {
    0: "#E63946",  # Beef - red
    1: "#F4A261",  # Chicken - orange
    2: "#2A9D8F",  # Vegetarian - teal
    3: "#457B9D",  # Seafood - blue
    4: "#8338EC",  # Balanced - purple
}

CLUSTER_NAMES = {
    0: "Beef Dominant",
    1: "Chicken Dominant",
    2: "Vegetarian Heavy",
    3: "Seafood Adapted",
    4: "Balanced Western"
}

os.makedirs("outputs", exist_ok=True)


def plot_feature_importance(feature_importance_df: pd.DataFrame, top_n: int = 15, save: bool = True):
    """Bar chart of top feature importances."""
    df = feature_importance_df.head(top_n).sort_values("importance")

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(df["feature"], df["importance"],
                   color=plt.cm.viridis(np.linspace(0.2, 0.8, len(df))),
                   edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f"Top {top_n} Most Predictive Features\nfor Menu Localization Cluster")
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    for bar, val in zip(bars, df["importance"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color="#333")

    plt.tight_layout()
    if save:
        plt.savefig("outputs/feature_importance.png", bbox_inches="tight")
        print("   Saved: outputs/feature_importance.png")
    plt.close()


def plot_country_clusters_pca(df_train: pd.DataFrame, save: bool = True):
    """PCA scatter plot of all training countries colored by cluster."""
    X, cols = get_feature_matrix(df_train)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X.values)

    fig, ax = plt.subplots(figsize=(12, 8))
    clusters = df_train["menu_cluster"].values

    for cluster_id, color in CLUSTER_COLORS.items():
        mask = clusters == cluster_id
        countries = df_train.index[mask]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, s=120, label=CLUSTER_NAMES[cluster_id],
                   edgecolors="white", linewidth=1.5, zorder=5, alpha=0.9)
        for i, country in enumerate(countries):
            idx = np.where(mask)[0][i]
            ax.annotate(country, (coords[idx, 0], coords[idx, 1]),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=7.5, color="#444")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("Country Menu Clusters — PCA Projection\n(2D view of 18-dimensional cultural space)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.2, linestyle="--")

    plt.tight_layout()
    if save:
        plt.savefig("outputs/country_clusters_pca.png", bbox_inches="tight")
        print("   Saved: outputs/country_clusters_pca.png")
    plt.close()

    return pca, coords


def plot_dietary_heatmap(df_train: pd.DataFrame, save: bool = True):
    """Heatmap of key dietary features across all training countries."""
    df_feat = compute_dietary_constraints(df_train.copy())

    cols_to_show = ["beef_taboo", "pork_taboo", "vegetarian_affinity",
                    "spice_culture_index", "chicken_pref", "fish_pref",
                    "beef_pref", "halal_required"]

    labels = {
        "beef_taboo": "Beef Taboo",
        "pork_taboo": "Pork Taboo",
        "vegetarian_affinity": "Vegetarian\nAffinity",
        "spice_culture_index": "Spice\nCulture",
        "chicken_pref": "Chicken\nPreference",
        "fish_pref": "Fish\nPreference",
        "beef_pref": "Beef\nPreference",
        "halal_required": "Halal\nRequired",
    }

    matrix = df_feat[cols_to_show].T
    matrix.index = [labels[c] for c in cols_to_show]

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(matrix, ax=ax, cmap="RdYlGn_r", vmin=0, vmax=1,
                linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.6, "label": "Score (0=low, 1=high)"},
                annot=True, fmt=".2f", annot_kws={"size": 7})

    ax.set_title("Cultural Dietary Constraint Heatmap — Training Countries", pad=12)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    plt.tight_layout()
    if save:
        plt.savefig("outputs/dietary_heatmap.png", bbox_inches="tight")
        print("   Saved: outputs/dietary_heatmap.png")
    plt.close()


def plot_recommendation_results(recommendations: dict, country: str, save: bool = True):
    """
    Horizontal bar chart of recommended items for a specific country,
    with color coding by protein type.
    """
    recs = recommendations[country]
    menu = recs["recommended_menu"]

    if not menu:
        print(f"No recommendations for {country}")
        return

    names = [item["name"] for item in menu]
    scores = [item["confidence_pct"] for item in menu]
    proteins = [item["protein"] for item in menu]
    vegetarian = [item["vegetarian"] for item in menu]

    protein_colors = {
        "beef": "#E63946", "pork": "#FFBE0B", "chicken": "#F4A261",
        "fish": "#457B9D", "seafood": "#1D3557", "lamb": "#8338EC",
        "dairy": "#2A9D8F", "none": "#6C757D"
    }
    colors = [protein_colors.get(p, "#999") for p in proteins]

    # Reverse for bottom-to-top ordering
    names_r = names[::-1]
    scores_r = scores[::-1]
    colors_r = colors[::-1]
    veg_r = vegetarian[::-1]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.55)))
    bars = ax.barh(names_r, scores_r, color=colors_r,
                   edgecolor="white", linewidth=0.8, height=0.7)

    # Add confidence labels
    for bar, score, is_veg in zip(bars, scores_r, veg_r):
        ax.text(score + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}%  {'🌿' if is_veg else ''}", va="center", fontsize=9)

    # Constraints banner
    constraints = recs["constraints"]
    constraint_text = (
        f"Beef Taboo: {constraints['beef_taboo']:.2f}  |  "
        f"Pork Taboo: {constraints['pork_taboo']:.2f}  |  "
        f"Veg Affinity: {constraints['vegetarian_affinity']:.2f}  |  "
        f"Halal: {'Yes' if constraints['halal_required'] else 'No'}"
    )

    ax.set_xlim(0, 115)
    ax.set_xlabel("Recommendation Confidence (%)")
    ax.set_title(f"Recommended Localized Menu — {country}\n"
                 f"Cluster: {recs['cluster_name']}  |  {constraint_text}",
                 fontsize=11)
    ax.axvline(x=50, color="#ccc", linestyle="--", linewidth=1)
    ax.grid(axis="x", alpha=0.2, linestyle="--")

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=p.title())
                      for p, c in protein_colors.items() if p in proteins]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, ncol=2)

    plt.tight_layout()
    if save:
        fname = f"outputs/recommendation_{country.lower().replace(' ','_')}.png"
        plt.savefig(fname, bbox_inches="tight")
        print(f"   Saved: {fname}")
    plt.close()


def plot_evaluation_results(eval_results: dict, save: bool = True):
    """Bar chart comparing Jaccard, Precision, Recall across holdout countries."""
    countries = list(eval_results.keys())
    jaccard  = [eval_results[c]["jaccard"]   for c in countries]
    precision = [eval_results[c]["precision"] for c in countries]
    recall   = [eval_results[c]["recall"]    for c in countries]

    x = np.arange(len(countries))
    width = 0.28

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, jaccard,   width, label="Jaccard Similarity", color="#2A9D8F", edgecolor="white")
    ax.bar(x,          precision, width, label="Precision@K",        color="#457B9D", edgecolor="white")
    ax.bar(x + width, recall,    width, label="Recall",              color="#E63946", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(countries, rotation=25, ha="right")
    ax.set_ylabel("Score (0–1)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Holdout Evaluation — Menu Reconstruction Quality\n(Countries removed from training, menus reconstructed)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Mean lines
    ax.axhline(np.mean(jaccard), color="#2A9D8F", linestyle=":", linewidth=1.5, alpha=0.6)
    ax.axhline(np.mean(precision), color="#457B9D", linestyle=":", linewidth=1.5, alpha=0.6)

    plt.tight_layout()
    if save:
        plt.savefig("outputs/evaluation_results.png", bbox_inches="tight")
        print("   Saved: outputs/evaluation_results.png")
    plt.close()


def plot_neighbor_similarity(recommendations: dict, country: str, save: bool = True):
    """Horizontal bars showing cultural neighbor similarity scores."""
    neighbors = recommendations[country]["neighbors"]
    nbr_names  = [n["country"] for n in neighbors]
    nbr_sims   = [n["similarity"] * 100 for n in neighbors]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(nbr_names)))
    bars = ax.barh(nbr_names[::-1], nbr_sims[::-1], color=colors, edgecolor="white")

    for bar, val in zip(bars, nbr_sims[::-1]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10)

    ax.set_xlim(0, 110)
    ax.set_xlabel("Cultural Similarity (%)")
    ax.set_title(f"Nearest Cultural Neighbors — {country}\n(Cosine similarity on 18-feature cultural vector)")
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    plt.tight_layout()
    if save:
        fname = f"outputs/neighbors_{country.lower().replace(' ','_')}.png"
        plt.savefig(fname, bbox_inches="tight")
        print(f"   Saved: {fname}")
    plt.close()


if __name__ == "__main__":
    # Quick test
    df_train = pd.read_csv("data/training_countries.csv", index_col="country")
    plot_dietary_heatmap(df_train)
    plot_country_clusters_pca(df_train)
    print("✅ Visualization test complete")
