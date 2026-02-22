"""
Streamlit Dashboard — Cultural Menu Optimizer
Interactive UI to explore country cultural profiles and menu recommendations.

Run with:
    streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cultural Menu Optimizer",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800; color: #DA1F3D;
        margin-bottom: 0; line-height: 1.2;
    }
    .sub-header {
        font-size: 1rem; color: #555; margin-top: 0; margin-bottom: 1rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px; padding: 16px;
        border-left: 4px solid #DA1F3D; margin-bottom: 8px;
    }
    .constraint-chip {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 0.82rem; font-weight: 600; margin: 3px;
    }
    .chip-red   { background: #FFE0E0; color: #c0392b; }
    .chip-green { background: #E0F5EE; color: #1e8449; }
    .chip-blue  { background: #E0ECFF; color: #1a5276; }
    .item-card {
        background: white; border: 1px solid #eee; border-radius: 8px;
        padding: 10px 14px; margin-bottom: 6px;
    }
    .confidence-bar {
        height: 6px; border-radius: 3px; background: #eee; margin-top: 4px;
    }
    .stProgress .st-bo { background-color: #DA1F3D; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    from models.train import MenuRecommender
    from utils.feature_engineering import compute_dietary_constraints

    df_train  = pd.read_csv("data/training_countries.csv", index_col="country")
    df_target = pd.read_csv("data/target_countries.csv",   index_col="country")
    df_all    = pd.concat([df_train.drop(columns=["menu_cluster"]), df_target])

    with open("data/menu_items.json") as f:
        menu_items = json.load(f)
    with open("data/country_menus.json") as f:
        country_menus = json.load(f)
    with open("data/cluster_labels.json") as f:
        cluster_labels = json.load(f)

    if os.path.exists("models/recommender.pkl"):
        try:
            recommender = MenuRecommender.load("models/recommender.pkl")
        except Exception:
            recommender = None

    if not os.path.exists("models/recommender.pkl") or recommender is None:
        recommender = MenuRecommender()
        recommender.fit(df_train, menu_items, country_menus, verbose=False)

    recommendations = recommender.recommend(df_target)

    return df_train, df_target, df_all, menu_items, country_menus, cluster_labels, recommender, recommendations


try:
    df_train, df_target, df_all, menu_items, country_menus, cluster_labels, recommender, precomputed_recs = load_assets()
except Exception as e:
    st.error(f"Error loading assets. Run `python run_pipeline.py` first to generate data.\n\n{e}")
    st.stop()

from utils.feature_engineering import compute_dietary_constraints
from utils.visualization import CLUSTER_COLORS, CLUSTER_NAMES

# Sidebar
with st.sidebar:
    st.markdown("## 🍔 Cultural Menu Optimizer")
    st.markdown("*Predicting localized McDonald's menus using cultural & religious data*")
    st.markdown("---")

    selected_country = st.selectbox(
        "Select Target Country",
        sorted(df_target.index.tolist()),
        index=0
    )
    top_n = st.slider("Items to Recommend", 5, 20, 12)

    elif mode == "📊 Model Insights":
        insight_type = st.selectbox("View", [
            "Feature Importance",
            "Country Cluster Map (PCA)",
            "Dietary Constraint Heatmap",
            "Holdout Evaluation"
        ])

    elif mode == "🔬 What-If Explorer":
        st.markdown("**Adjust country parameters**")
        wi_muslim  = st.slider("Muslim %",  0, 100, 50)
        wi_hindu   = st.slider("Hindu %",   0, 100, 0)
        wi_buddhist= st.slider("Buddhist %",0, 100, 0)
        wi_beef    = st.slider("Beef consumption (kg/capita)", 0, 50, 5)
        wi_pork    = st.slider("Pork consumption (kg/capita)", 0, 50, 0)
        wi_chicken = st.slider("Chicken consumption (kg/capita)", 0, 50, 20)
        wi_gdp     = st.slider("GDP per capita (USD)", 500, 80000, 5000)
        wi_urban   = st.slider("Urbanization %", 10, 100, 55)

# Header
st.markdown('<p class="main-header">🍔 Cultural Menu Optimizer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predictive modeling for fast-food menu localization using religious, dietary, and economic data</p>', unsafe_allow_html=True)
st.markdown("---")

# Mode: Country Recommendation
if mode == "🌍 Country Recommendation":
    recs = precomputed_recs[selected_country]
    country_data = compute_dietary_constraints(df_target).loc[selected_country]

col1, col2, col3 = st.columns([1.2, 1.4, 1.4])

    #Cultural Profile
    with col1:
        st.markdown(f"### 🌐 {selected_country}")
        st.markdown("**Cultural Profile**")

    constraints = recs["constraints"]

    # Radar chart (matplotlib)
    categories = ["Beef Taboo", "Pork Taboo", "Veg Affinity",
                  "Spice Culture", "Halal Req.", "Chicken Pref."]
    values = [
        constraints["beef_taboo"],
        constraints["pork_taboo"],
        constraints["vegetarian_affinity"],
        constraints["spice_culture_index"],
        1.0 if constraints["halal_required"] else 0.2,
        float(country_data.get("chicken_pref", 0.3)),
    ]
    values += values[:1]  # close polygon

    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]

    fig_radar, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="#DA1F3D", linewidth=2)
    ax.fill(angles, values, color="#DA1F3D", alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], size=7, color="gray")
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig_radar)
    plt.close()

    # Key metrics
    st.markdown(f"""
    <div class="metric-card">
        <b>Menu Cluster:</b> {recs['cluster_name']}<br>
        <b>Confidence:</b> {recs['cluster_confidence']*100:.0f}%<br>
        <small style="color:#777">{recs['cluster_description']}</small>
    </div>
    """, unsafe_allow_html=True)

    # Constraint chips
    chips_html = ""
    if constraints["beef_taboo"] > 0.65:
        chips_html += '<span class="constraint-chip chip-red">🚫 No Beef</span>'
    if constraints["pork_taboo"] > 0.50:
        chips_html += '<span class="constraint-chip chip-red">🚫 No Pork</span>'
    if constraints["halal_required"]:
        chips_html += '<span class="constraint-chip chip-green">✅ Halal Required</span>'
    if constraints["vegetarian_affinity"] > 0.4:
        chips_html += '<span class="constraint-chip chip-green">🌿 Veg Friendly</span>'
    if constraints["spice_culture_index"] > 0.6:
        chips_html += '<span class="constraint-chip chip-blue">🌶️ Spice Culture</span>'
    st.markdown(chips_html, unsafe_allow_html=True)

    #Nearest Neighbors
    with col2:
        st.markdown("### 🗺️ Nearest Cultural Neighbors")
        neighbors = recs["neighbors"]

    fig_nbr, ax = plt.subplots(figsize=(5, 4))
    nbr_names = [n["country"] for n in neighbors][::-1]
    nbr_sims  = [n["similarity"] * 100 for n in neighbors][::-1]
    colors = plt.cm.plasma(np.linspace(0.3, 0.85, len(nbr_names)))
    bars = ax.barh(nbr_names, nbr_sims, color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, nbr_sims):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Cultural Similarity (%)")
    ax.set_title(f"Countries most similar to {selected_country}")
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_nbr)
    plt.close()

    # Removed items
    if recs["removed_items"]:
        st.markdown("**❌ Items Excluded (Constraint Violations)**")
        for item in recs["removed_items"]:
            st.markdown(f"- ~~{item['name']}~~ — *{item['removal_reason']}*")

    #Recommended Menu
    with col3:
        st.markdown(f"### 📋 Recommended Menu ({len(recs['recommended_menu'])} items)")

    protein_colors_hex = {
        "beef": "#E63946", "pork": "#FFBE0B", "chicken": "#F4A261",
        "fish": "#457B9D", "seafood": "#1D3557", "lamb": "#8338EC",
        "dairy": "#2A9D8F", "none": "#6C757D"
    }

    menu = recs["recommended_menu"][:top_n]
    fig_menu, ax = plt.subplots(figsize=(5.5, max(5, len(menu) * 0.52)))
    names  = [item["name"] for item in menu][::-1]
    scores = [item["confidence_pct"] for item in menu][::-1]
    proteins = [item["protein"] for item in menu][::-1]
    vegs   = [item["vegetarian"] for item in menu][::-1]
    colors = [protein_colors_hex.get(p, "#999") for p in proteins]
    bars   = ax.barh(names, scores, color=colors, edgecolor="white", height=0.7)
    for bar, score, is_veg in zip(bars, scores, vegs):
        label = f"{score:.1f}%{'  🌿' if is_veg else ''}"
        ax.text(score + 0.5, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=8.5)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)")
    ax.set_title(f"Localized Menu Recommendations")
    ax.axvline(50, color="#ccc", linestyle="--", linewidth=1)
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_patches = [mpatches.Patch(color=c, label=p.title())
                      for p, c in protein_colors_hex.items() if p in proteins]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7.5, ncol=2)
    plt.tight_layout()
    st.pyplot(fig_menu)
    plt.close()


# Mode: Model Insights

elif mode == "📊 Model Insights":
    from utils.visualization import (plot_feature_importance, plot_country_clusters_pca,
                                     plot_dietary_heatmap)
    from sklearn.decomposition import PCA

    if insight_type == "Feature Importance":
        st.markdown("### 🔍 Feature Importance")
        st.markdown("Which cultural features most predict a country's menu localization cluster?")
        fi = recommender.classifier.feature_importance()

        fig, ax = plt.subplots(figsize=(9, 6))
        df_fi = fi.head(15).sort_values("importance")
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, 15))
        bars = ax.barh(df_fi["feature"], df_fi["importance"], color=colors, edgecolor="white")
        for bar, val in zip(bars, df_fi["importance"]):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
        ax.set_xlabel("Importance (Gini)")
        ax.set_title("Top 15 Features for Menu Cluster Prediction")
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    elif insight_type == "Country Cluster Map (PCA)":
        st.markdown("### 🗺️ Country Cluster Map (PCA)")
        st.markdown("2D projection of all 40 training countries in 18-dimensional cultural space.")

        from utils.feature_engineering import get_feature_matrix
        X, cols = get_feature_matrix(df_train)
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X.values)
        clusters = df_train["menu_cluster"].values

        fig, ax = plt.subplots(figsize=(13, 8))
        for cid, color in CLUSTER_COLORS.items():
            mask = clusters == cid
            ax.scatter(coords[mask, 0], coords[mask, 1], c=color, s=130,
                       label=CLUSTER_NAMES[cid], edgecolors="white", linewidth=1.5, zorder=5, alpha=0.9)
            for i, country in enumerate(df_train.index[mask]):
                idx = np.where(mask)[0][i]
                ax.annotate(country, (coords[idx, 0], coords[idx, 1]),
                            textcoords="offset points", xytext=(5, 3), fontsize=7.5, color="#444")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title("Training Countries by Menu Cluster — PCA Projection")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    elif insight_type == "Dietary Constraint Heatmap":
        st.markdown("### 🌶️ Dietary Constraint Heatmap")
        from utils.feature_engineering import compute_dietary_constraints

        df_feat = compute_dietary_constraints(df_train.copy())
        cols = ["beef_taboo", "pork_taboo", "vegetarian_affinity", "spice_culture_index",
                "chicken_pref", "fish_pref", "beef_pref", "halal_required"]
        col_labels = ["Beef Taboo", "Pork Taboo", "Veg Affinity", "Spice Culture",
                      "Chicken Pref", "Fish Pref", "Beef Pref", "Halal Req."]
        matrix = df_feat[cols].T
        matrix.index = col_labels

        import seaborn as sns
        fig, ax = plt.subplots(figsize=(16, 5))
        sns.heatmap(matrix, ax=ax, cmap="RdYlGn_r", vmin=0, vmax=1,
                    linewidths=0.5, linecolor="white",
                    cbar_kws={"shrink": 0.6}, annot=True, fmt=".2f",
                    annot_kws={"size": 7})
        ax.set_title("Cultural Dietary Constraints — All Training Countries")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    elif insight_type == "Holdout Evaluation":
        st.markdown("### 📐 Holdout Evaluation")
        st.markdown("Countries removed from training; menus reconstructed from scratch and compared to real menus.")
        from models.train import evaluate_model

        with st.spinner("Running holdout evaluation..."):
            holdout = ["India", "Japan", "Saudi Arabia", "Malaysia", "Turkey"]
            eval_results = evaluate_model(recommender, holdout)

        countries = list(eval_results.keys())
        jaccard   = [eval_results[c]["jaccard"]   for c in countries]
        precision = [eval_results[c]["precision"] for c in countries]
        recall    = [eval_results[c]["recall"]    for c in countries]

        x = np.arange(len(countries))
        w = 0.28
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - w, jaccard,   w, label="Jaccard", color="#2A9D8F", edgecolor="white")
        ax.bar(x,     precision, w, label="Precision@K", color="#457B9D", edgecolor="white")
        ax.bar(x + w, recall,    w, label="Recall", color="#E63946", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(countries)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("Menu Reconstruction Quality — Holdout Countries")
        ax.legend()
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("**Detailed Results:**")
        df_eval = pd.DataFrame({
            c: {"Jaccard": f"{eval_results[c]['jaccard']:.3f}",
                "Precision": f"{eval_results[c]['precision']:.3f}",
                "Recall": f"{eval_results[c]['recall']:.3f}",
                "Matched Items": ", ".join(eval_results[c]["correct_items"][:4])}
            for c in countries
        }).T
        st.dataframe(df_eval)

# Mode: What-If Explorer
elif mode == "🔬 What-If Explorer":
    st.markdown("### 🔬 What-If Menu Explorer")
    st.markdown("Adjust cultural/religious/dietary parameters and instantly see which menu would be recommended.")

    # Build synthetic country from slider values
    christian = max(0, 100 - wi_muslim - wi_hindu - wi_buddhist)
    wi_row = {
        "christian_pct": christian, "muslim_pct": wi_muslim,
        "hindu_pct": wi_hindu, "buddhist_pct": wi_buddhist,
        "jewish_pct": 0, "other_religion_pct": 0, "nonreligious_pct": 0,
        "beef_kg_capita": wi_beef, "pork_kg_capita": wi_pork,
        "chicken_kg_capita": wi_chicken, "fish_kg_capita": 8,
        "lamb_kg_capita": 3, "gdp_per_capita": wi_gdp, "urbanization_pct": wi_urban
    }
    df_wi = pd.DataFrame([wi_row], index=["CustomCountry"])
    recs_wi = recommender.recommend(df_wi, top_n=12)
    recs = recs_wi["CustomCountry"]

    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.markdown("**Derived Constraints:**")
        c = recs["constraints"]
        st.metric("Beef Taboo Score", f"{c['beef_taboo']:.2f}")
        st.metric("Pork Taboo Score", f"{c['pork_taboo']:.2f}")
        st.metric("Vegetarian Affinity", f"{c['vegetarian_affinity']:.2f}")
        st.metric("Halal Required", "Yes" if c["halal_required"] else "No")
        st.markdown(f"**Predicted Cluster:** `{recs['cluster_name']}`")
        st.markdown(f"*{recs['cluster_description']}*")

    with col2:
        menu = recs["recommended_menu"]
        protein_colors_hex = {
            "beef": "#E63946", "pork": "#FFBE0B", "chicken": "#F4A261",
            "fish": "#457B9D", "seafood": "#1D3557", "lamb": "#8338EC",
            "dairy": "#2A9D8F", "none": "#6C757D"
        }
        names  = [item["name"]           for item in menu][::-1]
        scores = [item["confidence_pct"] for item in menu][::-1]
        proteins = [item["protein"]      for item in menu][::-1]
        colors = [protein_colors_hex.get(p, "#999") for p in proteins]

        fig, ax = plt.subplots(figsize=(7, max(5, len(menu) * 0.52)))
        bars = ax.barh(names, scores, color=colors, edgecolor="white", height=0.7)
        for bar, score in zip(bars, scores):
            ax.text(score + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{score:.1f}%", va="center", fontsize=9)
        ax.set_xlim(0, 115)
        ax.set_xlabel("Confidence (%)")
        ax.set_title("What-If Menu Recommendation")
        ax.axvline(50, color="#ccc", linestyle="--")
        ax.grid(axis="x", alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# Footer
st.markdown("---")
st.markdown(
    "*Data sources: Pew Research Center (religion), FAO (meat consumption), "
    "McDonald's menus (localization patterns), World Population Review (target countries)*"
)
