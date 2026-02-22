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


# ─────────────────────────────────────────────────────────────
# Load Data & Model
# ─────────────────────────────────────────────────────────────
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

    # Check if trained model exists; otherwise train fresh
    if os.path.exists("models/recommender.pkl"):
        try:
            recommender = MenuRecommender.load("models/recommender.pkl")
        except Exception:
            recommender = None

    if not os.path.exists("models/recommender.pkl") or recommender is None:
        recommender = MenuRecommender()
        recommender.fit(df_train, menu_items, country_menus, verbose=False)

    # Pre-compute recommendations for all target countries
    recommendations = recommender.recommend(df_target)

    return df_train, df_target, df_all, menu_items, country_menus, cluster_labels, recommender, recommendations


try:
    df_train, df_target, df_all, menu_items, country_menus, cluster_labels, recommender, precomputed_recs = load_assets()
except Exception as e:
    st.error(f"Error loading assets. Run `python run_pipeline.py` first to generate data.\n\n{e}")
    st.stop()

from utils.feature_engineering import compute_dietary_constraints
from utils.visualization import CLUSTER_COLORS, CLUSTER_NAMES


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🍔 Cultural Menu Optimizer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predictive modeling for fast-food menu localization using religious, dietary, and economic data</p>', unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────────────────────
# Country Recommendation
# ─────────────────────────────────────────────────────────────
recs = precomputed_recs[selected_country]
country_data = compute_dietary_constraints(df_target).loc[selected_country]

col1, col2, col3 = st.columns([1.2, 1.4, 1.4])

# ── Cultural Profile ─────────────────────────────────────
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

# ── Nearest Neighbors ────────────────────────────────────
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

# ── Recommended Menu ─────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "*Data sources: Pew Research Center (religion), FAO (meat consumption), "
    "McDonald's menus (localization patterns), World Population Review (target countries)*"
)
