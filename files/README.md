# 🍔 Cultural Menu Optimizer
### Predictive Modeling for Fast-Food Menu Localization

A machine learning system that recommends localized McDonald's-style menus for countries
that don't currently have McDonald's, based on their religious, dietary, and economic profiles.

---

## 🗂️ Project Structure

```
cultural_menu_optimizer/
├── data/
│   ├── generate_data.py        # Synthetic dataset generator (religion + meat consumption)
│   ├── training_countries.csv  # 40 countries WITH McDonald's (generated)
│   ├── target_countries.csv    # 20 target countries WITHOUT McDonald's (generated)
│   ├── menu_items.json         # 25 menu items with metadata
│   └── country_menus.json      # Real-world menu presence by country
│
├── models/
│   ├── train.py                # LocalizationClassifier + CulturalNeighborFinder + MenuRecommender
│   └── recommender.pkl         # Trained model (generated after running pipeline)
│
├── utils/
│   ├── feature_engineering.py  # Derive beef_taboo, pork_taboo, halal scores etc.
│   └── visualization.py        # All matplotlib/seaborn charts
│
├── dashboard/
│   └── app.py                  # Streamlit interactive dashboard
│
├── outputs/                    # Generated plots + CSV/JSON results
├── run_pipeline.py             # Main end-to-end runner
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python run_pipeline.py
```
This will:
- Generate synthetic training/target data
- Train the localization classifier and neighbor finder
- Run holdout evaluation (India, Japan, Saudi Arabia, Malaysia, Turkey)
- Generate menu recommendations for 20 target countries
- Produce 30+ visualization charts in `outputs/`
- Export `outputs/recommendations.csv` and `outputs/recommendations.json`

### 3. Launch the interactive dashboard
```bash
streamlit run dashboard/app.py
```

---

## 🎯 Pipeline Arguments

```bash
# Run for specific countries only
python run_pipeline.py --countries Ethiopia Iran Nepal

# Change number of recommended items
python run_pipeline.py --top-n 10

# Skip visualizations (faster)
python run_pipeline.py --no-plots
```

---

## 🧠 How It Works

### Stage 1 — Feature Engineering (`utils/feature_engineering.py`)
Raw country data → derived cultural scores:

| Feature | Description |
|---|---|
| `beef_taboo` | 0–1 score from Hindu % + low beef consumption |
| `pork_taboo` | 0–1 score from Muslim/Jewish % + low pork consumption |
| `vegetarian_affinity` | Hindu + Buddhist % + overall low meat intake |
| `spice_culture_index` | South/SE Asian and Middle Eastern spice culture |
| `halal_required` | Boolean: Muslim majority > 60% |
| `chicken_pref` | Chicken's share of total meat consumption |
| `fish_pref` | Fish's share of total meat consumption |

### Stage 2 — Localization Classifier (`models/train.py`)
Random Forest trained on 40 countries with known menu clusters:

| Cluster | Profile |
|---|---|
| BEEF_DOMINANT | USA, Australia, Argentina |
| CHICKEN_DOMINANT | Saudi Arabia, UAE, Egypt, Malaysia |
| VEGETARIAN_HEAVY | India |
| SEAFOOD_ADAPTED | Japan, Korea, Philippines |
| BALANCED_WESTERN | Germany, France, UK |

### Stage 3 — Cultural Neighbor Finder
KNN with cosine similarity finds the 5 most culturally similar training countries
for any new target country.

### Stage 4 — Menu Recommendation
1. Aggregate neighbors' menus weighted by similarity score
2. Apply hard dietary constraints (remove beef/pork items if taboo > threshold)
3. Boost vegetarian items if high affinity
4. Boost lamb/spice items in relevant contexts
5. Return top N items with confidence percentages

### Stage 5 — Evaluation
Holdout test: remove known countries, reconstruct their menus, measure:
- **Jaccard Similarity** — overlap between predicted and actual item sets
- **Precision@K** — of recommended items, how many are actually on real menu
- **Recall** — of real menu items, how many did we correctly include

---

## 📊 Outputs

After running the pipeline:

| File | Description |
|---|---|
| `outputs/recommendations.csv` | Full menu recommendations in flat CSV |
| `outputs/recommendations.json` | Full recommendations with constraints + neighbors |
| `outputs/feature_importance.png` | Top 15 predictive features |
| `outputs/country_clusters_pca.png` | PCA projection of all training countries |
| `outputs/dietary_heatmap.png` | Constraint scores across all countries |
| `outputs/evaluation_results.png` | Holdout evaluation bar chart |
| `outputs/recommendation_ethiopia.png` | Menu chart for Ethiopia |
| `outputs/neighbors_ethiopia.png` | Cultural neighbor chart for Ethiopia |
| ... | (Charts for each target country) |

---

## 🔧 Extending the Project

**Add a real country**: Edit `data/generate_data.py` and add to `target_countries`

**Add a menu item**: Add to `menu_items` list in `generate_data.py`

**Add a new feature**: Edit `utils/feature_engineering.py`'s `get_feature_matrix()`

**Try XGBoost instead of Random Forest**: In `models/train.py`, swap `LocalizationClassifier.model`:
```python
from xgboost import XGBClassifier
self.model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1)
```

---

## 📚 Data Sources (Real-World Implementation)
- **Religion**: [Pew Research Global Religious Futures](https://www.pewresearch.org/religion/2015/04/02/religious-projections-2010-2050/)
- **Meat Consumption**: [FAO FAOSTAT](https://www.fao.org/faostat/en/#data)
- **Countries without McDonald's**: [World Population Review](https://worldpopulationreview.com/country-rankings/countries-without-mcdonalds)
- **McDonald's menus**: [McDonald's country websites / Open Food Facts](https://world.openfoodfacts.org/)
