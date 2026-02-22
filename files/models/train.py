"""
Model Training for Cultural Menu Optimizer
Two-stage modeling approach:
1. Localization Classifier: XGBoost/RandomForest to predict menu cluster
2. Cultural Neighbor Finder: KNN cosine similarity for nearest cultural matches
3. Item-Level Recommender: Binary classifiers per protein type
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
import sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_engineering import get_feature_matrix, compute_dietary_constraints


class LocalizationClassifier:
    CLUSTER_NAMES = {
        0: "BEEF_DOMINANT",
        1: "CHICKEN_DOMINANT",
        2: "VEGETARIAN_HEAVY",
        3: "SEAFOOD_ADAPTED",
        4: "BALANCED_WESTERN"
    }
    CLUSTER_DESCRIPTIONS = {
        0: "Heavy beef-focused menus; staple items like Big Mac and Quarter Pounder dominate",
        1: "Chicken-centric menus; halal-certified; strong spiced variants popular",
        2: "Vegetarian options lead; minimal beef/pork; strong spice culture",
        3: "Fish/seafood prominent; lighter fare; East/Southeast Asian palate",
        4: "Balanced European-style menus; mix of beef, chicken, pork offerings"
    }
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42
        )
        self.feature_cols = None
        self.is_trained = False
    def train(self, df_train: pd.DataFrame, verbose: bool = True):
        X, self.feature_cols = get_feature_matrix(df_train)
        y = df_train["menu_cluster"]

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring="accuracy")

        self.model.fit(X, y)
        self.is_trained = True

        if verbose:
            print(f"✅ LocalizationClassifier trained")
            print(f"   CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"   Training samples: {len(X)}")

        return cv_scores

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.is_trained, "Model not trained yet"
        X, _ = get_feature_matrix(df)
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)

        results = pd.DataFrame({
            "country": df.index,
            "predicted_cluster": preds,
            "cluster_name": [self.CLUSTER_NAMES[p] for p in preds],
            "confidence": probs.max(axis=1).round(3)
        })
        results["cluster_description"] = results["predicted_cluster"].map(self.CLUSTER_DESCRIPTIONS)
        return results.set_index("country")

    def feature_importance(self) -> pd.DataFrame:
        assert self.is_trained
        return pd.DataFrame({
            "feature": self.feature_cols,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)


class CulturalNeighborFinder:
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.knn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="cosine",
            algorithm="brute"
        )
        self.train_index = None
        self.feature_cols = None
        self.X_train = None
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame, verbose: bool = True):
        X, self.feature_cols = get_feature_matrix(df_train)
        self.X_train = X.values
        self.train_index = df_train.index.tolist()
        self.knn.fit(self.X_train)
        self.is_fitted = True

        if verbose:
            print(f"✅ CulturalNeighborFinder fitted on {len(df_train)} countries")

    def find_neighbors(self, df_query: pd.DataFrame, return_distances: bool = True) -> dict:
        """
        For each country in df_query, return its nearest training neighbors.
        """
        assert self.is_fitted
        X_query, _ = get_feature_matrix(df_query)
        distances, indices = self.knn.kneighbors(X_query.values)

        results = {}
        for i, country in enumerate(df_query.index):
            neighbors = []
            for dist, idx in zip(distances[i], indices[i]):
                similarity = 1 - dist  # cosine distance → similarity
                neighbor_name = self.train_index[idx]
                neighbors.append({
                    "country": neighbor_name,
                    "similarity": round(float(similarity), 3)
                })
            results[country] = neighbors

        return results

    def similarity_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute full pairwise similarity matrix for a set of countries."""
        X, _ = get_feature_matrix(df)
        sim = cosine_similarity(X.values)
        return pd.DataFrame(sim, index=df.index, columns=df.index)


class MenuRecommender:
    """
    1. Find nearest cultural neighbors
    2. Aggregate their menus (weighted by similarity)
    3. Apply hard dietary constraints (remove beef if beef_taboo > threshold, etc.)
    4. Score remaining items by weighted frequency
    5. Return top N items with confidence scores and rationale
    """

    BEEF_TABOO_THRESHOLD = 0.65
    PORK_TABOO_THRESHOLD = 0.50
    VEGETARIAN_BOOST_THRESHOLD = 0.40

    def __init__(self):
        self.classifier = LocalizationClassifier()
        self.neighbor_finder = CulturalNeighborFinder(n_neighbors=5)
        self.menu_items = None
        self.country_menus = None
        self.df_train = None

    def fit(self, df_train: pd.DataFrame, menu_items: list, country_menus: dict, verbose: bool = True):
        self.df_train = df_train
        self.menu_items = {item["id"]: item for item in menu_items}
        self.country_menus = country_menus

        self.classifier.train(df_train, verbose=verbose)
        self.neighbor_finder.fit(df_train, verbose=verbose)

    def recommend(self, df_target: pd.DataFrame, top_n: int = 14) -> dict:
        df_target_feat = compute_dietary_constraints(df_target)
        neighbors_map = self.neighbor_finder.find_neighbors(df_target)
        cluster_preds = self.classifier.predict(df_target)

        results = {}

        for country in df_target.index:
            country_data = df_target_feat.loc[country]
            neighbors = neighbors_map[country]
            cluster_row = cluster_preds.loc[country]

            # ── Compute dietary constraints ──────────────────────────────
            beef_taboo   = float(country_data.get("beef_taboo", 0))
            pork_taboo   = float(country_data.get("pork_taboo", 0))
            veg_affinity = float(country_data.get("vegetarian_affinity", 0))
            spice_index  = float(country_data.get("spice_culture_index", 0))
            halal_req    = float(country_data.get("halal_required", 0)) > 0.5

            constraints = {
                "beef_taboo": round(beef_taboo, 2),
                "pork_taboo": round(pork_taboo, 2),
                "vegetarian_affinity": round(veg_affinity, 2),
                "spice_culture_index": round(spice_index, 2),
                "halal_required": halal_req,
            }

            # ── Score items from neighbor menus ──────────────────────────
            item_scores = {}
            total_weight = sum(n["similarity"] for n in neighbors)

            for neighbor_info in neighbors:
                nbr = neighbor_info["country"]
                weight = neighbor_info["similarity"] / max(total_weight, 1e-9)

                if nbr not in self.country_menus:
                    continue

                for item_id in self.country_menus[nbr]:
                    item_scores[item_id] = item_scores.get(item_id, 0) + weight

            # Normalize scores to 0-1
            if item_scores:
                max_score = max(item_scores.values())
                item_scores = {k: v / max_score for k, v in item_scores.items()}

            # ── Apply hard constraints ────────────────────────────────────
            filtered_items = []
            for item_id, raw_score in item_scores.items():
                item = self.menu_items.get(item_id)
                if not item:
                    continue

                protein = item["protein"]
                spice   = item["spice"]

                # Hard exclusions
                if protein == "beef" and beef_taboo > self.BEEF_TABOO_THRESHOLD:
                    continue
                if protein == "pork" and pork_taboo > self.PORK_TABOO_THRESHOLD:
                    continue

                # Adjust score based on cultural fit
                adjusted_score = raw_score

                # Boost vegetarian if high affinity
                if item["vegetarian"] and veg_affinity > self.VEGETARIAN_BOOST_THRESHOLD:
                    adjusted_score *= 1.4

                # Boost spicy items in high-spice cultures
                if spice >= 2 and spice_index > 0.6:
                    adjusted_score *= 1.2

                # Boost lamb in Muslim-majority contexts
                if protein == "lamb" and halal_req:
                    adjusted_score *= 1.3

                # Always include base items (fries, desserts)
                if protein == "none" and item["vegetarian"]:
                    adjusted_score = max(adjusted_score, 0.7)

                # Penalize pork in halal-required contexts (don't remove, just penalize for soft cases)
                if protein == "pork" and pork_taboo > 0.3:
                    adjusted_score *= (1 - pork_taboo)

                filtered_items.append({
                    "id": item_id,
                    "name": item["name"],
                    "protein": protein,
                    "spice": spice,
                    "vegetarian": item["vegetarian"],
                    "score": min(round(adjusted_score, 3), 1.0),
                    "confidence_pct": min(round(adjusted_score * 100, 1), 99.0),
                    "removal_reason": None
                })

            # Add removed items for transparency
            removed_items = []
            for item_id, item in self.menu_items.items():
                if item_id in item_scores:
                    protein = item["protein"]
                    if protein == "beef" and beef_taboo > self.BEEF_TABOO_THRESHOLD:
                        removed_items.append({"id": item_id, "name": item["name"],
                                               "removal_reason": f"Beef taboo score: {beef_taboo:.2f}"})
                    elif protein == "pork" and pork_taboo > self.PORK_TABOO_THRESHOLD:
                        removed_items.append({"id": item_id, "name": item["name"],
                                               "removal_reason": f"Pork taboo score: {pork_taboo:.2f}"})

            # Sort and take top N
            filtered_items.sort(key=lambda x: x["score"], reverse=True)
            recommended = filtered_items[:top_n]

            results[country] = {
                "cluster": int(cluster_row["predicted_cluster"]),
                "cluster_name": cluster_row["cluster_name"],
                "cluster_confidence": float(cluster_row["confidence"]),
                "cluster_description": cluster_row["cluster_description"],
                "neighbors": neighbors,
                "constraints": constraints,
                "recommended_menu": recommended,
                "removed_items": removed_items[:5]
            }

        return results

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MenuRecommender":
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"✅ Model loaded from {path}")
        return model


def evaluate_model(recommender: MenuRecommender, holdout_countries: list) -> dict:
    """
    Evaluate recommender by holding out countries with known menus
    and measuring how well we reconstruct their actual menus.
    """
    df_train_full = recommender.df_train
    known_menus = recommender.country_menus

    results = {}
    for country in holdout_countries:
        if country not in known_menus:
            continue

        # Temporarily remove from training
        df_query = df_train_full.loc[[country]].drop(columns=["menu_cluster"], errors="ignore")

        # Get recommendations
        recs = recommender.recommend(df_query, top_n=12)
        predicted_ids = {item["id"] for item in recs[country]["recommended_menu"]}
        actual_ids = set(known_menus[country])

        # Jaccard similarity
        intersection = len(predicted_ids & actual_ids)
        union = len(predicted_ids | actual_ids)
        jaccard = intersection / union if union > 0 else 0

        # Precision@K
        precision_at_k = intersection / len(predicted_ids) if predicted_ids else 0
        recall = intersection / len(actual_ids) if actual_ids else 0

        results[country] = {
            "jaccard": round(jaccard, 3),
            "precision": round(precision_at_k, 3),
            "recall": round(recall, 3),
            "correct_items": list(predicted_ids & actual_ids),
            "missed_items": list(actual_ids - predicted_ids),
            "extra_items": list(predicted_ids - actual_ids)
        }

    return results


if __name__ == "__main__":
    df_train = pd.read_csv("data/training_countries.csv", index_col="country")
    df_target = pd.read_csv("data/target_countries.csv", index_col="country")

    with open("data/menu_items.json") as f:
        menu_items = json.load(f)
    with open("data/country_menus.json") as f:
        country_menus = json.load(f)

    recommender = MenuRecommender()
    recommender.fit(df_train, menu_items, country_menus)

    print("\n📊 Feature Importance (Top 10):")
    print(recommender.classifier.feature_importance().head(10).to_string(index=False))

    print("\n🍔 Recommendations for Ethiopia:")
    recs = recommender.recommend(df_target.loc[["Ethiopia"]])
    eth = recs["Ethiopia"]
    print(f"Cluster: {eth['cluster_name']}")
    print(f"Constraints: {eth['constraints']}")
    for item in eth["recommended_menu"][:8]:
        print(f"  {item['confidence_pct']:5.1f}%  {item['name']}")

    recommender.save("models/recommender.pkl")
