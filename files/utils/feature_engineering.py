"""
Feature Engineering for Cultural Menu Optimizer
Transforms raw country data into model-ready features including:
- Dietary constraint scores
- Cultural similarity vectors
- Normalized economic features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def compute_dietary_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive dietary constraint scores from religious and consumption data.
    All scores range 0-1 where 1 = strong constraint/preference.
    """
    df = df.copy()

    # BEEF TABOO: Hindu belief + low beef consumption signal
    df["beef_taboo"] = (
        0.6 * (df["hindu_pct"] / 100) +
        0.4 * (1 - df["beef_kg_capita"].clip(0, 30) / 30)
    ).clip(0, 1)

    # PORK TABOO: Muslim + Jewish belief (both prohibit pork)
    pork_religious = ((df["muslim_pct"] + df["jewish_pct"]) / 100).clip(0, 1)
    df["pork_taboo"] = (
        0.7 * pork_religious +
        0.3 * (1 - df["pork_kg_capita"].clip(0, 30) / 30)
    ).clip(0, 1)

    # VEGETARIAN AFFINITY: Hindu + Buddhist + low meat overall
    total_meat = (df["beef_kg_capita"] + df["pork_kg_capita"] +
                  df["chicken_kg_capita"] + df["lamb_kg_capita"]).clip(0, 100)
    df["vegetarian_affinity"] = (
        0.5 * (df["hindu_pct"] / 100) +
        0.2 * (df["buddhist_pct"] / 100) +
        0.3 * (1 - total_meat / 100)
    ).clip(0, 1)

    # SPICE CULTURE INDEX: South/Southeast Asia, Middle East, Africa
    df["spice_culture_index"] = (
        0.4 * (df["hindu_pct"] / 100) +
        0.3 * (df["muslim_pct"] / 100).clip(0, 0.8) +
        0.3 * (df["buddhist_pct"] / 100).clip(0, 0.5)
    ).clip(0.1, 0.95)

    # PROTEIN PREFERENCES (normalized fractions)
    total_meat_all = (
        df["beef_kg_capita"] + df["pork_kg_capita"] +
        df["chicken_kg_capita"] + df["fish_kg_capita"] +
        df["lamb_kg_capita"]
    ).replace(0, 1)

    df["chicken_pref"] = (df["chicken_kg_capita"] / total_meat_all).clip(0, 1)
    df["fish_pref"]    = (df["fish_kg_capita"] / total_meat_all).clip(0, 1)
    df["beef_pref"]    = (df["beef_kg_capita"] / total_meat_all).clip(0, 1)
    df["pork_pref"]    = (df["pork_kg_capita"] / total_meat_all).clip(0, 1)
    df["lamb_pref"]    = (df["lamb_kg_capita"] / total_meat_all).clip(0, 1)

    # HALAL REQUIREMENT (strong Muslim majority)
    df["halal_required"] = (df["muslim_pct"] >= 60).astype(float)
    df["halal_preferred"] = ((df["muslim_pct"] >= 20) & (df["muslim_pct"] < 60)).astype(float)

    return df


def normalize_features(df: pd.DataFrame) -> tuple:
    """
    Normalize numeric features to 0-1 range.
    Returns normalized df and the fitted scaler.
    """
    scale_cols = ["gdp_per_capita", "urbanization_pct",
                  "beef_kg_capita", "pork_kg_capita", "chicken_kg_capita",
                  "fish_kg_capita", "lamb_kg_capita"]

    scaler = MinMaxScaler()
    df = df.copy()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    return df, scaler


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the final feature matrix used for model training and prediction.
    """
    df = compute_dietary_constraints(df)

    feature_cols = [
        # Religious
        "christian_pct", "muslim_pct", "hindu_pct", "buddhist_pct", "nonreligious_pct",
        # Dietary derived
        "beef_taboo", "pork_taboo", "vegetarian_affinity", "spice_culture_index",
        "halal_required", "halal_preferred",
        # Protein preferences
        "chicken_pref", "fish_pref", "beef_pref", "pork_pref", "lamb_pref",
        # Economic
        "gdp_per_capita", "urbanization_pct",
    ]

    # Normalize economic cols only
    scaler = MinMaxScaler()
    df_feat = df[feature_cols].copy()
    df_feat[["gdp_per_capita", "urbanization_pct"]] = scaler.fit_transform(
        df_feat[["gdp_per_capita", "urbanization_pct"]]
    )
    # Normalize religious percentages to 0-1
    for col in ["christian_pct", "muslim_pct", "hindu_pct", "buddhist_pct", "nonreligious_pct"]:
        df_feat[col] = df_feat[col] / 100.0

    return df_feat, feature_cols


def get_cultural_profile(country_row: pd.Series) -> dict:
    """
    Return a human-readable cultural profile for a country.
    """
    profile = {}

    # Dominant religion
    rel_cols = {"christian_pct": "Christian", "muslim_pct": "Muslim",
                "hindu_pct": "Hindu", "buddhist_pct": "Buddhist",
                "jewish_pct": "Jewish", "nonreligious_pct": "Non-religious"}
    dominant = max(rel_cols, key=lambda c: country_row.get(c, 0))
    profile["dominant_religion"] = rel_cols[dominant]
    profile["dominant_religion_pct"] = country_row.get(dominant, 0)

    # Dietary flags
    profile["beef_taboo_score"] = round(country_row.get("beef_taboo", 0), 2)
    profile["pork_taboo_score"] = round(country_row.get("pork_taboo", 0), 2)
    profile["vegetarian_affinity"] = round(country_row.get("vegetarian_affinity", 0), 2)
    profile["spice_culture_index"] = round(country_row.get("spice_culture_index", 0), 2)
    profile["halal_required"] = bool(country_row.get("halal_required", 0))

    return profile


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    df_train = pd.read_csv("data/training_countries.csv", index_col="country")
    df_feat, cols = get_feature_matrix(df_train)
    print("Feature matrix shape:", df_feat.shape)
    print("Features:", cols)
    print("\nSample — India:")
    india = df_train.loc["India"]
    india_feat = compute_dietary_constraints(pd.DataFrame([india])).iloc[0]
    print(get_cultural_profile(india_feat))
