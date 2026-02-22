"""
Data Generator for Cultural Menu Optimizer
Generates realistic synthetic datasets based on real-world patterns:
- Religious composition (Pew Research-style)
- Meat consumption (FAO-style)
- McDonald's menu presence by country
- Countries without McDonald's (targets)
"""

import pandas as pd
import numpy as np
import json
import os

np.random.seed(42)

training_countries = {
    # Country: [christian%, muslim%, hindu%, buddhist%, jewish%, other%, none%,
    #           beef_kg, pork_kg, chicken_kg, fish_kg, lamb_kg,
    #           gdp_per_capita (USD), urbanization%,
    #           menu_cluster]
    # Menu clusters: 0=BEEF_DOMINANT, 1=CHICKEN_DOMINANT, 2=VEGETARIAN_HEAVY, 3=SEAFOOD_ADAPTED, 4=BALANCED_WESTERN

    "USA":          [65, 1, 1, 1, 2, 5, 25,   37.0, 27.0, 49.0, 7.0, 1.0,  63000, 83, 0],
    "Australia":    [52, 3, 2, 3, 1, 5, 34,   33.0, 24.0, 44.0, 8.0, 2.0,  55000, 86, 0],
    "Argentina":    [92, 1, 1, 1, 1, 1,  3,   55.0, 15.0, 36.0, 5.0, 2.0,  11000, 92, 0],
    "Brazil":       [88, 1, 1, 1, 1, 3,  5,   43.0, 14.0, 41.0, 8.0, 1.0,   8700, 87, 0],
    "Canada":       [63, 3, 2, 1, 1, 5, 25,   36.0, 25.0, 40.0, 9.0, 1.0,  46000, 82, 0],
    "Germany":      [57, 5, 1, 1, 1, 5, 30,   36.0, 39.0, 17.0, 9.0, 1.0,  46000, 77, 4],
    "France":       [63, 8, 1, 1, 1, 5, 21,   26.0, 32.0, 24.0,15.0, 2.0,  42000, 82, 4],
    "UK":           [59, 5, 2, 1, 1, 5, 27,   18.0, 25.0, 36.0,13.0, 4.0,  45000, 84, 4],
    "Spain":        [68, 2, 1, 1, 1, 3, 24,   14.0, 37.0, 35.0,17.0, 4.0,  30000, 80, 4],
    "Italy":        [74, 3, 1, 1, 1, 2, 18,   23.0, 40.0, 19.0,14.0, 2.0,  34000, 71, 4],
    "Japan":        [2,  1, 1,67, 1, 3, 25,    9.0,  8.0, 21.0,27.0, 1.0,  40000, 92, 3],
    "South Korea":  [28, 1, 1,22, 1, 3, 44,   13.0, 19.0, 22.0,22.0, 1.0,  32000, 82, 3],
    "Philippines":  [90, 6, 1, 1, 1, 1,  0,    8.0, 14.0, 18.0,29.0, 1.0,   3500, 47, 3],
    "Norway":       [71, 3, 1, 1, 1, 3, 20,   23.0, 13.0, 11.0,26.0, 4.0,  82000, 83, 3],
    "Malaysia":     [10,63, 7, 20,1, 1,  0,    2.0,  1.0, 42.0, 9.0, 3.0,  12000, 78, 1],
    "Saudi Arabia": [5, 93, 1,  1,0, 1,  0,    9.0,  0.0, 48.0, 8.0,18.0,  23000, 84, 1],
    "Egypt":        [10,90, 1,  1,0, 1,  0,   12.0,  0.0, 34.0, 7.0,11.0,   3600, 43, 1],
    "Turkey":       [1, 98, 1,  1,0, 1,  0,   17.0,  0.5, 22.0, 7.0,11.0,  10500, 76, 1],
    "UAE":          [10,76, 8,  1,1, 5,  0,   11.0,  0.1, 40.0, 9.0,14.0,  43000, 87, 1],
    "Indonesia":    [7, 87, 2,  2,0, 2,  0,    2.0,  0.1, 11.0, 8.0, 2.0,   4200, 57, 1],
    "India":        [3,  14,80, 1,0, 1,  1,    2.0,  0.4,  4.0, 4.0, 1.0,   2200, 35, 2],
    "Israel":       [2,  18, 1, 1,75,3,  0,   21.0,  8.0, 68.0, 8.0, 5.0,  47000, 92, 1],
    "Mexico":       [88, 1, 1,  1,1, 3,  5,   19.0, 14.0, 32.0, 6.0, 1.0,  10500, 81, 0],
    "South Africa": [85, 2, 2,  1,1, 4,  5,   18.0,  4.0, 42.0, 5.0, 6.0,   7000, 67, 1],
    "Thailand":     [1,  5, 1, 93,0, 1,  0,    4.0,  8.0, 17.0,17.0, 1.0,   7300, 52, 3],
    "Vietnam":      [7,  1, 1, 16,1, 74, 0,    8.0, 14.0, 14.0,19.0, 1.0,   3600, 38, 3],
    "China":        [5,  2, 1, 18,1, 73, 0,    6.0, 44.0, 14.0,12.0, 1.0,  12500, 64, 3],
    "Russia":       [71, 8, 1,  1,1, 3, 15,   17.0, 22.0, 35.0,13.0, 2.0,  12100, 75, 4],
    "Poland":       [92, 1, 1,  1,1, 1,  3,   16.0, 42.0, 29.0, 9.0, 1.0,  16000, 60, 4],
    "Sweden":       [57, 3, 1,  1,1, 4, 33,   24.0, 18.0, 12.0,18.0, 2.0,  54000, 88, 4],
    "Pakistan":     [2, 97, 1,  0,0, 0,  0,    6.0,  0.0, 10.0, 2.0,12.0,   1500, 37, 1],
    "Bangladesh":   [9, 90, 1,  0,0, 0,  0,    4.0,  0.0,  5.0,11.0, 3.0,   2000, 39, 1],
    "Morocco":      [1, 99, 0,  0,0, 0,  0,    7.0,  0.0, 18.0,10.0,12.0,   3400, 64, 1],
    "Lebanon":      [40,58, 1,  1,0, 0,  0,   10.0,  2.0, 30.0, 9.0,10.0,  13000, 88, 1],
    "Singapore":    [19,14,5,  34,1, 22, 5,    5.0,  8.0, 28.0,17.0, 2.0,  65000, 100,3],
    "New Zealand":  [44, 2, 2,  2,1, 4, 45,   35.0, 19.0, 28.0,13.0,16.0,  42000, 87, 0],
    "Chile":        [66, 1, 1,  1,1, 3, 27,   23.0, 16.0, 42.0,12.0, 1.0,  15000, 88, 0],
    "Peru":         [97, 1, 1,  1,1, 0,  0,    6.0,  5.0, 34.0,22.0, 1.0,   7100, 79, 3],
    "Colombia":     [93, 1, 1,  1,1, 2,  1,   19.0,  3.0, 30.0, 5.0, 1.0,   6400, 81, 0],
    "Nigeria":      [48,50, 0,  0,0, 2,  0,    3.0,  1.0,  9.0, 6.0, 2.0,   2100, 53, 1],
    "Kenya":        [85,11, 2,  0,0, 2,  0,    6.0,  1.0, 11.0, 7.0, 3.0,   1900, 28, 1],
    "Ghana":        [71,18, 1,  0,0,10,  0,    4.0,  2.0, 10.0,12.0, 1.0,   2200, 58, 1],
}

# ─────────────────────────────────────────────
# 2. TARGET COUNTRIES — No McDonald's Yet
# ─────────────────────────────────────────────

target_countries = {
    # Based on World Population Review list of countries without McDonald's
    "Ethiopia":     [63, 34, 0,  0, 0, 3,  0,  5.0, 0.5,  4.0, 4.0, 5.0,  950,  22],
    "Afghanistan":  [1,  99, 0,  0, 0, 0,  0,  5.0, 0.0,  5.0, 1.0,10.0,  500,  26],
    "Nepal":        [3,   4,82,  8, 1, 2,  0,  4.0, 2.5,  3.0, 4.0, 2.0, 1200,  21],
    "Bolivia":      [90,  1, 1,  0, 0, 8,  0, 11.0, 6.0, 18.0, 4.0, 3.0, 3600,  71],
    "Cambodia":     [2,   2, 1, 95, 0, 0,  0,  6.0, 6.0, 10.0,16.0, 1.0, 1700,  25],
    "Laos":         [2,   2, 1, 65, 0,30,  0,  6.0, 7.0, 11.0,14.0, 1.0, 2700,  37],
    "Myanmar":      [4,   4, 1, 88, 0, 3,  0,  4.0, 4.0,  9.0,17.0, 1.0, 1200,  31],
    "Mozambique":   [55,  18,0,  0, 0,27,  0,  3.0, 2.0,  5.0, 7.0, 2.0,  500,  37],
    "Zambia":       [95,  1, 0,  0, 0, 4,  0,  4.0, 2.0,  7.0, 5.0, 1.0, 1300,  45],
    "Iran":         [1,  98, 0,  0,0, 1,  0,  8.0, 0.0, 25.0, 6.0, 8.0, 5600,  76],
    "Sudan":        [3,  97, 0,  0, 0, 0,  0,  5.0, 0.0,  6.0, 3.0,10.0, 700,   35],
    "Syria":        [10, 87, 0,  0, 0, 3,  0,  5.0, 0.5, 18.0, 5.0, 9.0, 1500,  55],
    "Cuba":         [58,  1, 1,  1, 1,10, 28, 14.0, 7.0, 18.0, 8.0, 1.0, 8900,  77],
    "North Korea":  [2,   1, 0, 14, 0,83,  0,  3.0, 6.0,  6.0, 8.0, 1.0,  600,  63],
    "Algeria":      [1,  99, 0,  0, 0, 0,  0,  9.0, 0.0, 15.0, 6.0,12.0, 4200,  74],
    "Libya":        [3,  97, 0,  0, 0, 0,  0,  8.0, 0.0, 13.0, 5.0,11.0, 7400,  80],
    "Yemen":        [1,  99, 0,  0, 0, 0,  0,  6.0, 0.0, 10.0, 5.0,11.0, 1100,  37],
    "Eritrea":      [50, 48, 0,  0, 0, 2,  0,  3.0, 0.0,  4.0, 4.0, 4.0,  700,  41],
    "Bhutan":       [3,   5,22, 70, 0, 0,  0,  4.0, 2.0,  4.0, 6.0, 2.0, 3500,  44],
    "Timor-Leste":  [97,  2, 0,  0, 0, 1,  0,  5.0, 4.0,  8.0,12.0, 1.0, 2000,  32],
}

cols_train = [
    "christian_pct", "muslim_pct", "hindu_pct", "buddhist_pct",
    "jewish_pct", "other_religion_pct", "nonreligious_pct",
    "beef_kg_capita", "pork_kg_capita", "chicken_kg_capita",
    "fish_kg_capita", "lamb_kg_capita",
    "gdp_per_capita", "urbanization_pct",
    "menu_cluster"
]

cols_target = cols_train[:-1]  # No menu_cluster for targets

df_train = pd.DataFrame.from_dict(training_countries, orient="index", columns=cols_train)
df_train.index.name = "country"

df_target = pd.DataFrame.from_dict(target_countries, orient="index", columns=cols_target)
df_target.index.name = "country"

# ─────────────────────────────────────────────
# 3. McDONALD'S MENU ITEM DATABASE
# ─────────────────────────────────────────────

menu_items = [
    # name, protein_type, spice_level(0-3), price_tier(1=budget,2=mid,3=premium), vegetarian
    {"id": "BIG_MAC",         "name": "Big Mac",               "protein": "beef",    "spice": 0, "tier": 2, "vegetarian": False},
    {"id": "QPC",             "name": "Quarter Pounder",       "protein": "beef",    "spice": 0, "tier": 2, "vegetarian": False},
    {"id": "DOUBLE_BURGER",   "name": "Double Cheeseburger",   "protein": "beef",    "spice": 0, "tier": 1, "vegetarian": False},
    {"id": "ANGUS",           "name": "Angus Deluxe",          "protein": "beef",    "spice": 0, "tier": 3, "vegetarian": False},
    {"id": "MCRIB",           "name": "McRib",                 "protein": "pork",    "spice": 1, "tier": 2, "vegetarian": False},
    {"id": "BACON_EGG",       "name": "Bacon Egg McMuffin",    "protein": "pork",    "spice": 0, "tier": 1, "vegetarian": False},
    {"id": "MCCHICKEN",       "name": "McChicken",             "protein": "chicken", "spice": 1, "tier": 1, "vegetarian": False},
    {"id": "SPICY_CHICKEN",   "name": "Spicy McChicken",       "protein": "chicken", "spice": 2, "tier": 1, "vegetarian": False},
    {"id": "CHICKEN_LEGEND",  "name": "Chicken Legend",        "protein": "chicken", "spice": 1, "tier": 2, "vegetarian": False},
    {"id": "GRILLED_CHICKEN", "name": "Grilled Chicken Wrap",  "protein": "chicken", "spice": 0, "tier": 2, "vegetarian": False},
    {"id": "MCSPICY",         "name": "McSpicy Chicken",       "protein": "chicken", "spice": 3, "tier": 2, "vegetarian": False},
    {"id": "CHICKEN_MAHARAJA","name": "Chicken Maharaja Mac",  "protein": "chicken", "spice": 3, "tier": 2, "vegetarian": False},
    {"id": "FILLET_O_FISH",   "name": "Filet-O-Fish",         "protein": "fish",    "spice": 0, "tier": 1, "vegetarian": False},
    {"id": "SHRIMP_BURGER",   "name": "Ebi Shrimp Burger",    "protein": "seafood", "spice": 1, "tier": 2, "vegetarian": False},
    {"id": "FISH_FILLET_DLX", "name": "Fish Fillet Deluxe",   "protein": "fish",    "spice": 0, "tier": 2, "vegetarian": False},
    {"id": "LAMB_WRAP",       "name": "Lamb Wrap",             "protein": "lamb",    "spice": 2, "tier": 2, "vegetarian": False},
    {"id": "MCALOO_TIKKI",    "name": "McAloo Tikki",          "protein": "none",    "spice": 2, "tier": 1, "vegetarian": True},
    {"id": "MCVEGGIE",        "name": "McVeggie",              "protein": "none",    "spice": 1, "tier": 1, "vegetarian": True},
    {"id": "PANEER_BURGER",   "name": "McSpicy Paneer",        "protein": "dairy",   "spice": 3, "tier": 2, "vegetarian": True},
    {"id": "VEGGIE_DELUXE",   "name": "Veggie Deluxe",         "protein": "none",    "spice": 1, "tier": 2, "vegetarian": True},
    {"id": "FRIES",           "name": "French Fries",          "protein": "none",    "spice": 0, "tier": 1, "vegetarian": True},
    {"id": "APPLE_PIE",       "name": "Apple Pie",             "protein": "none",    "spice": 0, "tier": 1, "vegetarian": True},
    {"id": "MCFLURRY",        "name": "McFlurry",              "protein": "dairy",   "spice": 0, "tier": 1, "vegetarian": True},
    {"id": "RICE_BURGER",     "name": "Rice Burger",           "protein": "chicken", "spice": 1, "tier": 2, "vegetarian": False},
    {"id": "TERIYAKI",        "name": "Teriyaki McBurger",     "protein": "chicken", "spice": 1, "tier": 2, "vegetarian": False},
]

# Country → which item IDs are on their menu (based on real-world localization patterns)
country_menus = {
    "USA":          ["BIG_MAC","QPC","DOUBLE_BURGER","ANGUS","MCRIB","BACON_EGG","MCCHICKEN","SPICY_CHICKEN","CHICKEN_LEGEND","FILLET_O_FISH","FRIES","APPLE_PIE","MCFLURRY"],
    "Australia":    ["BIG_MAC","QPC","DOUBLE_BURGER","MCCHICKEN","SPICY_CHICKEN","FILLET_O_FISH","LAMB_WRAP","FRIES","APPLE_PIE","MCFLURRY"],
    "India":        ["MCCHICKEN","SPICY_CHICKEN","CHICKEN_MAHARAJA","MCALOO_TIKKI","MCVEGGIE","PANEER_BURGER","VEGGIE_DELUXE","FILLET_O_FISH","FRIES","APPLE_PIE","MCFLURRY"],
    "Japan":        ["MCCHICKEN","FILLET_O_FISH","SHRIMP_BURGER","FISH_FILLET_DLX","RICE_BURGER","TERIYAKI","MCFLURRY","FRIES","APPLE_PIE","MCSPICY"],
    "Saudi Arabia": ["MCCHICKEN","SPICY_CHICKEN","CHICKEN_LEGEND","LAMB_WRAP","FILLET_O_FISH","MCSPICY","FRIES","APPLE_PIE","MCFLURRY"],
    "Germany":      ["BIG_MAC","QPC","BACON_EGG","MCCHICKEN","FILLET_O_FISH","MCRIB","FRIES","APPLE_PIE","MCFLURRY"],
    "Malaysia":     ["MCCHICKEN","SPICY_CHICKEN","CHICKEN_MAHARAJA","MCSPICY","FILLET_O_FISH","SHRIMP_BURGER","FRIES","APPLE_PIE","MCFLURRY"],
    "Philippines":  ["MCCHICKEN","SPICY_CHICKEN","FILLET_O_FISH","SHRIMP_BURGER","RICE_BURGER","FRIES","APPLE_PIE","MCFLURRY"],
    "Thailand":     ["MCCHICKEN","SPICY_CHICKEN","MCSPICY","FILLET_O_FISH","SHRIMP_BURGER","RICE_BURGER","FRIES","APPLE_PIE","MCFLURRY"],
    "South Korea":  ["MCCHICKEN","SPICY_CHICKEN","FILLET_O_FISH","SHRIMP_BURGER","RICE_BURGER","TERIYAKI","MCFLURRY","FRIES"],
    "Israel":       ["MCCHICKEN","CHICKEN_LEGEND","GRILLED_CHICKEN","FILLET_O_FISH","FRIES","APPLE_PIE","MCFLURRY"],
    "Egypt":        ["MCCHICKEN","SPICY_CHICKEN","CHICKEN_LEGEND","LAMB_WRAP","FILLET_O_FISH","FRIES","APPLE_PIE","MCFLURRY"],
    "Turkey":       ["MCCHICKEN","SPICY_CHICKEN","LAMB_WRAP","FILLET_O_FISH","FRIES","APPLE_PIE","MCFLURRY"],
    "UAE":          ["MCCHICKEN","SPICY_CHICKEN","CHICKEN_LEGEND","LAMB_WRAP","FILLET_O_FISH","MCSPICY","FRIES","APPLE_PIE","MCFLURRY"],
    "France":       ["BIG_MAC","QPC","MCCHICKEN","CHICKEN_LEGEND","FILLET_O_FISH","MCFLURRY","FRIES","APPLE_PIE"],
}

# Save everything
os.makedirs("data", exist_ok=True)
df_train.to_csv("data/training_countries.csv")
df_target.to_csv("data/target_countries.csv")

with open("data/menu_items.json", "w") as f:
    json.dump(menu_items, f, indent=2)

with open("data/country_menus.json", "w") as f:
    json.dump(country_menus, f, indent=2)

cluster_labels = {0: "BEEF_DOMINANT", 1: "CHICKEN_DOMINANT", 2: "VEGETARIAN_HEAVY", 3: "SEAFOOD_ADAPTED", 4: "BALANCED_WESTERN"}
with open("data/cluster_labels.json", "w") as f:
    json.dump(cluster_labels, f, indent=2)

print("✅ Data generated successfully!")
print(f"   Training countries: {len(df_train)}")
print(f"   Target countries: {len(df_target)}")
print(f"   Menu items: {len(menu_items)}")
print(f"   Countries with full menus: {len(country_menus)}")
