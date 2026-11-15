
# =====================================
# E-COMMERCE PURCHASE PREDICTION
# DATA EXPLORATION AND EDA
# =====================================

import os
import pandas as pd
import matplotlib

# Use a non-interactive backend to save charts
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# Create visualization folder if it does not exist
os.makedirs("visualizations", exist_ok=True)

# Load the original dataset
df = pd.read_csv("dataset/online_shoppers_intention.csv")

# =====================================
# BASIC DATASET EXPLORATION
# =====================================

print("\n========== ORIGINAL DATASET ==========")
print(df)

print("\n========== FIRST FIVE RECORDS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DATASET INFORMATION ==========")
df.info()

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE RECORDS ==========")
print("Number of duplicates:", df.duplicated().sum())

print("\n========== PURCHASE DISTRIBUTION ==========")
print(df["Revenue"].value_counts())

print("\n========== PURCHASE PERCENTAGES ==========")
print(df["Revenue"].value_counts(normalize=True) * 100)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# =====================================
# DATA TYPES AND CLEANING INSPECTION
# =====================================

numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_columns = df.select_dtypes(
    include=["object", "bool"]
).columns

print("\nNUMERICAL COLUMNS")
print(numerical_columns.tolist())

print("\nCATEGORICAL COLUMNS")
print(categorical_columns.tolist())

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())

# =====================================
# VISUALIZATION 1
# PURCHASE VS NO PURCHASE
# =====================================

purchase_counts = df["Revenue"].value_counts()

print("\nPURCHASE DISTRIBUTION")
print(purchase_counts)

purchase_counts.plot(
    kind="bar",
    figsize=(7, 5)
)

plt.title("Purchase vs No Purchase")
plt.xlabel("Purchase Result")
plt.ylabel("Number of Sessions")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "visualizations/purchase_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: visualizations/purchase_distribution.png")

# =====================================
# VISUALIZATION 2
# PURCHASE BEHAVIOR BY VISITOR TYPE
# =====================================

visitor_purchase = pd.crosstab(
    df["VisitorType"],
    df["Revenue"]
)

print("\nVISITOR TYPE VS PURCHASE")
print(visitor_purchase)

visitor_purchase.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Purchase Behavior by Visitor Type")
plt.xlabel("Visitor Type")
plt.ylabel("Number of Sessions")
plt.xticks(rotation=0)
plt.legend(title="Revenue")

plt.tight_layout()

plt.savefig(
    "visualizations/visitor_purchase_behavior.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: visualizations/visitor_purchase_behavior.png")

# =====================================
# VISUALIZATION 3
# PRODUCT BROWSING DURATION
# =====================================

df.boxplot(
    column="ProductRelated_Duration",
    by="Revenue",
    figsize=(8, 5)
)

plt.title("Product Browsing Duration by Purchase Result")
plt.suptitle("")
plt.xlabel("Purchase Result")
plt.ylabel("Product Browsing Duration (seconds)")

plt.tight_layout()

plt.savefig(
    "visualizations/product_browsing_duration.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: visualizations/product_browsing_duration.png")

# =====================================
# VISUALIZATION 4
# FEATURE CORRELATION HEATMAP
# =====================================

numerical_data = df.select_dtypes(
    include=["int64", "float64"]
)

correlation = numerical_data.corr()

print("\nCORRELATION MATRIX")
print(correlation)

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    "visualizations/feature_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: visualizations/feature_correlation_heatmap.png")

print("\n=====================================")
print("DATA EXPLORATION AND EDA COMPLETED!")
print("ALL FOUR VISUALIZATIONS SAVED!")
print("=====================================")
