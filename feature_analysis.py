
# =====================================
# E-COMMERCE PURCHASE PREDICTION
# FEATURE ANALYSIS
# =====================================

import os
import pandas as pd
import matplotlib

# Use a non-interactive backend for saving charts
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Create visualization folder if needed
os.makedirs("visualizations", exist_ok=True)

# Load the cleaned dataset
df = pd.read_csv("dataset/cleaned_online_shoppers.csv")

# =====================================
# VISITOR TYPE ANALYSIS
# =====================================

print("\n========== VISITOR TYPE ANALYSIS ==========")

visitor_analysis = pd.crosstab(
    df["VisitorType"],
    df["Revenue"],
    normalize="index"
) * 100

print(visitor_analysis)

# =====================================
# MONTHLY PURCHASE ANALYSIS
# =====================================

print("\n========== MONTHLY PURCHASE ANALYSIS ==========")

month_analysis = pd.crosstab(
    df["Month"],
    df["Revenue"],
    normalize="index"
) * 100

print(month_analysis)

# =====================================
# WEEKEND PURCHASE ANALYSIS
# =====================================

print("\n========== WEEKEND PURCHASE ANALYSIS ==========")

weekend_analysis = pd.crosstab(
    df["Weekend"],
    df["Revenue"],
    normalize="index"
) * 100

print(weekend_analysis)

# =====================================
# PAGE VALUES ANALYSIS
# =====================================

print("\n========== PAGE VALUES ANALYSIS ==========")

print(df.groupby("Revenue")["PageValues"].mean())

# =====================================
# PRODUCT BROWSING DURATION ANALYSIS
# =====================================

print("\n========== PRODUCT DURATION ANALYSIS ==========")

print(df.groupby("Revenue")["ProductRelated_Duration"].mean())

# =====================================
# VISUALIZATION 5
# PURCHASE PERCENTAGE BY VISITOR TYPE
# =====================================

visitor_analysis.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Purchase Behaviour by Visitor Type")
plt.xlabel("Visitor Type")
plt.ylabel("Percentage of Sessions")
plt.xticks(rotation=0)
plt.legend(title="Revenue")

plt.tight_layout()

plt.savefig(
    "visualizations/visitor_purchase_percentage.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSaved: visualizations/visitor_purchase_percentage.png"
)

print("\n=====================================")
print("FEATURE ANALYSIS COMPLETED!")
print("VISUALIZATION SAVED SUCCESSFULLY!")
print("=====================================")
