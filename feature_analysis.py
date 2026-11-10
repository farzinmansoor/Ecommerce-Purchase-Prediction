#Feature Analysis

import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv("dataset/cleaned_online_shoppers.csv")

# Analyse purchases by visitor type
print("\n========== VISITOR TYPE ANALYSIS ==========")

visitor_analysis = pd.crosstab(
    df["VisitorType"],
    df["Revenue"],
    normalize="index"
) * 100

print(visitor_analysis)

# Analyse purchases by month
print("\n========== MONTHLY PURCHASE ANALYSIS ==========")

month_analysis = pd.crosstab(
    df["Month"],
    df["Revenue"],
    normalize="index"
) * 100

print(month_analysis)

# Analyse purchases by weekend
print("\n========== WEEKEND PURCHASE ANALYSIS ==========")

weekend_analysis = pd.crosstab(
    df["Weekend"],
    df["Revenue"],
    normalize="index"
) * 100

print(weekend_analysis)

# Compare average page values
print("\n========== PAGE VALUES ANALYSIS ==========")

print(df.groupby("Revenue")["PageValues"].mean())

# Compare product-related browsing duration
print("\n========== PRODUCT DURATION ANALYSIS ==========")

print(df.groupby("Revenue")["ProductRelated_Duration"].mean())

# Display purchase percentage by visitor type
visitor_analysis.plot(kind="bar", figsize=(8, 5))

plt.title("Purchase Behaviour by Visitor Type")
plt.xlabel("Visitor Type")
plt.ylabel("Percentage of Sessions")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nFEATURE ANALYSIS COMPLETED!")
