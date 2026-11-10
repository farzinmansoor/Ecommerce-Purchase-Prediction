
# Inspect Application Inputs

import pandas as pd

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_online_shoppers.csv")

# Display categorical options
categorical_columns = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

print("\n========== CATEGORICAL OPTIONS ==========")

for column in categorical_columns:
    print(f"\n{column}:")
    print(sorted(df[column].unique(), key=str))

# Display numerical ranges
numerical_columns = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay"
]

print("\n========== NUMERICAL RANGES ==========")

for column in numerical_columns:
    print(f"\n{column}:")
    print("Minimum:", df[column].min())
    print("Maximum:", df[column].max())

print("\nINPUT INSPECTION COMPLETED!")
