#Final Data Preparation

import pandas as pd

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_online_shoppers.csv")

# Separate input features and target
X = df.drop("Revenue", axis=1)
y = df["Revenue"]

# Display input features
print("\n========== INPUT FEATURES ==========")
print(X.head())

# Display target values
print("\n========== TARGET VALUES ==========")
print(y.head())

# Display dataset dimensions
print("\n========== DATASET DIMENSIONS ==========")
print("Input Features:", X.shape)
print("Target:", y.shape)

# Display numerical and categorical columns
numerical_columns = X.select_dtypes(include=["int64", "float64"]).columns
categorical_columns = X.select_dtypes(include=["object", "bool"]).columns

print("\nNumerical Columns:")
print(numerical_columns.tolist())

print("\nCategorical Columns:")
print(categorical_columns.tolist())

print("\n COMPLETED SUCCESSFULLY!")
