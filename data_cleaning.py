# Data Cleaning

import pandas as pd
# Load the dataset
df = pd.read_csv("dataset/online_shoppers_intention.csv")

# Display original dataset shape
print("Original Dataset Shape:", df.shape)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate Records:", df.duplicated().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Check dataset shape after removing duplicates
print("\nDataset Shape After Removing Duplicates:", df.shape)

# Check data types
print("\nData Types:")
print(df.dtypes)

# Separate input features and target
X = df.drop("Revenue", axis=1)
y = df["Revenue"]

print("\nNumber of Input Features:", X.shape[1])
print("Target Column:", y.name)

# Display target distribution
print("\nPurchase Distribution:")
print(y.value_counts())

# Save the cleaned dataset
df.to_csv("dataset/cleaned_online_shoppers.csv", index=False)

print("\nCleaned dataset saved successfully!")
