# Import Pandas,.....
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("dataset/online_shoppers_intention.csv")

# Display the complete dataset information
print("\n========== ORIGINAL DATASET ==========")
print(df)

# Display the first five records
print("\n========== FIRST FIVE RECORDS ==========")
print(df.head())

# Display the number of rows and columns
print("\n========== DATASET SHAPE ==========")
print(df.shape)

# Display column names and data types
print("\n========== DATASET INFORMATION ==========")
df.info()

# Display statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# Display missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Check duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print("Number of duplicates:", df.duplicated().sum())

# Display purchase and no-purchase counts
print("\n========== PURCHASE DISTRIBUTION ==========")
print(df["Revenue"].value_counts())

# Display purchase percentages
print("\n========== PURCHASE PERCENTAGES ==========")
print(df["Revenue"].value_counts(normalize=True) * 100)

# Display column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\nDATASET EXPLORATION COMPLETED!")

# DATA CLEANING AND EDA
# Separate numerical and categorical columns
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

# Check data types
print("\nDATA TYPES")
print(df.dtypes)

# Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

#  Check duplicate records
print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())


# Purchase distribution
purchase_counts = df["Revenue"].value_counts()

print("\nPURCHASE DISTRIBUTION")
print(purchase_counts)

# Plot purchase distribution
purchase_counts.plot(
    kind="bar",
    figsize=(7, 5)
)

plt.title("Purchase vs No Purchase")
plt.xlabel("Purchase Result")
plt.ylabel("Number of Sessions")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

# Purchase behavior by visitor type

visitor_purchase = pd.crosstab(
    df["VisitorType"],
    df["Revenue"]
)

print("\nVISITOR TYPE VS PURCHASE")
print(visitor_purchase)

# Visualize visitor purchase behavior

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
plt.show()


# Compare product browsing duration

df.boxplot(
    column="ProductRelated_Duration",
    by="Revenue",
    figsize=(8, 5)
)

plt.title("Product Browsing Duration by Purchase Result")
plt.suptitle("")
plt.xlabel("Purchase Result")
plt.ylabel("Product Browsing Duration")

plt.tight_layout()
plt.show()


# Correlation analysis

numerical_data = df.select_dtypes(
    include=["int64", "float64"]
)

correlation = numerical_data.corr()

print("\nCORRELATION MATRIX")
print(correlation)

# Correlation heatmap

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
