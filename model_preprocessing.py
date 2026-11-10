#Data Preprocessing

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the cleaned dataset
df = pd.read_csv("dataset/cleaned_online_shoppers.csv")

# Separate input features and target
X = df.drop("Revenue", axis=1)
y = df["Revenue"]

# Identify numerical columns
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

# Identify categorical columns
categorical_columns = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create numerical preprocessing
numerical_transformer = StandardScaler()

# Create categorical preprocessing
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# Combine both preprocessing methods
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_columns),
        ("cat", categorical_transformer, categorical_columns)
    ]
)

# Fit preprocessing on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform testing data
X_test_processed = preprocessor.transform(X_test)

# Display results
print("\n========== DATASET SPLITTING ==========")
print("Training Records:", X_train.shape[0])
print("Testing Records:", X_test.shape[0])

print("\n========== PREPROCESSED DATA ==========")
print("Training Data Shape:", X_train_processed.shape)
print("Testing Data Shape:", X_test_processed.shape)

print("\n========== TARGET DISTRIBUTION ==========")
print("Training Target:")
print(y_train.value_counts())

print("\nTesting Target:")
print(y_test.value_counts())

print("\nDAY 2 PREPROCESSING COMPLETED!")
