
# Save Best Model

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load cleaned dataset
df = pd.read_csv("dataset/cleaned_online_shoppers.csv")

# STEP 2: Separate input features and target
X = df.drop("Revenue", axis=1)
y = df["Revenue"]

# Identify numerical columns
numerical_columns = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay"
]

# Identify categorical columns
categorical_columns = [
    "Month", "OperatingSystems", "Browser",
    "Region", "TrafficType", "VisitorType", "Weekend"
]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ]
)

# Create best Random Forest model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ))
    ]
)

# Train the model
model.fit(X_train, y_train)

# Save the complete pipeline
joblib.dump(model, "models/best_purchase_model.pkl")

print("\nBest model saved successfully!")
print("Location: models/best_purchase_model.pkl")
print("\nMODEL SAVING COMPLETED!")
