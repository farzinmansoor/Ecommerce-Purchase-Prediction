
# ==========================================
# FINAL ANN EVALUATION
# E-COMMERCE PURCHASE PREDICTION
# ==========================================

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/cleaned_online_shoppers.csv")

X = df.drop(columns=["Revenue"])
y = df["Revenue"].astype(int)

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

categorical_columns = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

# ==========================================
# RECREATE ORIGINAL DATA SPLIT
# ==========================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=42,
    stratify=y_train_full
)

# ==========================================
# LOAD SAVED ANN PREPROCESSOR
# ==========================================

preprocessor = joblib.load(
    "models/ann_preprocessor.pkl"
)

X_test_processed = preprocessor.transform(X_test)

if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()

X_test_processed = np.asarray(
    X_test_processed,
    dtype=np.float32
)

print("Saved ANN preprocessor loaded successfully!")
print("Processed Test Features:", X_test_processed.shape[1])

# ==========================================
# LOAD BEST EXPERIMENTAL ANN
# ==========================================

model = tf.keras.models.load_model(
    "models/ann_best_experiment.keras"
)

# ==========================================
# PREDICT AND EVALUATE
# ==========================================

probabilities = model.predict(
    X_test_processed,
    verbose=0
).ravel()

predictions = (probabilities >= 0.5).astype(int)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions, zero_division=0
)
recall = recall_score(
    y_test, predictions, zero_division=0
)
f1 = f1_score(
    y_test, predictions, zero_division=0
)

print("\n========== BEST ANN FINAL TEST RESULTS ==========")

print("Test Records:", len(y_test))
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nFINAL ANN EVALUATION COMPLETED!")
