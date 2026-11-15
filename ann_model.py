
# ==========================================
# E-COMMERCE PURCHASE PREDICTION
# ARTIFICIAL NEURAL NETWORK (ANN)
# ==========================================

import os
import random

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf

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
# STEP 1: REPRODUCIBILITY
# ==========================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

os.makedirs("visualizations", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ==========================================
# STEP 2: LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/cleaned_online_shoppers.csv")

X = df.drop("Revenue", axis=1)
y = df["Revenue"].astype(int)

print("\n========== DATASET INFORMATION ==========")
print("Dataset Shape:", df.shape)

# ==========================================
# STEP 3: IDENTIFY FEATURES
# ==========================================

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
# STEP 4: TRAIN / VALIDATION / TEST SPLIT
# ==========================================

# Same 80/20 outer split used by traditional ML models
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Validation set comes only from training data
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=42,
    stratify=y_train_full
)

print("\n========== DATA SPLIT ==========")
print("Training Records:", len(X_train))
print("Validation Records:", len(X_val))
print("Testing Records:", len(X_test))

# ==========================================
# STEP 5: PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# Convert sparse matrices to dense arrays if needed
def to_dense_float32(data):
    if hasattr(data, "toarray"):
        data = data.toarray()
    return np.asarray(data, dtype=np.float32)

X_train_processed = to_dense_float32(X_train_processed)
X_val_processed = to_dense_float32(X_val_processed)
X_test_processed = to_dense_float32(X_test_processed)

print("Number of Processed Features:", X_train_processed.shape[1])

# ==========================================
# STEP 6: BUILD ANN
# ==========================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(X_train_processed.shape[1],)
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\n========== ANN ARCHITECTURE ==========")
model.summary()

# ==========================================
# STEP 7: TRAIN ANN
# ==========================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

print("\n========== ANN TRAINING ==========")

history = model.fit(
    X_train_processed,
    y_train.to_numpy(),
    validation_data=(
        X_val_processed,
        y_val.to_numpy()
    ),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# ==========================================
# STEP 8: TEST SET EVALUATION
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

print("\n========== ANN TEST RESULTS ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# ==========================================
# STEP 9: TRAINING HISTORY GRAPHS
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("ANN Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "visualizations/ann_training_loss.png",
    dpi=300
)

plt.close()

plt.figure(figsize=(9, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("ANN Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "visualizations/ann_training_accuracy.png",
    dpi=300
)

plt.close()

# ==========================================
# STEP 10: SAVE ANN MODEL
# ==========================================

model.save("models/ann_purchase_model.keras")

print("\nSaved ANN model:")
print("models/ann_purchase_model.keras")

print("\nSaved visualizations:")
print("visualizations/ann_training_loss.png")
print("visualizations/ann_training_accuracy.png")

print("\nANN TRAINING AND EVALUATION COMPLETED!")
