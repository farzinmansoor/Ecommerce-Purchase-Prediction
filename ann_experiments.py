
# ==========================================
# ANN HYPERPARAMETER EXPERIMENTS
# E-COMMERCE PURCHASE PREDICTION
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
    f1_score
)

SEED = 42

os.makedirs("visualizations", exist_ok=True)
os.makedirs("models", exist_ok=True)

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
# TRAIN / VALIDATION / TEST SPLIT
# ==========================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=SEED,
    stratify=y_train_full
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_columns),
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_columns
    )
])

X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)

def to_dense(data):
    if hasattr(data, "toarray"):
        data = data.toarray()
    return np.asarray(data, dtype=np.float32)

X_train = to_dense(X_train)
X_val = to_dense(X_val)

y_train = y_train.to_numpy()
y_val = y_val.to_numpy()

print("\nDataset:", df.shape)
print("Training Records:", len(X_train))
print("Validation Records:", len(X_val))
print("Test Records Reserved:", len(X_test))
print("Processed Features:", X_train.shape[1])

# ==========================================
# ANN CONFIGURATIONS
# ==========================================

experiments = [
    {
        "name": "ANN 1",
        "layers": [32],
        "batch_size": 32,
        "optimizer": "adam",
        "epochs": 40
    },
    {
        "name": "ANN 2",
        "layers": [64, 32],
        "batch_size": 32,
        "optimizer": "adam",
        "epochs": 40
    },
    {
        "name": "ANN 3",
        "layers": [128, 64, 32],
        "batch_size": 64,
        "optimizer": "adam",
        "epochs": 40
    },
    {
        "name": "ANN 4",
        "layers": [64, 32],
        "batch_size": 64,
        "optimizer": "rmsprop",
        "epochs": 40
    }
]

# ==========================================
# BUILD ANN
# ==========================================

def build_model(layers, optimizer, input_size):
    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Input(shape=(input_size,))
    )

    for neurons in layers:
        model.add(
            tf.keras.layers.Dense(
                neurons,
                activation="relu"
            )
        )
        model.add(tf.keras.layers.Dropout(0.2))

    model.add(
        tf.keras.layers.Dense(1, activation="sigmoid")
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ==========================================
# RUN EXPERIMENTS
# ==========================================

results = []
trained_models = {}

for config in experiments:

    print("\n===================================")
    print("Training:", config["name"])
    print("Layers:", config["layers"])
    print("Batch Size:", config["batch_size"])
    print("Optimizer:", config["optimizer"])
    print("===================================")

    tf.keras.backend.clear_session()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    model = build_model(
        config["layers"],
        config["optimizer"],
        X_train.shape[1]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=[early_stopping],
        verbose=0
    )

    probabilities = model.predict(
        X_val,
        verbose=0
    ).ravel()

    predictions = (probabilities >= 0.5).astype(int)

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    result = {
        "Model": config["name"],
        "Layers": str(config["layers"]),
        "Batch Size": config["batch_size"],
        "Optimizer": config["optimizer"],
        "Max Epochs": config["epochs"],
        "Epochs Trained": len(history.history["loss"]),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    results.append(result)
    trained_models[config["name"]] = model

    print("Epochs Trained:", result["Epochs Trained"])
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation Recall: {recall:.4f}")
    print(f"Validation F1 Score: {f1:.4f}")

# ==========================================
# COMPARE EXPERIMENTS
# ==========================================

results_df = pd.DataFrame(results)

print("\n========== ANN EXPERIMENT RESULTS ==========")
print(results_df.to_string(index=False))

results_df.to_csv(
    "models/ann_experiment_results.csv",
    index=False
)

# ==========================================
# VISUALIZE VALIDATION F1
# ==========================================

plt.figure(figsize=(9, 5))

plt.bar(
    results_df["Model"],
    results_df["F1 Score"]
)

plt.title("ANN Experiments - Validation F1 Score")
plt.xlabel("ANN Configuration")
plt.ylabel("Validation F1 Score")
plt.ylim(0, 1)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "visualizations/ann_experiment_comparison.png",
    dpi=300
)

plt.close()

# ==========================================
# SAVE BEST EXPERIMENTAL MODEL
# ==========================================

best_row = results_df.loc[
    results_df["F1 Score"].idxmax()
]

best_name = best_row["Model"]

best_model = trained_models[best_name]

best_model.save(
    "models/ann_best_experiment.keras"
)

print("\n========== BEST ANN EXPERIMENT ==========")
print("Model:", best_name)
print("Layers:", best_row["Layers"])
print("Optimizer:", best_row["Optimizer"])
print("Validation F1:", round(best_row["F1 Score"], 4))

print("\nSaved:")
print("models/ann_experiment_results.csv")
print("models/ann_best_experiment.keras")
print("visualizations/ann_experiment_comparison.png")

print("\nANN EXPERIMENTS COMPLETED!")
