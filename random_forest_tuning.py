
# ==========================================
# RANDOM FOREST HYPERPARAMETER TUNING
# AND OVERFITTING / UNDERFITTING ANALYSIS
# ==========================================

import os

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/cleaned_online_shoppers.csv")

X = df.drop("Revenue", axis=1)
y = df["Revenue"]

print("\n========== DATASET INFORMATION ==========")
print("Dataset Shape:", df.shape)

# ==========================================
# STEP 2: IDENTIFY FEATURES
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
# STEP 3: TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# ==========================================
# STEP 4: PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_columns
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)

# ==========================================
# STEP 5: RANDOM FOREST PIPELINE
# ==========================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

# ==========================================
# STEP 6: HYPERPARAMETER SEARCH SPACE
# ==========================================

parameters = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4]
}

# ==========================================
# STEP 7: RANDOMIZED SEARCH CV
# ==========================================

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=parameters,
    n_iter=8,
    cv=3,
    scoring="f1",
    random_state=42,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

print("\nStarting Random Forest Hyperparameter Tuning...")

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

# ==========================================
# STEP 8: TUNING RESULTS
# ==========================================

print("\n========== TUNING RESULTS ==========")

print("\nBest Parameters:")
print(random_search.best_params_)

print("\nBest Cross-Validation F1 Score:")
print(random_search.best_score_)

# ==========================================
# STEP 9: TEST SET EVALUATION
# ==========================================

y_pred = best_model.predict(X_test)

print("\n========== TUNED MODEL PERFORMANCE ==========")

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# STEP 10: OVERFITTING / UNDERFITTING
# ==========================================

print("\n========== OVERFITTING ANALYSIS ==========")

train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

train_precision = precision_score(
    y_train,
    train_predictions,
    zero_division=0
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

train_recall = recall_score(
    y_train,
    train_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

train_f1 = f1_score(
    y_train,
    train_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Training": [
        train_accuracy,
        train_precision,
        train_recall,
        train_f1
    ],
    "Testing": [
        test_accuracy,
        test_precision,
        test_recall,
        test_f1
    ]
})

print("\nTraining vs Testing Performance:")
print(results.to_string(index=False))

accuracy_gap = train_accuracy - test_accuracy
f1_gap = train_f1 - test_f1

print("\nAccuracy Gap:", round(accuracy_gap, 4))
print("F1 Score Gap:", round(f1_gap, 4))

# ==========================================
# STEP 11: CROSS-VALIDATION DIAGNOSTICS
# ==========================================

best_index = random_search.best_index_

cv_train_f1 = random_search.cv_results_[
    "mean_train_score"
][best_index]

cv_validation_f1 = random_search.cv_results_[
    "mean_test_score"
][best_index]

cv_f1_gap = cv_train_f1 - cv_validation_f1

print("\n========== CROSS-VALIDATION ANALYSIS ==========")

print("Mean CV Training F1:", round(cv_train_f1, 4))
print("Mean CV Validation F1:", round(cv_validation_f1, 4))
print("CV F1 Gap:", round(cv_f1_gap, 4))

print("\nInterpretation:")

if cv_f1_gap > 0.10:
    print(
        "The model shows a substantial training-validation "
        "performance gap, suggesting possible overfitting."
    )
elif cv_f1_gap > 0.05:
    print(
        "The model shows a moderate training-validation "
        "performance gap. Some overfitting may be present."
    )
else:
    print(
        "The training-validation F1 gap is relatively small. "
        "There is limited evidence of overfitting from this gap."
    )

print(
    "\nNote: These thresholds are illustrative. "
    "Underfitting cannot be determined from the gap alone; "
    "absolute performance and a simple baseline also matter."
)

# ==========================================
# STEP 12: SAVE PERFORMANCE VISUALIZATION
# ==========================================

os.makedirs("visualizations", exist_ok=True)

metrics = results["Metric"].tolist()

training_scores = results["Training"].to_numpy()
testing_scores = results["Testing"].to_numpy()

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    x - width / 2,
    training_scores,
    width,
    label="Training",
    color="steelblue"
)

plt.bar(
    x + width / 2,
    testing_scores,
    width,
    label="Testing",
    color="salmon"
)

plt.xticks(x, metrics)
plt.ylabel("Score")
plt.ylim(0, 1.05)

plt.title(
    "Random Forest: Training vs Testing Performance"
)

plt.legend()
plt.grid(axis="y", alpha=0.25)

plt.tight_layout()

plt.savefig(
    "visualizations/random_forest_overfitting_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSaved visualization: "
    "visualizations/random_forest_overfitting_analysis.png"
)

print("\nOVERFITTING ANALYSIS COMPLETED!")
print("\nRANDOM FOREST TUNING COMPLETED!")