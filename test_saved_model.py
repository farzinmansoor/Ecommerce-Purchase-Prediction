
# Test Saved Model

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the cleaned dataset
df = pd.read_csv("dataset/cleaned_online_shoppers.csv")

# Separate input features and target
X = df.drop("Revenue", axis=1)
y = df["Revenue"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load the saved model
model = joblib.load("models/best_purchase_model.pkl")

print("\nSaved model loaded successfully!")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate predictions
print("\n========== SAVED MODEL TEST RESULTS ==========")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict one sample
sample = X_test.iloc[[0]]

prediction = model.predict(sample)

print("\n========== SAMPLE PREDICTION ==========")

if prediction[0]:
    print("Prediction: Customer is likely to make a purchase.")
else:
    print("Prediction: Customer is unlikely to make a purchase.")

print("\nSAVED MODEL TEST COMPLETED!")
