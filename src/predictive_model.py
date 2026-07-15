import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("mydataset.csv")
# Show dataset preview
print("Dataset Preview:")
print(data.head())

# Input features
X = data[[
    'TORQUE',
    'TOT_DEF',
    'DIRECT_DEF',
    'EQV_STR',
    'EQV_STRESS',
    'STR_ENERGY'
]]

# Output labels
y = data['CONDITION']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("\nModel Accuracy:")
print(accuracy_score(y_test, y_pred))

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Example prediction
sample = [[
    280,
    0.00060,
    2.4e-06,
    0.00065,
    140000000,
    0.006
]]

prediction = model.predict(sample)

print("\nPredicted Condition:")
print(prediction[0])

# Plot
plt.figure(figsize=(8,5))

plt.plot(
    data['TORQUE'],
    data['EQV_STRESS'],
    marker='o'
)

plt.xlabel("Torque (Nm)")
plt.ylabel("Equivalent Stress (Pa)")
plt.title("Torque vs Equivalent Stress")

plt.grid(True)

plt.show()

import joblib

joblib.dump(model, "crankshaft_model.pkl")

print("Crankshaft model saved!")