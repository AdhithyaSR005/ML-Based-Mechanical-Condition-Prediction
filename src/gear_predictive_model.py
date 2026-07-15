import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("geardataset.csv")

print("Dataset Preview:")
print(data.head())

# Features
X = data[['FORCE', 'TOT_DEF', 'DIRECT_DEF', 'EQV_STR', 'EQV_STRESS', 'STR_ENERGY']]

# Target
y = data['CONDITION']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("\nModel Accuracy:")
print(accuracy_score(y_test, y_pred))

# Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Example prediction
sample = [[300, 3.13E-06, 2.59E-07, 8.36E-05, 1.50E+07, 1.09E-05]]

prediction = model.predict(sample)

print("\nPredicted Condition:")
print(prediction[0])

import matplotlib.pyplot as plt

# Stress graph
plt.figure(figsize=(8,5))
plt.plot(data.iloc[:,0], data['EQV_STRESS'], marker='o')

plt.xlabel("Load")
plt.ylabel("Equivalent Stress")
plt.title("Load vs Equivalent Stress")

plt.grid(True)
plt.show()

# Deformation graph
plt.figure(figsize=(8,5))
plt.plot(data.iloc[:,0], data['TOT_DEF'], marker='o')

plt.xlabel("Load")
plt.ylabel("Total Deformation")

plt.title("Load vs Total Deformation")

plt.grid(True)
plt.show()


import joblib

joblib.dump(model, "saved_model.pkl")

print("Model saved successfully!")

import joblib

joblib.dump(model, "gear_model.pkl")

print("Gear model saved!")
