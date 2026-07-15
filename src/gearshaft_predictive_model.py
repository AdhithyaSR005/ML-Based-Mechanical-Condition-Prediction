import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

# Load dataset
data = pd.read_csv("gearshaftdataset.csv")

print("Dataset Preview:")
print(data.head())

# Features
X = data[['TORQUE', 'TOT_DEF', 'DIRECT_DEF', 'EQV_STR', 'EQV_STRESS', 'STR_ENERGY']]

# Target
y = data['CONDITION']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("\nModel Accuracy:")
print(accuracy_score(y_test, y_pred))

# Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "gearshaft_model.pkl")

print("\nModel saved successfully!")

# Graph 1
plt.figure(figsize=(8,5))
plt.plot(data['TORQUE'], data['EQV_STRESS'], marker='o')

plt.xlabel("Torque (Nm)")
plt.ylabel("Equivalent Stress (Pa)")
plt.title("Torque vs Equivalent Stress")

plt.grid(True)
plt.show()

# Graph 2
plt.figure(figsize=(8,5))
plt.plot(data['TORQUE'], data['TOT_DEF'], marker='o')

plt.xlabel("Torque (Nm)")
plt.ylabel("Total Deformation (m)")
plt.title("Torque vs Total Deformation")

plt.grid(True)
plt.show()

# Example Prediction
sample = [[200, 2.35E-04, 6.01E-07, 8.34E-03, 1.60E+09, 2.79E-02]]

prediction = model.predict(sample)

print("\nPredicted Condition:")
print(prediction[0])