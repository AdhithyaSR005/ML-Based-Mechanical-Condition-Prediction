import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib


# LOAD DATASET
data = pd.read_csv("bearingdataset - mydataset.csv.csv")

# REMOVE EXTRA SPACES FROM COLUMN NAMES
data.columns = data.columns.str.strip()

# SHOW COLUMN NAMES
print("\nColumns in Dataset:\n")
print(data.columns)

# SHOW DATA
print("\nDataset Preview:\n")
print(data.head())

# FEATURES
X = data[[
    'FORCE',
    'TOT_DEF',
    'DIRECT_DEF',
    'EQV_STR',
    'EQV_STRESS',
    'STR_ENERGY'
]]

# TARGET
y = data['CONDITION']

# SPLIT DATASET
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# CREATE MODEL
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# TRAIN MODEL
model.fit(X_train, y_train)

# PREDICT
y_pred = model.predict(X_test)

# ACCURACY
print("\nModel Accuracy:\n")
print(accuracy_score(y_test, y_pred))

# REPORT
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# SAVE MODEL
joblib.dump(model, "bearing_model.pkl")

print("\nBearing model saved successfully!")

# GRAPH 1
plt.figure(figsize=(8,5))
plt.plot(data['FORCE'], data['EQV_STRESS'], marker='o')

plt.xlabel("Force (N)")
plt.ylabel("Equivalent Stress (Pa)")
plt.title("Force vs Equivalent Stress")

plt.grid(True)
plt.show()

# GRAPH 2
plt.figure(figsize=(8,5))
plt.plot(data['FORCE'], data['TOT_DEF'], marker='o')

plt.xlabel("Force (N)")
plt.ylabel("Total Deformation (m)")
plt.title("Force vs Total Deformation")

plt.grid(True)
plt.show()

# SAMPLE PREDICTION
sample = [[
    600,
    8.41E-07,
    6.29E-09,
    1.03E-04,
    1.56E+07,
    1.69E-06
]]

prediction = model.predict(sample)

print("\nPredicted Condition:")
print(prediction[0])

# SAVE MODEL AS PKL
joblib.dump(model, "bearing_model.pkl")

print("\nBearing model saved successfully!")