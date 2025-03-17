import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import pickle

# Generate synthetic data
def generate_data(num_rows):
    data = {
        "Weapon ID": [f"W{i+1:05}" for i in range(num_rows)],
        "Weapon Type": np.random.choice(["Shotgun", "Missile", "Cannon", "Drone", "Artillery", "Interceptor"], num_rows),
        "Weapon Speed (m/s)": np.random.randint(200, 2000, num_rows),
        "Target Distance (m)": np.random.randint(1000, 50000, num_rows),
        "Target Type": np.random.choice(["Vehicle", "Aircraft", "Infantry", "Bunker", "Missile"], num_rows),
    }
    df = pd.DataFrame(data)
    df["Assigned Time (s)"] = (df["Target Distance (m)"] / df["Weapon Speed (m/s)"]) + np.random.randint(10, 100, num_rows)
    df["Success"] = np.random.choice([0, 1], num_rows, p=[0.2, 0.8])  # 80% success rate
    return df

# Generate 10,000 rows of data
data = generate_data(10000)

# Feature Engineering
data = pd.get_dummies(data, columns=["Weapon Type", "Target Type"], drop_first=True)

# Features (X) and Target (y)
X = data.drop(columns=["Weapon ID", "Assigned Time (s)", "Success"])
y_time = data["Assigned Time (s)"]
y_success = data["Success"]

# Split the data into training and testing sets
X_train, X_test, y_time_train, y_time_test, y_success_train, y_success_test = train_test_split(
    X, y_time, y_success, test_size=0.2, random_state=42
)

# Train a model to predict Assigned Time (Regression)
time_model = RandomForestRegressor(random_state=42)
time_model.fit(X_train, y_time_train)

# Train a model to predict Success (Classification)
success_model = RandomForestClassifier(random_state=42)
success_model.fit(X_train, y_success_train)

# Evaluate the models
y_time_pred = time_model.predict(X_test)
y_success_pred = success_model.predict(X_test)

print("Assigned Time Model - Mean Squared Error:", mean_squared_error(y_time_test, y_time_pred))
print("Success Model - Accuracy:", accuracy_score(y_success_test, y_success_pred))

# Save the models as pickle files
with open("time_model.pkl", "wb") as f:
    pickle.dump(time_model, f)

with open("success_model.pkl", "wb") as f:
    pickle.dump(success_model, f)

print("Models saved as 'time_model.pkl' and 'success_model.pkl'")

# Visualization
# 1. Histogram for Weapon Speed and Target Distance
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(data["Weapon Speed (m/s)"], bins=30, color='blue', kde=True)
plt.title("Weapon Speed Distribution")
plt.xlabel("Weapon Speed (m/s)")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
sns.histplot(data["Target Distance (m)"], bins=30, color='green', kde=True)
plt.title("Target Distance Distribution")
plt.xlabel("Target Distance (m)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 2. Correlation Matrix
correlation_matrix = data.drop(columns=["Weapon ID", "Success"]).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# 3. Feature Importance Plot for both models
importances_time = time_model.feature_importances_
importances_success = success_model.feature_importances_

plt.figure(figsize=(10, 6))
sns.barplot(x=importances_time, y=X.columns)
plt.title("Feature Importance (Assigned Time Prediction)")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x=importances_success, y=X.columns)
plt.title("Feature Importance (Success Prediction)")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# 4. Predicted vs Actual for Regression Model
plt.figure(figsize=(8, 6))
plt.scatter(y_time_test, y_time_pred, alpha=0.5)
plt.plot([y_time_test.min(), y_time_test.max()], [y_time_test.min(), y_time_test.max()], color='red', linestyle='--')
plt.title("Predicted vs Actual (Assigned Time)")
plt.xlabel("Actual Assigned Time (s)")
plt.ylabel("Predicted Assigned Time (s)")
plt.show()

# 5. Confusion Matrix for Classification Model
conf_matrix = confusion_matrix(y_success_test, y_success_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', cbar=False, xticklabels=["Failure", "Success"], yticklabels=["Failure", "Success"])
plt.title("Confusion Matrix (Success Prediction)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
