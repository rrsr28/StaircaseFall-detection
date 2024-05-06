import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Step 1: Data Preprocessing
data = pd.read_csv("../data/keypoints_data.csv")

# Check for missing values
print(data.isnull().sum())

# Split features and target variable
X = data.drop("Label", axis=1)
y = data["Label"]

# Encode target variable (if categorical)
y = y.replace({"fall": 1, "not-fall": 0})

# Step 2: Feature Engineering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Model Selection and Training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 4: Model Evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

# Save the model
# with open('model.pkl', 'wb') as file:
#     pickle.dump(model, file)

# Load the model
# with open('model.pkl', 'rb') as file:
#     model = pickle.load(file)