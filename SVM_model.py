import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_and_print(y_true, y_pred, title):
    """Helper function to print out metrics clearly for comparison."""
    print(f"\n--- {title} Performance ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

# %%
print("Loading data from 'heart.csv'...")
# 1. Load the data directly in this script
df = pd.read_csv("heart.csv")

# Basic cleaning
df = df.drop_duplicates()
df = df.dropna()

# Assume the target column is the last one in the CSV
target_col = df.columns[-1]
print(f"Using '{target_col}' as the target variable to predict.")

# 2. Separate Features (X) and Target (y)
X = df.drop(columns=[target_col, 'id'], errors='ignore')
y = df[target_col]

# 3. Split into 80% Training and 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data successfully split into training and testing sets!")

# %%
# 4. Scale the Features
# SVMs use distance calculations under the hood, so scaling is MANDATORY.
print("Scaling features (Crucial for SVM)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler so the future web app scales user input the exact same way
if not os.path.exists("models"):
    os.makedirs("models")
joblib.dump(scaler, "models/scaler.pkl") 
print("Scaler saved to 'models/scaler.pkl'.")

# %%
# 5. Train Standard SVM
print(f"\nTraining Standard Support Vector Machine (SVM)...")
# Note: SVM can take a long time on large datasets. We use kernel='rbf' (Radial Basis Function)
svm_standard = SVC(kernel='rbf', random_state=42)
svm_standard.fit(X_train_scaled, y_train)

# Make predictions and evaluate
y_pred_std = svm_standard.predict(X_test_scaled)
evaluate_and_print(y_test, y_pred_std, "Standard SVM")

# %%
# 6. Train Weighted SVM (Focuses more on finding sick patients)
print(f"\nTraining Weighted Support Vector Machine...")
# class_weight='balanced' penalizes the model heavily if it misses an at-risk patient
svm_weighted = SVC(kernel='rbf', class_weight='balanced', random_state=42)
svm_weighted.fit(X_train_scaled, y_train)

# Make predictions and evaluate
y_pred_weighted = svm_weighted.predict(X_test_scaled)
evaluate_and_print(y_test, y_pred_weighted, "Weighted SVM (Balanced)")

# %%
# 7. Save the trained models for deployment later
print("\nSaving SVM models...")
joblib.dump(svm_standard, "models/svm_model_standard.pkl")
joblib.dump(svm_weighted, "models/svm_model_weighted.pkl")

print("\n--- Comparison Guide ---")
print("Look closely at the 'Recall' metric between the Standard and Weighted models.")
print("In medical predictions, Recall is often more important because it measures how")
print("many of the actual 'at-risk' patients we successfully caught.")
print("\n✅ Both SVM models trained and saved inside the 'models/' folder.")