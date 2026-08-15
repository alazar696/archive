# %%
import pandas as pd
import numpy as np
import os
import joblib

# Scikit-Learn modules
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def print_metrics(y_true, y_pred, title):
    print(f"\n--- {title} ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")

print("Libraries loaded successfully!")

# %%
print("Loading and preparing data from 'heart1.csv'...")
df = pd.read_csv("heart1.csv")
df = df.drop_duplicates().dropna()

target_col = df.columns[-1]
X = df.drop(columns=[target_col, 'id'], errors='ignore')
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data ready!")

# %%
print("\nStep 1: Training BASELINE Random Forest (Default Settings)...")
# This is the standard model with no special adjustments
rf_baseline = RandomForestClassifier(random_state=42)
rf_baseline.fit(X_train_scaled, y_train)

y_pred_baseline = rf_baseline.predict(X_test_scaled)
print_metrics(y_test, y_pred_baseline, "BASELINE MODEL")

# %%
print("\nStep 2: Setting up Grid Search for FINE-TUNING...")
# We give Python a "grid" of options to test
param_grid = {
    'n_estimators': [50, 100, 200],      # Number of trees
    'max_depth': [None, 5, 10, 20],      # Maximum depth of the trees
    'min_samples_split': [2, 5, 10]      # Minimum samples required to split an internal node
}

# Setup the Grid Search
rf_tuner = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_tuner, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)

print("Grid Search is ready to run!")

# %%
print("\nStep 3: Running Grid Search (Testing 36 different combinations!)...")
# This might take a minute or two as it builds hundreds of models in the background
grid_search.fit(X_train_scaled, y_train)

print(f"\n✅ Grid Search Complete!")
print(f"The absolute BEST settings found were: {grid_search.best_params_}")

# %%
print("\nStep 4: Evaluating the TUNED Model...")
# We grab the best version of the model that Grid Search found
best_rf_model = grid_search.best_estimator_

# Make predictions with the optimized model
y_pred_tuned = best_rf_model.predict(X_test_scaled)

print_metrics(y_test, y_pred_tuned, "FINE-TUNED MODEL")

# Save the supercharged model!
joblib.dump(best_rf_model, "models/tuned_random_forest.pkl")
print("\n✅ Tuned model successfully saved as 'tuned_random_forest.pkl'!")