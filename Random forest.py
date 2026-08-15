import pandas as pd
import numpy as np
import os
import joblib

# Scikit-Learn modules for preprocessing and modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Calculates and prints performance metrics for a classification model.
    """
    print(f"\n--- {model_name} Performance ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

def load_and_preprocess_data(filepath="heart1.csv"):
    """
    Loads data from CSV, cleans it, and splits it into training and testing sets.
    """
    if not os.path.exists(filepath):
        print(f"Error: '{filepath}' not found. Make sure it's in the same folder as this script.")
        return None, None, None, None, None

    print(f"Loading data from '{filepath}'...")
    df = pd.read_csv(filepath)
    
    # Basic cleaning: Drop duplicates and missing values
    df = df.drop_duplicates()
    df = df.dropna()
    
    # We will assume the target column (healthy/sick) is the last column in your CSV
    target_col = df.columns[-1]
    print(f"Using '{target_col}' as the target variable to predict.")

    # Separate features (X) and target (y)
    X = df.drop(columns=[target_col, 'id'], errors='ignore') # 'id' is dropped if it exists
    y = df[target_col]

    # Split the dataset: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize the features (Crucial for Machine Learning models)
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for later use in our web app interface
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(scaler, "models/scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Trains and evaluates the Random Forest model.
    """
    print("\nTraining Random Forest Classifier...")
    # Initialize the model with 100 trees
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    # Make predictions on the test data
    rf_pred = rf_model.predict(X_test)
    
    # Grade the model's performance
    evaluate_model(y_test, rf_pred, "Random Forest")
    
    # Save the trained model to a file
    joblib.dump(rf_model, "models/random_forest.pkl")
    print("\n✅ Random Forest model trained and saved to 'models/random_forest.pkl'.")

if __name__ == "__main__":
    print("Starting Model Training Phase...")
    
    # 1. Load and prepare the data
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data("heart1.csv")
    
    if X_train is not None:
        # 2. Train and evaluate the model
        train_random_forest(X_train, X_test, y_train, y_test)