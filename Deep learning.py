# %%
import pandas as pd
import numpy as np
import os
import joblib

# Scikit-Learn for preprocessing and evaluation metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# TensorFlow and Keras for building the Neural Network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def evaluate_and_print(y_true, y_pred, title):
    """Helper function to print out metrics clearly."""
    print(f"\n--- {title} Performance ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

print("Libraries loaded successfully!")

# %%
print("Loading data from 'heart.csv'...")
# Load the dataset
df = pd.read_csv("heart.csv")

# Basic cleaning
df = df.drop_duplicates()
df = df.dropna()

# Assume the target column is the last one in the CSV
target_col = df.columns[-1]
print(f"Using '{target_col}' as the target variable to predict.")

# Separate Features (X) and Target (y)
X = df.drop(columns=[target_col, 'id'], errors='ignore')
y = df[target_col]

# Split into 80% Training and 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data successfully split! Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# %%
print("Scaling features...")
# Neural networks perform terribly if features aren't scaled to a similar range.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler so the future web app scales user input the exact same way
if not os.path.exists("models"):
    os.makedirs("models")
joblib.dump(scaler, "models/scaler.pkl") 
print("Features scaled and scaler saved to 'models/scaler.pkl'.")

# %%
print("\nBuilding Deep Learning Architecture...")

# We build the network layer by layer
model = Sequential([
    # Input Layer & First Hidden Layer: 64 neurons, ReLU activation
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    
    # Dropout Layer: Randomly turns off 30% of neurons to prevent memorizing the data (overfitting)
    Dropout(0.3),
    
    # Second Hidden Layer: 32 neurons to distill patterns further
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    # Output Layer: 1 neuron because we want a single binary answer (1 for Risk, 0 for Healthy)
    # Sigmoid activation squishes the math into a probability between 0 and 1.
    Dense(1, activation='sigmoid')
])

# Compile the model rules
# Binary Crossentropy is the standard math for a Yes/No prediction
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

print("Architecture built successfully!")
model.summary()

# %%
print("\nTraining the Neural Network...")
# epochs=20: It will look at the entire dataset 20 times.
# batch_size=32: It updates its internal math after looking at 32 patients at a time.
history = model.fit(
    X_train_scaled, y_train, 
    epochs=20, 
    batch_size=32, 
    validation_split=0.2, # Sets aside 20% of training data to self-check during training
    verbose=1
)

# %%
print("\nEvaluating Neural Network on unseen Test Data...")

# The model outputs a probability (e.g., 0.85). We need to convert it to a hard 0 or 1.
# If probability is > 0.5, we predict 1 (At Risk).
y_pred_prob = model.predict(X_test_scaled)
y_pred_binary = (y_pred_prob > 0.5).astype(int)

# Print the report card
evaluate_and_print(y_test, y_pred_binary, "Deep Learning (Neural Network)")

# Save the trained model
model.save("models/deep_learning_model.h5")
print("\n✅ Deep Learning model saved to 'models/deep_learning_model.h5'.")