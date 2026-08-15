# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Libraries loaded successfully!")

# %%
print("--- STEP 1: LOADING RAW DATA ---")
filepath = "heart.csv"
df = pd.read_csv(filepath)

print(f"Original dataset size: {df.shape[0]} rows and {df.shape[1]} columns.")
print("\nFirst 5 rows:")
print(df.head())

# %%
print("\n--- STEP 2: DATA CLEANING ---")
# Check for missing values and duplicates
missing_values = df.isnull().sum().sum()
duplicate_count = df.duplicated().sum()

print(f"Found {missing_values} missing values and {duplicate_count} duplicate rows.")

# Drop duplicates and missing values permanently for our clean dataset
print(f"Dropping {duplicate_count} duplicates...")
df_clean = df.drop_duplicates()
df_clean = df_clean.dropna()

print(f"Cleaned dataset size: {df_clean.shape[0]} rows and {df_clean.shape[1]} columns.")
print(f"Total rows removed: {df.shape[0] - df_clean.shape[0]}")

# Save to the brand new CSV file!
clean_filepath = "heart1.csv"
df_clean.to_csv(clean_filepath, index=False)
print(f"\n✅ SUCCESS: Cleaned data saved as '{clean_filepath}'!")
print("We will use 'heart1.csv' for all our model training going forward.")

# %%
print("\n--- STEP 3: VISUALIZING CLEAN DATA ---")
sns.set_theme(style="whitegrid")

# Assume target is the last column
target_col = df_clean.columns[-1]
print(f"Using '{target_col}' as the target variable for plots.")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Check if the classes are balanced
sns.countplot(x=target_col, data=df_clean, ax=axes[0], palette="Set2")
axes[0].set_title(f'Distribution of Target Variable ({target_col})')

# Plot 2: Correlation Matrix to see which features link to heart disease
numeric_df = df_clean.select_dtypes(include=['float64', 'int64'])
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=axes[1], vmin=-1, vmax=1)
axes[1].set_title('Feature Correlation Matrix')

plt.tight_layout()
print("Displaying visualizations! Close the plot window to finish the script.")
plt.show()