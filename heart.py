import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_explore_data(filepath="heart.csv"):
    """
    Loads the NEW cardiovascular dataset (CSV format) and performs initial exploratory data analysis.
    """
    print(f"Loading '{filepath}'...")
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully!\n")
    except Exception as e:
        print(f"An error occurred while loading the file. Make sure the path is correct: {e}")
        return None

    print("-" * 30)
    print("DATASET OVERVIEW")
    print("-" * 30)
    
    # Display the first few rows to understand the new structure and column names
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    # Display data types
    print("\nDataset Information (Pay attention to the column names!):")
    df.info()

    # Calculate basic statistical details
    print("\nDescriptive Statistics:")
    print(df.describe())

    print("\n" + "-" * 30)
    print("DATA QUALITY CHECKS")
    print("-" * 30)
    
    missing_values = df.isnull().sum()
    print("\nMissing values in each column:")
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found.")

    duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicates}")

    return df

def visualize_data(df):
    """
    Generates basic visualizations to understand data distribution and correlations.
    """
    if df is None:
        return

    sns.set_theme(style="whitegrid")
    
    # We will assume the target column is the very last column in the CSV file
    target_col = df.columns[-1]
    print(f"\nUsing '{target_col}' as the target variable for plots.")

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Check if the classes are balanced
    sns.countplot(x=target_col, data=df, ax=axes[0], palette="Set2")
    axes[0].set_title(f'Distribution of Target Variable ({target_col})')
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel('Count')

    # Plot 2: Correlation Matrix
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_df.corr()
    
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=axes[1], vmin=-1, vmax=1)
    axes[1].set_title('Feature Correlation Matrix')

    plt.tight_layout()
    print("\nDisplaying visualizations...")
    plt.show()

if __name__ == "__main__":
    # Ensure heart.csv is saved in the exact same folder as this Python script
    dataset = load_and_explore_data("heart.csv") 
    visualize_data(dataset)
