# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, silhouette_score

print("Libraries loaded successfully!")

# %%
print("Loading data from 'heart1.csv'...")
df = pd.read_csv("heart1.csv")

# Basic cleaning
df = df.drop_duplicates()
df = df.dropna()

# Assume the target column is the last one in the CSV
target_col = df.columns[-1]
print(f"Using '{target_col}' as the actual diagnosis (hidden from K-Means).")

# Separate Features (X) and Target (y)
# We drop 'target_col' and 'id' (if it exists) from X
X = df.drop(columns=[target_col, 'id'], errors='ignore')
y_true = df[target_col]

# %%
print("Scaling features...")
# K-Means groups data based on "distance". If we don't scale, large numbers 
# like Weight (e.g., 200) will overpower small numbers like Age (e.g., 50).
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features scaled successfully! Ready for clustering.")

# %%
print("\nRunning K-Means Clustering (Looking for 2 natural groups)...")

# Initialize K-Means. We ask for 2 clusters to see if it separates Healthy vs At-Risk.
# Note: n_init='auto' is the modern standard for K-Means in scikit-learn
kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')

# FIT THE MODEL: Notice we only pass 'X_scaled'. We DO NOT pass 'y_true'.
# The model is completely blind to who actually has heart disease.
cluster_labels = kmeans.fit_predict(X_scaled)

# %%
print("\n--- Clustering Analysis ---")

# Calculate Silhouette Score (How distinct are the clusters? Range: -1 to 1)
# Using a sample size if the dataset is huge to save time
sample_size = min(10000, X_scaled.shape[0])
sil_score = silhouette_score(X_scaled[:sample_size], cluster_labels[:sample_size])
print(f"Silhouette Score (Cluster Quality): {sil_score:.4f}")

# Map clusters to actual labels
# K-Means just calls the groups "0" and "1". We don't know if "0" means healthy or sick.
cm = confusion_matrix(y_true, cluster_labels)
accuracy = np.trace(cm) / np.sum(cm)

# If accuracy is below 50%, it means K-Means assigned 0 to Sick and 1 to Healthy. 
# We flip them just so the confusion matrix is easier to read.
if accuracy < 0.5:
    print("\nNote: K-Means assigned Cluster 0 to 'At Risk'. Flipping labels for readability...")
    cluster_labels = 1 - cluster_labels 
    cm = confusion_matrix(y_true, cluster_labels)
    accuracy = np.trace(cm) / np.sum(cm)
    
print(f"\nOverlap with actual medical diagnoses: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(cm)
print("\n(Note: We don't expect 90%+ accuracy here. K-Means is grouping by physical similarities, not targeted medical diagnosis!)")

# %%
print("\nPreparing 2D Visualization using PCA...")
# We have many patient features. PCA squishes them down to 2 dimensions (X and Y)
# so we can actually draw them on a flat screen.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for easy plotting
plot_df = pd.DataFrame({
    'PCA_X': X_pca[:, 0],
    'PCA_Y': X_pca[:, 1],
    'True_Label': y_true,
    'KMeans_Cluster': cluster_labels
})

# %%
# Create a side-by-side plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: The TRUE medical diagnoses
sns.scatterplot(data=plot_df, x='PCA_X', y='PCA_Y', hue='True_Label', 
                palette=['blue', 'red'], alpha=0.3, ax=axes[0])
axes[0].set_title('Reality (Actual Diagnoses: 0=Healthy, 1=Sick)')

# Plot 2: What K-Means thought the groups were
sns.scatterplot(data=plot_df, x='PCA_X', y='PCA_Y', hue='KMeans_Cluster', 
                palette=['blue', 'red'], alpha=0.3, ax=axes[1])
axes[1].set_title('K-Means Natural Clusters')

plt.tight_layout()
print("\nDisplaying Plot! Close the plot window to finish the script.")
plt.show()