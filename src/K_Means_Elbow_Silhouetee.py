# ============================================================
# PLACEMENT PREDICTION - PREPROCESSED DATASET
# K-MEANS AND K-MEANS++ CLUSTERING
#
# Features:
# CGPA, HistoryOfBacklogs, Internships, AptitudeTestScore
#
# Methods:
# 1. K-Means
# 2. K-Means++
# 3. Elbow Method
# 4. Silhouette Score
# 5. PCA visualization
# ============================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    accuracy_score
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. INPUT FILE & OUTPUT FOLDERS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "dataset", "final_preprocess_M2.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "K_Means_K++Means_Elbow_Silhoute_Outputs")

KMEANS_FOLDER = os.path.join(OUTPUT_FOLDER, "KMeans")
KMEANS_PP_FOLDER = os.path.join(OUTPUT_FOLDER, "KMeansPlusPlus")
ELBOW_FOLDER = os.path.join(OUTPUT_FOLDER, "Elbow")
SILHOUETTE_FOLDER = os.path.join(OUTPUT_FOLDER, "Silhouette")
ACCURACY_FOLDER = os.path.join(OUTPUT_FOLDER, "Accuracy")

for folder in [KMEANS_FOLDER, KMEANS_PP_FOLDER, ELBOW_FOLDER, SILHOUETTE_FOLDER, ACCURACY_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ============================================================
# 2. LOAD DATASET
# ============================================================
print("=" * 75)
print("K-MEANS AND K-MEANS++")
print("PLACEMENT PREDICTION - PREPROCESSED DATASET")
print("=" * 75)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"\nPreprocessed dataset was not found.\nCheck this path:\n{INPUT_FILE}")

data = pd.read_csv(INPUT_FILE)
print("\nPreprocessed dataset loaded successfully.")
print("Rows :", data.shape[0])
print("Columns :", data.shape[1])

# Column Normalization
def normalize_column_name(column):
    return (
        str(column)
        .strip()
        .lower()
        .replace("_", "")
        .replace(" ", "")
        .replace("-", "")
    )

normalized_columns = {
    normalize_column_name(column): column
    for column in data.columns
}

required_features = ["CGPA", "HistoryOfBacklogs", "Internships", "AptitudeTestScore"]
feature_columns = []

for feature in required_features:
    normalized_feature = normalize_column_name(feature)
    if normalized_feature in normalized_columns:
        feature_columns.append(normalized_columns[normalized_feature])
    else:
        alternatives = ["AptitudeTestScore", "Aptitude_Test_Score", "AptitudeScore", "AptituteScore", "AptituteTestScore"]
        found = None
        for alt in alternatives:
            norm_alt = normalize_column_name(alt)
            if norm_alt in normalized_columns:
                found = normalized_columns[norm_alt]
                break
        if found is not None:
            feature_columns.append(found)
        else:
            raise ValueError(f"\nRequired column '{feature}' was not found.")

print("\n" + "=" * 75)
print("FEATURES USED FOR CLUSTERING")
print("=" * 75)
for feature in feature_columns:
    print("✓", feature)

X = data[feature_columns].copy()
for column in feature_columns:
    X[column] = pd.to_numeric(X[column], errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)
for column in feature_columns:
    X[column] = X[column].fillna(X[column].median())

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nFeature standardization completed.")

# ============================================================
# K RANGE & CLUSTERING ANALYSIS
# ============================================================
K_VALUES = range(2, 11)
print("\n" + "=" * 75)
print("K-MEANS ANALYSIS")
print("=" * 75)

kmeans_inertia = []
kmeans_silhouette = []

for k in K_VALUES:
    model = KMeans(n_clusters=k, init="random", n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)
    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, labels)
    kmeans_inertia.append(inertia)
    kmeans_silhouette.append(silhouette)
    print(f"K = {k:2d} | Inertia = {inertia:.4f} | Silhouette = {silhouette:.4f}")

print("\n" + "=" * 75)
print("K-MEANS++ ANALYSIS")
print("=" * 75)

kmeans_pp_inertia = []
kmeans_pp_silhouette = []

for k in K_VALUES:
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)
    inertia = model.inertia_
    silhouette = silhouette_score(X_scaled, labels)
    kmeans_pp_inertia.append(inertia)
    kmeans_pp_silhouette.append(silhouette)
    print(f"K = {k:2d} | Inertia = {inertia:.4f} | Silhouette = {silhouette:.4f}")

best_k_kmeans = list(K_VALUES)[np.argmax(kmeans_silhouette)]
best_k_kmeans_pp = list(K_VALUES)[np.argmax(kmeans_pp_silhouette)]

print("\n" + "=" * 75)
print("SELECTED NUMBER OF CLUSTERS")
print("=" * 75)
print("K-Means best K   :", best_k_kmeans)
print("K-Means++ best K :", best_k_kmeans_pp)

# ============================================================
# ELBOW & SILHOUETTE GRAPHS
# ============================================================
# Elbow K-Means
plt.figure(figsize=(10, 6))
plt.plot(list(K_VALUES), kmeans_inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method - K-Means")
plt.xticks(list(K_VALUES))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(ELBOW_FOLDER, "elbow_kmeans.png"), dpi=300, bbox_inches="tight")
plt.close()

# Elbow K-Means++
plt.figure(figsize=(10, 6))
plt.plot(list(K_VALUES), kmeans_pp_inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method - K-Means++")
plt.xticks(list(K_VALUES))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(ELBOW_FOLDER, "elbow_kmeans_plus_plus.png"), dpi=300, bbox_inches="tight")
plt.close()

# Silhouette K-Means
plt.figure(figsize=(10, 6))
plt.plot(list(K_VALUES), kmeans_silhouette, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score - K-Means")
plt.xticks(list(K_VALUES))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SILHOUETTE_FOLDER, "silhouette_kmeans.png"), dpi=300, bbox_inches="tight")
plt.close()

# Silhouette K-Means++
plt.figure(figsize=(10, 6))
plt.plot(list(K_VALUES), kmeans_pp_silhouette, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score - K-Means++")
plt.xticks(list(K_VALUES))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SILHOUETTE_FOLDER, "silhouette_kmeans_plus_plus.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# FINAL MODELS & DATA SAVING
# ============================================================
kmeans_model = KMeans(n_clusters=best_k_kmeans, init="random", n_init=10, random_state=42)
kmeans_labels = kmeans_model.fit_predict(X_scaled)

kmeans_pp_model = KMeans(n_clusters=best_k_kmeans_pp, init="k-means++", n_init=10, random_state=42)
kmeans_pp_labels = kmeans_pp_model.fit_predict(X_scaled)

kmeans_result = X.copy()
kmeans_result["KMeans_Cluster"] = kmeans_labels + 1
kmeans_result.to_csv(os.path.join(KMEANS_FOLDER, "kmeans_clustered_data.csv"), index=False)

kmeans_pp_result = X.copy()
kmeans_pp_result["KMeansPlusPlus_Cluster"] = kmeans_pp_labels + 1
kmeans_pp_result.to_csv(os.path.join(KMEANS_PP_FOLDER, "kmeans_plus_plus_clustered_data.csv"), index=False)

# Metrics calculation
km_silhouette = silhouette_score(X_scaled, kmeans_labels)
km_calinski = calinski_harabasz_score(X_scaled, kmeans_labels)
km_davies = davies_bouldin_score(X_scaled, kmeans_labels)

pp_silhouette = silhouette_score(X_scaled, kmeans_pp_labels)
pp_calinski = calinski_harabasz_score(X_scaled, kmeans_pp_labels)
pp_davies = davies_bouldin_score(X_scaled, kmeans_pp_labels)

kmeans_metrics = pd.DataFrame({
    "Method": ["K-Means"],
    "Number_of_Clusters": [best_k_kmeans],
    "Inertia": [kmeans_model.inertia_],
    "Silhouette_Score": [km_silhouette],
    "Calinski_Harabasz_Score": [km_calinski],
    "Davies_Bouldin_Score": [km_davies]
})
kmeans_metrics.to_csv(os.path.join(KMEANS_FOLDER, "kmeans_metrics.csv"), index=False)

kmeans_pp_metrics = pd.DataFrame({
    "Method": ["K-Means++"],
    "Number_of_Clusters": [best_k_kmeans_pp],
    "Inertia": [kmeans_pp_model.inertia_],
    "Silhouette_Score": [pp_silhouette],
    "Calinski_Harabasz_Score": [pp_calinski],
    "Davies_Bouldin_Score": [pp_davies]
})
kmeans_pp_metrics.to_csv(os.path.join(KMEANS_PP_FOLDER, "kmeans_plus_plus_metrics.csv"), index=False)

# 3D Clustering Graphs
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(X[feature_columns[0]], X[feature_columns[1]], X[feature_columns[2]], c=kmeans_labels, s=20, alpha=0.7)
ax.set_xlabel(feature_columns[0])
ax.set_ylabel(feature_columns[1])
ax.set_zlabel(feature_columns[2])
ax.set_title("K-Means Clustering")
fig.colorbar(scatter, ax=ax, label="Cluster")
plt.tight_layout()
plt.savefig(os.path.join(KMEANS_FOLDER, "kmeans_3D_clustering.png"), dpi=300, bbox_inches="tight")
plt.close()

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(X[feature_columns[0]], X[feature_columns[1]], X[feature_columns[2]], c=kmeans_pp_labels, s=20, alpha=0.7)
ax.set_xlabel(feature_columns[0])
ax.set_ylabel(feature_columns[1])
ax.set_zlabel(feature_columns[2])
ax.set_title("K-Means++ Clustering")
fig.colorbar(scatter, ax=ax, label="Cluster")
plt.tight_layout()
plt.savefig(os.path.join(KMEANS_PP_FOLDER, "kmeans_plus_plus_3D_clustering.png"), dpi=300, bbox_inches="tight")
plt.close()

# PCA 2-D Visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, s=20, alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clusters - PCA Visualization")
plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(KMEANS_FOLDER, "kmeans_PCA_clustering.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_pp_labels, s=20, alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means++ Clusters - PCA Visualization")
plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(KMEANS_PP_FOLDER, "kmeans_plus_plus_PCA_clustering.png"), dpi=300, bbox_inches="tight")
plt.close()

# Cluster Matching Accuracy
target_candidates = ["Placement", "Placed", "Status", "Target", "PlacementStatus"]
target_column = None
for candidate in target_candidates:
    norm_candidate = normalize_column_name(candidate)
    if norm_candidate in normalized_columns:
        target_column = normalized_columns[norm_candidate]
        break

def cluster_matching_accuracy(true_values, cluster_values):
    true_values = pd.Series(true_values).reset_index(drop=True)
    cluster_values = pd.Series(cluster_values).reset_index(drop=True)
    predicted_values = np.empty(len(cluster_values), dtype=object)
    for cluster in np.unique(cluster_values):
        indexes = np.where(cluster_values == cluster)[0]
        cluster_targets = true_values.iloc[indexes]
        majority = cluster_targets.mode()
        if len(majority) > 0:
            predicted_values[indexes] = majority.iloc[0]
    return accuracy_score(true_values, predicted_values)

if target_column is not None:
    print("\n" + "=" * 75)
    print("PLACEMENT CLUSTER MATCHING ACCURACY")
    print("=" * 75)
    target = data[target_column].copy()
    valid = target.notna()
    target_valid = target[valid].reset_index(drop=True)
    km_labels_valid = pd.Series(kmeans_labels, index=data.index)[valid].reset_index(drop=True)
    pp_labels_valid = pd.Series(kmeans_pp_labels, index=data.index)[valid].reset_index(drop=True)
    km_accuracy = cluster_matching_accuracy(target_valid, km_labels_valid)
    pp_accuracy = cluster_matching_accuracy(target_valid, pp_labels_valid)
    print("Target column:", target_column)
    print(f"K-Means   : {km_accuracy * 100:.2f}%")
    print(f"K-Means++ : {pp_accuracy * 100:.2f}%")

    accuracy_result = pd.DataFrame({
        "Method": ["K-Means", "K-Means++"],
        "Cluster_Matching_Accuracy": [km_accuracy, pp_accuracy],
        "Accuracy_Percentage": [km_accuracy * 100, pp_accuracy * 100]
    })
    accuracy_result.to_csv(os.path.join(ACCURACY_FOLDER, "placement_cluster_matching_accuracy.csv"), index=False)

print("\n" + "=" * 75)
print("FINAL CLUSTERING RESULTS")
print("=" * 75)
print(f"K-Means   Best K: {best_k_kmeans} | Silhouette: {km_silhouette:.4f} | Calinski: {km_calinski:.4f} | Davies-Bouldin: {km_davies:.4f}")
print(f"K-Means++ Best K: {best_k_kmeans_pp} | Silhouette: {pp_silhouette:.4f} | Calinski: {pp_calinski:.4f} | Davies-Bouldin: {pp_davies:.4f}")
print("\nOutputs saved to:", OUTPUT_FOLDER)
print("PROGRAM COMPLETED SUCCESSFULLY.")
