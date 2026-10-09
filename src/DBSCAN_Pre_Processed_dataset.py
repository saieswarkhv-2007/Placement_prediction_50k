# ============================================================
# DBSCAN CLUSTERING ON PREPROCESSED PLACEMENT PREDICTION DATASET
# Features:
# 1. CGPA
# 2. HistoryOfBacklogs
# 3. Internships
#
# Original Preprocessed Dataset is NOT Modified
# All Outputs and Performance Results -> ONE Folder
# ============================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ============================================================
# 1. INPUT AND OUTPUT PATH
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "dataset", "final_preprocess_M2.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "DBSCAN_Pre_Processed_dataset_outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 2. LOAD PREPROCESSED DATASET
# ============================================================
print("=" * 75)
print("DBSCAN CLUSTERING - PREPROCESSED PLACEMENT DATASET")
print("=" * 75)

if not os.path.exists(INPUT_FILE):
    print("\nERROR: Input file not found!")
    print("Check the following path:", INPUT_FILE)
    raise SystemExit

data = pd.read_csv(INPUT_FILE)
print("\nOriginal Preprocessed Dataset Shape:", data.shape)
print("\nAvailable Columns:", list(data.columns))

# ============================================================
# 3. SELECT REQUIRED ATTRIBUTES
# ============================================================
FEATURES = ["CGPA", "HistoryOfBacklogs", "Internships"]
missing_columns = [col for col in FEATURES if col not in data.columns]
if missing_columns:
    print("\nERROR: The following required columns are missing:", missing_columns)
    print("\nPlease check the column names in the preprocessed dataset.")
    raise SystemExit

# ============================================================
# 4. CREATE A SEPARATE COPY
# ============================================================
dbscan_data = data[FEATURES].copy()

# ============================================================
# 5. CONVERT FEATURES TO NUMERIC
# ============================================================
for column in FEATURES:
    dbscan_data[column] = pd.to_numeric(dbscan_data[column], errors="coerce")

# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================
print("\nMissing Values:")
print(dbscan_data.isnull().sum())

# ============================================================
# 7. REMOVE INVALID ROWS FROM THE COPY ONLY
# ============================================================
before_cleaning = len(dbscan_data)
dbscan_data = dbscan_data.dropna().reset_index(drop=True)
after_cleaning = len(dbscan_data)
removed_rows = before_cleaning - after_cleaning
print("\nRows before cleaning:", before_cleaning)
print("Rows after cleaning :", after_cleaning)
print("Rows removed        :", removed_rows)

# ============================================================
# 8. SAVE SELECTED FEATURES
# ============================================================
selected_features_file = os.path.join(OUTPUT_FOLDER, "DBSCAN_selected_features.csv")
dbscan_data.to_csv(selected_features_file, index=False)

# ============================================================
# 9. STANDARDIZATION
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(dbscan_data[FEATURES])

# ============================================================
# 10. SAVE STANDARDIZED DATA
# ============================================================
standardized_data = pd.DataFrame(X_scaled, columns=FEATURES)
standardized_file = os.path.join(OUTPUT_FOLDER, "DBSCAN_standardized_features.csv")
standardized_data.to_csv(standardized_file, index=False)

# ============================================================
# 11. DBSCAN PARAMETERS
# ============================================================
EPS = 0.80
MIN_SAMPLES = 10
print("\nDBSCAN Parameters")
print("-" * 40)
print("eps         :", EPS)
print("min_samples :", MIN_SAMPLES)

# ============================================================
# 12. APPLY DBSCAN
# ============================================================
dbscan_model = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
cluster_labels = dbscan_model.fit_predict(X_scaled)

# ============================================================
# 13. CREATE CLUSTERED DATASET
# ============================================================
clustered_data = dbscan_data.copy()
clustered_data["DBSCAN_Cluster"] = cluster_labels

# ============================================================
# 14. CLUSTER INFORMATION
# ============================================================
unique_labels = sorted(np.unique(cluster_labels))
actual_clusters = [label for label in unique_labels if label != -1]
number_of_clusters = len(actual_clusters)
noise_count = int(np.sum(cluster_labels == -1))
clustered_count = len(cluster_labels) - noise_count
total_records = len(cluster_labels)

if total_records > 0:
    noise_percentage = (noise_count / total_records) * 100
    clustered_percentage = (clustered_count / total_records) * 100
else:
    noise_percentage = 0
    clustered_percentage = 0

print("\n" + "=" * 75)
print("DBSCAN RESULTS")
print("=" * 75)
print("\nTotal Records       :", total_records)
print("Number of Clusters  :", number_of_clusters)
print("Noise Points        :", noise_count)
print("Clustered Points    :", clustered_count)
print("Noise Percentage    :", round(noise_percentage, 2), "%")
print("Clustered Percentage:", round(clustered_percentage, 2), "%")

# ============================================================
# 15. SAVE CLUSTERED DATA
# ============================================================
clustered_file = os.path.join(OUTPUT_FOLDER, "DBSCAN_clustered_data.csv")
clustered_data.to_csv(clustered_file, index=False)

# ============================================================
# 16. CLUSTER DISTRIBUTION
# ============================================================
cluster_counts = clustered_data["DBSCAN_Cluster"].value_counts().sort_index()
print("\nCluster Distribution:")
print(cluster_counts)

cluster_distribution = pd.DataFrame({
    "Cluster": cluster_counts.index,
    "Number_of_Records": cluster_counts.values
})
cluster_distribution_file = os.path.join(OUTPUT_FOLDER, "DBSCAN_cluster_distribution.csv")
cluster_distribution.to_csv(cluster_distribution_file, index=False)

# ============================================================
# 17. CLUSTER SUMMARY
# ============================================================
summary_records = []
for cluster_id in unique_labels:
    cluster_subset = clustered_data[clustered_data["DBSCAN_Cluster"] == cluster_id]
    cluster_name = "Noise" if cluster_id == -1 else f"Cluster_{cluster_id}"
    summary_records.append({
        "Cluster": cluster_name,
        "Cluster_ID": cluster_id,
        "Number_of_Records": len(cluster_subset),
        "Average_CGPA": round(cluster_subset["CGPA"].mean(), 3),
        "Average_HistoryOfBacklogs": round(cluster_subset["HistoryOfBacklogs"].mean(), 3),
        "Average_Internships": round(cluster_subset["Internships"].mean(), 3)
    })

cluster_summary = pd.DataFrame(summary_records)
cluster_summary_file = os.path.join(OUTPUT_FOLDER, "DBSCAN_cluster_summary.csv")
cluster_summary.to_csv(cluster_summary_file, index=False)

# ============================================================
# 18. PERFORMANCE RESULTS
# ============================================================
performance_results = pd.DataFrame({
    "Metric": [
        "Algorithm",
        "Input Dataset",
        "Original Dataset Rows",
        "Rows Used",
        "Rows Removed",
        "Number of Features",
        "Features Used",
        "Scaling Method",
        "DBSCAN eps",
        "DBSCAN min_samples",
        "Number of Clusters",
        "Noise Points",
        "Clustered Points",
        "Noise Percentage",
        "Clustered Percentage"
    ],
    "Value": [
        "DBSCAN",
        os.path.basename(INPUT_FILE),
        before_cleaning,
        after_cleaning,
        removed_rows,
        len(FEATURES),
        ", ".join(FEATURES),
        "StandardScaler",
        EPS,
        MIN_SAMPLES,
        number_of_clusters,
        noise_count,
        clustered_count,
        round(noise_percentage, 2),
        round(clustered_percentage, 2)
    ]
})
performance_file = os.path.join(OUTPUT_FOLDER, "DBSCAN_performance_results.csv")
performance_results.to_csv(performance_file, index=False)

# ============================================================
# 19. PCA FOR 2-D VISUALIZATION
# ============================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_data = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "DBSCAN_Cluster": cluster_labels
})
pca_file = os.path.join(OUTPUT_FOLDER, "DBSCAN_PCA_coordinates.csv")
pca_data.to_csv(pca_file, index=False)

# ============================================================
# 20. GRAPHS
# ============================================================
# Graph 1 - CGPA vs Internships
plt.figure(figsize=(10, 7))
plt.scatter(dbscan_data["CGPA"], dbscan_data["Internships"], c=cluster_labels, s=25, alpha=0.7)
plt.xlabel("CGPA")
plt.ylabel("Internships")
plt.title("DBSCAN Clustering: CGPA vs Internships")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "01_DBSCAN_CGPA_vs_Internships.png"), dpi=300, bbox_inches="tight")
plt.close()

# Graph 2 - CGPA vs History of Backlogs
plt.figure(figsize=(10, 7))
plt.scatter(dbscan_data["CGPA"], dbscan_data["HistoryOfBacklogs"], c=cluster_labels, s=25, alpha=0.7)
plt.xlabel("CGPA")
plt.ylabel("HistoryOfBacklogs")
plt.title("DBSCAN Clustering: CGPA vs HistoryOfBacklogs")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "02_DBSCAN_CGPA_vs_HistoryOfBacklogs.png"), dpi=300, bbox_inches="tight")
plt.close()

# Graph 3 - Backlogs vs Internships
plt.figure(figsize=(10, 7))
plt.scatter(dbscan_data["HistoryOfBacklogs"], dbscan_data["Internships"], c=cluster_labels, s=25, alpha=0.7)
plt.xlabel("HistoryOfBacklogs")
plt.ylabel("Internships")
plt.title("DBSCAN Clustering: HistoryOfBacklogs vs Internships")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "03_DBSCAN_HistoryOfBacklogs_vs_Internships.png"), dpi=300, bbox_inches="tight")
plt.close()

# Graph 4 - PCA Cluster Visualization
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, s=25, alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("DBSCAN Cluster Visualization using PCA")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "04_DBSCAN_PCA_Clusters.png"), dpi=300, bbox_inches="tight")
plt.close()

# Graph 5 - Cluster Distribution
plt.figure(figsize=(10, 6))
plt.bar(cluster_distribution["Cluster"].astype(str), cluster_distribution["Number_of_Records"])
plt.xlabel("DBSCAN Cluster")
plt.ylabel("Number of Records")
plt.title("DBSCAN Cluster Distribution")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "05_DBSCAN_Cluster_Distribution.png"), dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 25. SAVE EXECUTION INFORMATION
# ============================================================
execution_information = pd.DataFrame({
    "Item": [
        "Input Dataset",
        "Output Folder",
        "Original Dataset Modified",
        "Features Used",
        "Preprocessing",
        "Scaling",
        "Algorithm",
        "eps",
        "min_samples"
    ],
    "Details": [
        INPUT_FILE,
        OUTPUT_FOLDER,
        "NO",
        ", ".join(FEATURES),
        "Existing preprocessed dataset",
        "StandardScaler",
        "DBSCAN",
        EPS,
        MIN_SAMPLES
    ]
})
execution_file = os.path.join(OUTPUT_FOLDER, "execution_information.csv")
execution_information.to_csv(execution_file, index=False)

# ============================================================
# 26. FINAL OUTPUT
# ============================================================
print("\n" + "=" * 75)
print("ALL OUTPUTS GENERATED SUCCESSFULLY")
print("=" * 75)
print("\nOutput Folder:", OUTPUT_FOLDER)
print("\nFiles Generated:")
for filename in sorted(os.listdir(OUTPUT_FOLDER)):
    print(" ", filename)
print("\n" + "=" * 75)
print("DBSCAN ANALYSIS COMPLETED")
print("Original preprocessed dataset was NOT modified.")
print("=" * 75)
