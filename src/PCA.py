# ============================================================
# PCA ON PLACEMENT PREDICTION PREPROCESSED DATASET
# ============================================================
# Original Dataset is NOT Modified
# All Outputs and Performance Results -> ONE Folder
# ============================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ============================================================
# 1. INPUT AND OUTPUT PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "dataset", "final_preprocess_M2.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "PCA_Outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 2. LOAD PREPROCESSED DATASET
# ============================================================
print("=" * 75)
print("PCA - PLACEMENT PREDICTION PREPROCESSED DATASET")
print("=" * 75)

if not os.path.exists(INPUT_FILE):
    print("\nERROR: Input dataset not found!")
    print("Check this path:", INPUT_FILE)
    raise SystemExit

data = pd.read_csv(INPUT_FILE)
print("\nOriginal Dataset Shape:")
print(data.shape)
print("\nDataset Columns:")
print(list(data.columns))

# ============================================================
# 3. CREATE A COPY
# ============================================================
pca_data = data.copy()

# ============================================================
# 4. IDENTIFY NUMERIC COLUMNS
# ============================================================
numeric_columns = pca_data.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumber of Numeric Attributes:")
print(len(numeric_columns))
print("\nNumeric Attributes:")
print(numeric_columns)

if len(numeric_columns) < 2:
    print("\nERROR: PCA requires at least two numeric attributes.")
    raise SystemExit

# ============================================================
# 5. EXTRACT NUMERIC DATA
# ============================================================
X = pca_data[numeric_columns].copy()

# ============================================================
# 6. CHECK MISSING VALUES
# ============================================================
print("\nMissing Values Before Processing:")
print(X.isnull().sum())
missing_before = int(X.isnull().sum().sum())

# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================
for column in numeric_columns:
    if X[column].isnull().any():
        median_value = X[column].median()
        X[column] = X[column].fillna(median_value)

missing_after = int(X.isnull().sum().sum())
print("\nMissing Values After Processing:")
print(X.isnull().sum())

# ============================================================
# 8. SAVE NUMERIC DATA USED FOR PCA
# ============================================================
numeric_data_file = os.path.join(OUTPUT_FOLDER, "PCA_numeric_data_used.csv")
X.to_csv(numeric_data_file, index=False)

# ============================================================
# 9. STANDARDIZATION
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 10. SAVE STANDARDIZED DATA
# ============================================================
standardized_data = pd.DataFrame(X_scaled, columns=numeric_columns)
standardized_file = os.path.join(OUTPUT_FOLDER, "PCA_standardized_data.csv")
standardized_data.to_csv(standardized_file, index=False)

# ============================================================
# 11. PCA WITH ALL COMPONENTS
# ============================================================
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

# ============================================================
# 12. EXPLAINED VARIANCE
# ============================================================
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# ============================================================
# 13. CREATE COMPONENT NAMES
# ============================================================
component_names = [f"PC{i + 1}" for i in range(len(explained_variance))]

# ============================================================
# 14. EXPLAINED VARIANCE TABLE
# ============================================================
variance_table = pd.DataFrame({
    "Principal_Component": component_names,
    "Eigenvalue": pca_full.explained_variance_,
    "Explained_Variance_Ratio": explained_variance,
    "Explained_Variance_Percentage": explained_variance * 100,
    "Cumulative_Variance_Ratio": cumulative_variance,
    "Cumulative_Variance_Percentage": cumulative_variance * 100
})
variance_file = os.path.join(OUTPUT_FOLDER, "PCA_explained_variance.csv")
variance_table.to_csv(variance_file, index=False)

# ============================================================
# 15. PCA TRANSFORMED DATA
# ============================================================
pca_transformed = pd.DataFrame(X_pca_full, columns=component_names)
pca_transformed_file = os.path.join(OUTPUT_FOLDER, "PCA_transformed_data_all_components.csv")
pca_transformed.to_csv(pca_transformed_file, index=False)

# ============================================================
# 16. COMPONENT LOADINGS
# ============================================================
loadings = pd.DataFrame(
    pca_full.components_.T,
    index=numeric_columns,
    columns=component_names
)
loadings.index.name = "Feature"
loadings_file = os.path.join(OUTPUT_FOLDER, "PCA_component_loadings.csv")
loadings.to_csv(loadings_file)

# ============================================================
# 17. DETERMINE COMPONENTS FOR 80%, 90%, 95% VARIANCE
# ============================================================
components_80 = int(np.argmax(cumulative_variance >= 0.80) + 1)
components_90 = int(np.argmax(cumulative_variance >= 0.90) + 1)
components_95 = int(np.argmax(cumulative_variance >= 0.95) + 1)

print("\n" + "=" * 75)
print("PCA VARIANCE RESULTS")
print("=" * 75)
print("\nComponents required for 80% variance:", components_80)
print("Components required for 90% variance:", components_90)
print("Components required for 95% variance:", components_95)

# ============================================================
# 18. PCA WITH 95% VARIANCE
# ============================================================
pca_95 = PCA(n_components=0.95)
X_pca_95 = pca_95.fit_transform(X_scaled)
pca95_component_names = [f"PC{i + 1}" for i in range(X_pca_95.shape[1])]

pca95_data = pd.DataFrame(X_pca_95, columns=pca95_component_names)
pca95_file = os.path.join(OUTPUT_FOLDER, "PCA_transformed_data_95_percent.csv")
pca95_data.to_csv(pca95_file, index=False)

# ============================================================
# 19. PERFORMANCE / SUMMARY RESULTS
# ============================================================
original_rows = data.shape[0]
original_columns = data.shape[1]
number_numeric_features = len(numeric_columns)
number_pca_components = len(component_names)
number_pca95_components = X_pca_95.shape[1]
first_component_variance = (explained_variance[0] * 100) if number_pca_components > 0 else 0

summary_results = pd.DataFrame({
    "Metric": [
        "Input Dataset",
        "Original Number of Rows",
        "Original Number of Columns",
        "Number of Numeric Features",
        "Number of PCA Components",
        "Components for 80% Variance",
        "Components for 90% Variance",
        "Components for 95% Variance",
        "PCA Components Retained at 95%",
        "First Principal Component Variance (%)",
        "Total Variance Explained (%)",
        "Missing Values Before Processing",
        "Missing Values After Processing",
        "Standardization Applied",
        "Original Dataset Modified"
    ],
    "Value": [
        os.path.basename(INPUT_FILE),
        original_rows,
        original_columns,
        number_numeric_features,
        number_pca_components,
        components_80,
        components_90,
        components_95,
        number_pca95_components,
        round(first_component_variance, 4),
        round(cumulative_variance[-1] * 100, 4),
        missing_before,
        missing_after,
        "Yes - StandardScaler",
        "NO"
    ]
})
summary_file = os.path.join(OUTPUT_FOLDER, "PCA_performance_results.csv")
summary_results.to_csv(summary_file, index=False)

# ============================================================
# 20. SCREE PLOT
# ============================================================
plt.figure(figsize=(10, 7))
plt.plot(
    range(1, len(explained_variance) + 1),
    explained_variance * 100,
    marker="o"
)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance (%)")
plt.title("PCA Scree Plot")
plt.grid(True, alpha=0.3)
plt.tight_layout()
scree_file = os.path.join(OUTPUT_FOLDER, "01_PCA_Scree_Plot.png")
plt.savefig(scree_file, dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 21. CUMULATIVE EXPLAINED VARIANCE PLOT
# ============================================================
plt.figure(figsize=(10, 7))
plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance * 100,
    marker="o"
)
plt.axhline(y=80, linestyle="--", label="80% Variance")
plt.axhline(y=90, linestyle="--", label="90% Variance")
plt.axhline(y=95, linestyle="--", label="95% Variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("PCA Cumulative Explained Variance")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
cumulative_file = os.path.join(OUTPUT_FOLDER, "02_PCA_Cumulative_Explained_Variance.png")
plt.savefig(cumulative_file, dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 22. 2-D PCA VISUALIZATION
# ============================================================
if X_pca_full.shape[1] >= 2:
    plt.figure(figsize=(10, 7))
    plt.scatter(
        X_pca_full[:, 0],
        X_pca_full[:, 1],
        s=20,
        alpha=0.7
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Placement Dataset - PCA Visualization")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pca_plot_file = os.path.join(OUTPUT_FOLDER, "03_PCA_2D_Visualization.png")
    plt.savefig(pca_plot_file, dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================
# 23. FEATURE CONTRIBUTION TO PC1
# ============================================================
pc1_loadings = loadings["PC1"].abs().sort_values(ascending=False)
plt.figure(figsize=(10, 7))
plt.bar(pc1_loadings.index, pc1_loadings.values)
plt.xlabel("Original Features")
plt.ylabel("Absolute Loading")
plt.title("Feature Contribution to Principal Component 1")
plt.xticks(rotation=90)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
pc1_plot_file = os.path.join(OUTPUT_FOLDER, "04_PCA_PC1_Feature_Contribution.png")
plt.savefig(pc1_plot_file, dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 24. SAVE EXECUTION INFORMATION
# ============================================================
execution_information = pd.DataFrame({
    "Item": [
        "Input Dataset",
        "Output Folder",
        "Features Used",
        "PCA Method",
        "Scaling Method",
        "Missing Value Handling",
        "Variance Retention Analysis",
        "Original Dataset Modified"
    ],
    "Details": [
        INPUT_FILE,
        OUTPUT_FOLDER,
        "All numeric attributes",
        "Principal Component Analysis",
        "StandardScaler",
        "Median imputation",
        "80%, 90%, 95%",
        "NO"
    ]
})
execution_file = os.path.join(OUTPUT_FOLDER, "execution_information.csv")
execution_information.to_csv(execution_file, index=False)

# ============================================================
# 25. FINAL OUTPUT LIST
# ============================================================
print("\n" + "=" * 75)
print("ALL PCA OUTPUTS GENERATED SUCCESSFULLY")
print("=" * 75)
print("\nOutput Folder:")
print(OUTPUT_FOLDER)
print("\nGenerated Files:")
for filename in sorted(os.listdir(OUTPUT_FOLDER)):
    print(" ", filename)
print("\n" + "=" * 75)
print("PCA ANALYSIS COMPLETED")
print("Original preprocessed dataset was NOT modified.")
print("=" * 75)
