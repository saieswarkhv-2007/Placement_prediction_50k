# ============================================================
# t-SNE DIMENSIONALITY REDUCTION
# Placement Prediction Preprocessed Dataset
#
# IMPORTANT:
# 1. Original dataset is NOT modified.
# 2. All processing is performed on a copy.
# 3. All outputs are stored in ONE folder.
# 4. Uses sklearn.manifold.TSNE.
# ============================================================
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================
# 1. INPUT AND OUTPUT PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "dataset", "final_preprocess_M2.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "t-SNE_outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# 2. LOAD DATASET
# ============================================================
print("=" * 70)
print("t-SNE DIMENSIONALITY REDUCTION")
print("PLACEMENT PREDICTION PREPROCESSED DATASET")
print("=" * 70)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"\nDataset not found:\n{INPUT_FILE}")

data = pd.read_csv(INPUT_FILE)
print("\nDataset loaded successfully.")
print("Original dataset shape:", data.shape)

# ============================================================
# 3. VALIDATE DATASET
# ============================================================
if data.empty:
    raise ValueError("The dataset is empty.")

if data.shape[1] < 2:
    raise ValueError("Dataset must contain at least two columns.")

# ============================================================
# 4. CREATE COPY
# ============================================================
working_data = data.copy()

# ============================================================
# 5. SELECT NUMERIC FEATURES
# ============================================================
numeric_data = working_data.select_dtypes(include=[np.number]).copy()
print("\nNumber of numeric features:", numeric_data.shape[1])
print("\nNumeric features used:")
for column in numeric_data.columns:
    print(" -", column)

if numeric_data.shape[1] < 2:
    raise ValueError("At least two numeric features are required for t-SNE.")

# ============================================================
# 6. HANDLE INFINITE VALUES
# ============================================================
numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)

# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================
missing_before = int(numeric_data.isnull().sum().sum())
print("\nMissing values before processing:", missing_before)

numeric_data = numeric_data.fillna(numeric_data.median())
numeric_data = numeric_data.fillna(0)
missing_after = int(numeric_data.isnull().sum().sum())
print("Missing values after processing:", missing_after)

# ============================================================
# 8. REMOVE CONSTANT FEATURES
# ============================================================
constant_columns = [
    column for column in numeric_data.columns
    if numeric_data[column].nunique() <= 1
]
if constant_columns:
    print("\nConstant features removed:")
    for column in constant_columns:
        print(" -", column)
    numeric_data = numeric_data.drop(columns=constant_columns)
else:
    print("\nNo constant features found.")

if numeric_data.shape[1] < 2:
    raise ValueError("Fewer than two usable features remain.")

# ============================================================
# 9. STANDARDIZE FEATURES
# ============================================================
scaler = StandardScaler()
X = scaler.fit_transform(numeric_data)
print("\nStandardization completed.")
print("Feature matrix shape:", X.shape)

# ============================================================
# 10. SAVE FEATURE INFORMATION
# ============================================================
feature_information = pd.DataFrame({
    "Feature_Name": numeric_data.columns
})
feature_file = os.path.join(OUTPUT_FOLDER, "features_used.csv")
feature_information.to_csv(feature_file, index=False)

# ============================================================
# 11. SET t-SNE PARAMETERS
# ============================================================
n_samples = X.shape[0]
if n_samples <= 10:
    perplexity = max(2, n_samples // 3)
elif n_samples <= 50:
    perplexity = 5
elif n_samples <= 100:
    perplexity = 10
else:
    perplexity = 30

perplexity = min(perplexity, n_samples - 1)

# ============================================================
# 12. RUN t-SNE
# ============================================================
print("\n" + "=" * 70)
print("RUNNING t-SNE")
print("=" * 70)
print("\nNumber of samples:", n_samples)
print("Number of input features:", X.shape[1])
print("Perplexity:", perplexity)

tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    learning_rate="auto",
    max_iter=1000,
    init="pca",
    random_state=42
)
X_tsne = tsne.fit_transform(X)
print("\nt-SNE completed successfully.")

# ============================================================
# 13. CREATE t-SNE RESULT DATAFRAME
# ============================================================
tsne_results = pd.DataFrame({
    "Record_ID": np.arange(1, n_samples + 1),
    "tSNE_Component_1": X_tsne[:, 0],
    "tSNE_Component_2": X_tsne[:, 1]
})

# ============================================================
# 14. ADD ORIGINAL DATA COLUMNS
# ============================================================
for column in data.columns:
    tsne_results[column] = data[column].values

# ============================================================
# 15. SAVE t-SNE TRANSFORMED DATA
# ============================================================
tsne_file = os.path.join(OUTPUT_FOLDER, "tsne_transformed_data.csv")
tsne_results.to_csv(tsne_file, index=False)
print("\nt-SNE transformed data saved to:", tsne_file)

# ============================================================
# 16. SAVE ONLY t-SNE COORDINATES
# ============================================================
coordinates = pd.DataFrame({
    "Record_ID": np.arange(1, n_samples + 1),
    "tSNE_1": X_tsne[:, 0],
    "tSNE_2": X_tsne[:, 1]
})
coordinates_file = os.path.join(OUTPUT_FOLDER, "tsne_coordinates.csv")
coordinates.to_csv(coordinates_file, index=False)

# ============================================================
# 17. t-SNE STATISTICS
# ============================================================
tsne_statistics = pd.DataFrame({
    "Statistic": [
        "Minimum t-SNE Component 1",
        "Maximum t-SNE Component 1",
        "Mean t-SNE Component 1",
        "Standard Deviation t-SNE Component 1",
        "Minimum t-SNE Component 2",
        "Maximum t-SNE Component 2",
        "Mean t-SNE Component 2",
        "Standard Deviation t-SNE Component 2"
    ],
    "Value": [
        X_tsne[:, 0].min(),
        X_tsne[:, 0].max(),
        X_tsne[:, 0].mean(),
        X_tsne[:, 0].std(),
        X_tsne[:, 1].min(),
        X_tsne[:, 1].max(),
        X_tsne[:, 1].mean(),
        X_tsne[:, 1].std()
    ]
})
statistics_file = os.path.join(OUTPUT_FOLDER, "tsne_statistics.csv")
tsne_statistics.to_csv(statistics_file, index=False)

# ============================================================
# 18. SAVE MODEL PARAMETERS / PERFORMANCE RESULTS
# ============================================================
performance_results = pd.DataFrame({
    "Parameter": [
        "Input Dataset",
        "Original Number of Records",
        "Original Number of Columns",
        "Numeric Features Used",
        "Constant Features Removed",
        "Missing Values Before Processing",
        "Missing Values After Processing",
        "t-SNE Output Dimensions",
        "Perplexity",
        "Learning Rate",
        "Maximum Iterations",
        "Initialization",
        "Random State",
        "KL Divergence",
        "Final Iteration Count"
    ],
    "Value": [
        os.path.basename(INPUT_FILE),
        data.shape[0],
        data.shape[1],
        numeric_data.shape[1],
        len(constant_columns),
        missing_before,
        missing_after,
        2,
        perplexity,
        "auto",
        1000,
        "PCA",
        42,
        getattr(tsne, 'kl_divergence_', np.nan),
        getattr(tsne, 'n_iter_', 1000)
    ]
})
performance_file = os.path.join(OUTPUT_FOLDER, "performance_results.csv")
performance_results.to_csv(performance_file, index=False)

# ============================================================
# 19. PLOT 1 - BASIC t-SNE SCATTER PLOT
# ============================================================
plt.figure(figsize=(10, 7))
plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    s=15,
    alpha=0.7
)
plt.title("t-SNE Visualization of Placement Prediction Dataset")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot1 = os.path.join(OUTPUT_FOLDER, "tsne_scatter_plot.png")
plt.savefig(plot1, dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 20. PLOT 2 - t-SNE COMPONENT DISTRIBUTION
# ============================================================
plt.figure(figsize=(10, 6))
plt.hist(X_tsne[:, 0], bins=40, alpha=0.7, label="t-SNE Component 1")
plt.hist(X_tsne[:, 1], bins=40, alpha=0.7, label="t-SNE Component 2")
plt.title("Distribution of t-SNE Components")
plt.xlabel("t-SNE Value")
plt.ylabel("Number of Records")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot2 = os.path.join(OUTPUT_FOLDER, "tsne_component_distribution.png")
plt.savefig(plot2, dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 21. PLOT 3 - t-SNE RECORD INDEX
# ============================================================
plt.figure(figsize=(12, 6))
plt.plot(np.arange(1, n_samples + 1), X_tsne[:, 0], linewidth=0.8, label="t-SNE Component 1")
plt.plot(np.arange(1, n_samples + 1), X_tsne[:, 1], linewidth=0.8, label="t-SNE Component 2")
plt.title("t-SNE Components Across Dataset Records")
plt.xlabel("Record ID")
plt.ylabel("t-SNE Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot3 = os.path.join(OUTPUT_FOLDER, "tsne_components_by_record.png")
plt.savefig(plot3, dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 22. SAVE EXPLAINABILITY / INTERPRETATION NOTE
# ============================================================
interpretation = pd.DataFrame({
    "Item": [
        "Purpose",
        "Method",
        "Input",
        "Output",
        "Interpretation",
        "Distance Interpretation",
        "Original Dataset"
    ],
    "Description": [
        "Non-linear dimensionality reduction and visualization",
        "t-SNE",
        "Standardized numeric features",
        "Two-dimensional embedding",
        "Nearby points generally represent similar observations in the local structure",
        "Local neighborhood relationships are more meaningful than global distances",
        "Not modified"
    ]
})
interpretation_file = os.path.join(OUTPUT_FOLDER, "tsne_interpretation.csv")
interpretation.to_csv(interpretation_file, index=False)

# ============================================================
# 23. FINAL OUTPUT
# ============================================================
print("\n" + "=" * 70)
print("t-SNE PROCESS COMPLETED SUCCESSFULLY")
print("=" * 70)
print("\nOriginal dataset was NOT modified.")
print("\nOriginal dataset shape:", data.shape)
print("t-SNE output shape:", X_tsne.shape)
print("\nAll outputs are stored in:")
print(OUTPUT_FOLDER)
print("\nGenerated files:")
for filename in sorted(os.listdir(OUTPUT_FOLDER)):
    print(" -", filename)
print("\n" + "=" * 70)
