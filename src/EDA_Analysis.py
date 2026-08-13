import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------
# Configuration
# ----------------------

DATASET_PATH = "/Users/khvsaieswar/Desktop/Placement_prediction_recovered/dataset/placement_predict_50K_Raw.csv"
OUTPUT_FOLDER = "/Users/khvsaieswar/Desktop/Placement_prediction_recovered/outputs/EDA_Analysis_outputs"

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Plot style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

df = pd.read_csv(DATASET_PATH)

print("=" * 60)

print("First Five Records")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumn Names")

print(df.columns.tolist())

print("\nData Types")
print(df.dtypes)
print("\nDataset Information")
print(df.info())
print("\nMissing Values")
print(df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
target = None
possible_targets = ["Placement", "PlacementStatus", "Status", "Placed"]
for col in possible_targets:
    if col in df.columns:
        target = col
        break
for col in numeric_cols:

    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], bins=20, kde=True, color='skyblue')
    plt.title(f"Histogram - {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(OUTPUT_FOLDER, "univariate_histogram.png"))
    plt.close()
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df[col], color="orange")
    plt.title(f"Box Plot - {col}")
    plt.savefig(os.path.join(OUTPUT_FOLDER, "boxplot.png"))
    plt.savefig(os.path.join(OUTPUT_FOLDER,
                             f"{col}_boxplot.png"),
                dpi=300,
                bbox_inches='tight')
    plt.close()
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df[col], color='red')
    plt.title(f"Outlier Detection - {col}")
    plt.savefig(os.path.join(OUTPUT_FOLDER, "outlier.png"))

    plt.close()

summary = df.describe(include='all')
print("\nStatistical Summary")
print(summary)


summary.to_csv(os.path.join(OUTPUT_FOLDER, "Statistical_Summary.csv"))


# -------------------------------
# Correlation Matrix
# -------------------------------
numeric_df = df.select_dtypes(include=np.number)


corr = numeric_df.corr()
corr.to_csv(os.path.join(OUTPUT_FOLDER, "Correlation_Matrix.csv"))


plt.figure(figsize=(12, 8))
sns.heatmap(corr,
            annot=True,
            cmap="coolwarm",
            linewidths=0.5)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "Correlation_Heatmap.png"))
plt.close()


# -------------------------------
# Histograms
# -------------------------------
for col in numeric_df.columns:
    plt.figure()
    sns.histplot(df[col], kde=True, color="steelblue")
    plt.title(f"Histogram of {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER,
                             f"Histogram_{col}.png"))
    plt.close()


# -------------------------------
# Boxplots
# -------------------------------
for col in numeric_df.columns:
    plt.figure()
    sns.boxplot(x=df[col], color="orange")
    plt.title(f"Boxplot of {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER,
                             f"Boxplot_{col}.png"))
    plt.close()


# -------------------------------
# Count Plots for Categorical Columns
# -------------------------------
categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns


for col in categorical_columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=col)

    plt.xticks(rotation=45)
    plt.title(f"Count Plot of {col}")
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_FOLDER,
                             f"Countplot_{col}.png"))
    plt.close()


# -------------------------------
# Pair Plot
# -------------------------------
if len(numeric_df.columns) > 1:
    pair = sns.pairplot(numeric_df)
    pair.savefig(os.path.join(OUTPUT_FOLDER,
                              "Pairplot.png"))
    plt.close()


# -------------------------------
# Missing Value Heatmap
# -------------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(),
            cbar=False,
            cmap="viridis")


plt.title("Missing Values Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER,
                         "Missing_Values_Heatmap.png"))
plt.close()


# -----------------------------
# Create Scatter Plot
# -----------------------------
# Select X and Y Columns
x_col = numeric_cols[0]
y_col = numeric_cols[1]


# Detect target column (if available)
target = None
for col in ["Placement", "PlacementStatus", "Status", "Placed"]:
    if col in df.columns:
        target = col
        break


# -----------------------------
# Generate Scatter Plot
# -----------------------------
plt.figure(figsize=(8, 6))


if target:
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=target,
        palette="Set1",
        s=80
    )
else:
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        color="blue",
        s=80
    )


plt.title(f"Scatter Plot: {x_col} vs {y_col}")
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, "sctterplot.png"))
plt.close()


# -----------------------------
# Find target column (optional)
# -----------------------------
target = None
possible_targets = ["PlacementStatus", "Placement", "Status", "Placed"]


for col in possible_targets:
    if col in df.columns:
        target = col
        break


# -------------------------------
# Target Variable Distribution
# -------------------------------
target_candidates = ["PlacementStatus", "Placed", "Placement", "Status"]


target = None
for col in target_candidates:
    if col in df.columns:
        target = col
        break


if target is not None:
    plt.figure()
    sns.countplot(data=df, x=target)
    plt.title(f"{target} Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER,
                             "Target_Distribution.png"))
    plt.close()


# -------------------------------
# Feature vs Target Boxplots
# -------------------------------
if target is not None:

    for col in numeric_df.columns:

        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df[target], y=df[col])

        plt.title(f"{col} vs {target}")
        plt.tight_layout()

        plt.savefig(os.path.join(
            OUTPUT_FOLDER,
            f"{col}_vs_{target}.png"
        ))

        plt.close()


print("\nEDA Completed Successfully.")
print("All figures are saved in:", OUTPUT_FOLDER)