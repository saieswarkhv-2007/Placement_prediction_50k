import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

output_dir = "/Users/khvsaieswar/Desktop/Placement_prediction_recovered/outputs/Boxplots_correlation/"
os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv('/Users/khvsaieswar/Desktop/Placement_prediction_recovered/dataset/placement_predict_50K_Raw.csv')


print("Dataset Loaded Successfully. Shape:", df.shape)

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[numerical_cols].corr()

print("\n--- Correlation Matrix ---")
print(corr_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5
)
plt.title("Correlation Heatmap of Numerical Features", fontsize=14, fontweight="bold")
plt.tight_layout()
heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()
print(f"Exported heatmap to: {heatmap_path}")
target_col = "PlacementStatus"


if target_col in df.columns:
    for col in numerical_cols:
        plt.figure(figsize=(6, 5))
        sns.boxplot(
            x=target_col,
            y=col,
            data=df,
            palette="Set2",
            hue=target_col,
            legend=False
        )
        plt.title(f"{col} vs {target_col}", fontsize=12, fontweight="bold")
        plt.xlabel(target_col)
        plt.ylabel(col)
        plt.tight_layout()
        boxplot_filename = f"boxplot_{col}_vs_{target_col}.png"
        boxplot_path = os.path.join(output_dir, boxplot_filename)
        plt.savefig(boxplot_path, dpi=300)
        plt.close()
        print(f"Exported boxplot to: {boxplot_path}")
else:
    print(f"\nTarget column '{target_col}' not found in dataset. Skipping boxplots.")


print("\nAll EDA tasks completed successfully!")
