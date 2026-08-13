import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/khvsaieswar/Desktop/Placement_prediction_recovered/dataset/placement_predict_50K_Raw.csv')

print("--- First 5 Rows ---")
print(df.head())
print("----print 6 columns----")
subset = df.iloc[:, 0:6]
print(subset)
missing_counts = df.isnull().sum()
total_missing_values = df.isnull().sum().sum()
print("-----Missing Values Per Column:----------")
print(missing_counts)
print(total_missing_values)
print("-" * 40)
duplicate_rows = df[df.duplicated()]
print(f"Total duplicate rows detected: {len(duplicate_rows)}")
print(duplicate_rows)
print("-" * 40)
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()
