import pandas as pd
import numpy as np


from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer


# ==========================================================
# Load Placement Prediction Dataset
# Original dataset will NOT be modified
# ==========================================================


df = pd.read_csv("/Users/khvsaieswar/Desktop/Placement_prediction/dataset/placement_predict_50K_Raw.csv")


# Create a copy for processing
data = df.copy()


# ==========================================================
# 1. Remove Leading and Trailing Spaces
# ==========================================================


for col in data.select_dtypes(include="object").columns:
    data[col] = data[col].str.strip()


# ==========================================================
# 2. Identify Missing Values
# ==========================================================


print("Missing Values Before Cleaning:")
print(data.isnull().sum())


# ==========================================================
# 3. Remove Duplicate Records
# ==========================================================


before_duplicates = data.shape[0]


data = data.drop_duplicates()


after_duplicates = data.shape[0]


print("\nDuplicate Records Removed:",
      before_duplicates - after_duplicates)


# ==========================================================
# Separate Numerical and Categorical Columns
# ==========================================================


num_cols = data.select_dtypes(
    include=np.number
).columns.tolist()


cat_cols = data.select_dtypes(
    exclude=np.number
).columns.tolist()


# ==========================================================
# 4. Fill Missing Numerical Values with Mean
# ==========================================================


if len(num_cols) > 0:


    num_imputer = SimpleImputer(
        strategy="mean"
    )


    data[num_cols] = num_imputer.fit_transform(
        data[num_cols]
    )


# ==========================================================
# 5. Fill Missing Categorical Values with Mode
# ==========================================================


if len(cat_cols) > 0:


    cat_imputer = SimpleImputer(
        strategy="most_frequent"
    )


    data[cat_cols] = cat_imputer.fit_transform(
        data[cat_cols]
    )


# ==========================================================
# 6. Apply Ordinal Encoding
# ==========================================================


if len(cat_cols) > 0:


    ordinal_encoder = OrdinalEncoder()


    encoded_values = ordinal_encoder.fit_transform(
        data[cat_cols]
    )


    encoded_df = pd.DataFrame(
        encoded_values,
        columns=[
            "Ordinal_" + col
            for col in cat_cols
        ]
    )


    # Keep numerical columns and encoded columns
    final_output = pd.concat(
        [
            data[num_cols].reset_index(drop=True),
            encoded_df.reset_index(drop=True)
        ],
        axis=1
    )


else:
    final_output = data.copy()


# ==========================================================
# 7. Check Missing Values After Cleaning
# ==========================================================


print("\nMissing Values After Cleaning:")
print(final_output.isnull().sum())


# ==========================================================
# Save Final Result
# ==========================================================


final_output.to_csv(
    "/Users/khvsaieswar/Desktop/Placement_prediction/dataset/clean_ordinal_encode_M2.csv",
    index=False
)


print("\n======================================")
print("Original dataset is NOT modified.")
print("Ordinal Encoding completed successfully.")
print("Output file:")
print("clean_ordinal_encode_M2.csv")
print("======================================")
