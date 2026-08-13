import pandas as pd


# ============================================================
# PLACEMENT PREDICTION DATASET
# HANDLING MISSING VALUES USING PANDAS ONLY
#
# Techniques:
# 1. Deletion
# 2. Mean Imputation
# 3. Median Imputation
# 4. Model-Based Imputation
# 5. Missing Indicator Features
#
# Original dataset is NOT modified.
# All results are stored in ONE CSV file.
# ============================================================


# ------------------------------------------------------------
# 1. READ DATASET
# ------------------------------------------------------------


input_file = "/Users/khvsaieswar/Desktop/Placement_prediction_recovered/dataset/placement_predict_50K_Raw.csv"
output_file = "/Users/khvsaieswar/Desktop/Placement_prediction_recovered/dataset/clean_del_mean_model_M2.csv"


df = pd.read_csv(input_file)


print("=" * 70)
print("ORIGINAL PLACEMENT PREDICTION DATASET")
print("=" * 70)


print(df.head())


print("\nDataset Shape:")
print(df.shape)


# ------------------------------------------------------------
# 2. CHECK MISSING VALUES
# ------------------------------------------------------------


print("\nMissing Values in Original Dataset:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 3. CREATE COPY
# ------------------------------------------------------------


df_original = df.copy()


# ============================================================
# METHOD 1: DELETION
# ============================================================


df_deletion = df_original.dropna().copy()


print("\n" + "=" * 70)
print("1. DELETION")
print("=" * 70)


print("Original rows:", len(df_original))
print("Rows after deletion:", len(df_deletion))
print("Rows deleted:", len(df_original) - len(df_deletion))


deletion_indicator = (
    df_original.isnull().any(axis=1)
).astype(int)


# ============================================================
# METHOD 2: MEAN IMPUTATION
# ============================================================


df_mean = df_original.copy()


numeric_columns = df_original.select_dtypes(
    include="number"
).columns.tolist()


print("\nNumerical Columns:")
print(numeric_columns)


for column in numeric_columns:


    if df_mean[column].isnull().any():
        mean_value = df_mean[column].mean()
        df_mean[column] = df_mean[column].fillna(
            mean_value
        )


        print(
            "Mean used for",
            column,
            "=",
            mean_value
        )


print("\n" + "=" * 70)
print("2. MEAN IMPUTATION")
print("=" * 70)


print(df_mean.head())


# ============================================================
# METHOD 3: MEDIAN IMPUTATION
# ============================================================


df_median = df_original.copy()


for column in numeric_columns:


    if df_median[column].isnull().any():
        median_value = df_median[column].median()


        df_median[column] = df_median[column].fillna(
            median_value
        )


        print(
            "Median used for",
            column,
            "=",
            median_value
        )


print("\n" + "=" * 70)
print("3. MEDIAN IMPUTATION")
print("=" * 70)


print(df_median.head())


# ============================================================
# METHOD 4: MODEL-BASED IMPUTATION
# ============================================================


df_model = df_original.copy()


print("\n" + "=" * 70)
print("4. MODEL-BASED IMPUTATION")
print("=" * 70)


if len(numeric_columns) >= 2:


    for target_column in numeric_columns:


        if not df_model[target_column].isnull().any():
            continue


        predictor_column = None


        for column in numeric_columns:


            if column != target_column:


                if (
                    df_model[column].notnull().sum()
                    >= 2
                ):


                    predictor_column = column
                    break


        if predictor_column is None:
            continue


        training_data = df_model[
            df_model[target_column].notnull()
            &
            df_model[predictor_column].notnull()
        ].copy()


        if len(training_data) < 2:
            continue


        x = training_data[predictor_column]
        y = training_data[target_column]


        x_mean = x.mean()
        y_mean = y.mean()


        numerator = (
            (x - x_mean) *
            (y - y_mean)
        ).sum()


        denominator = (
            (x - x_mean) ** 2
        ).sum()


        if denominator == 0:
            continue


        slope = numerator / denominator


        intercept = y_mean - (
            slope * x_mean
        )


        missing_rows = df_model[
            df_model[target_column].isnull()
            &
            df_model[predictor_column].notnull()
        ].copy()


        if len(missing_rows) > 0:


            predicted_values = (
                slope *
                missing_rows[predictor_column]
                +
                intercept
            )


            df_model.loc[
                missing_rows.index,
                target_column
            ] = predicted_values


            print(
                "\nTarget column:",
                target_column
            )


            print(
                "Predictor column:",
                predictor_column
            )


            print(
                "Slope:",
                slope
            )


            print(
                "Intercept:",
                intercept
            )


            print(
                "Number of values predicted:",
                len(missing_rows)
            )


for column in numeric_columns:


    if df_model[column].isnull().any():


        median_value = df_model[column].median()


        df_model[column] = df_model[column].fillna(
            median_value
        )


print("\nModel-Based Imputation Result:")
print(df_model.head())


categorical_columns = df_original.select_dtypes(
    exclude="number"
).columns.tolist()


for column in categorical_columns:


    if df_model[column].isnull().any():


        mode_values = df_model[column].mode()


        if len(mode_values) > 0:


            df_model[column] = df_model[column].fillna(
                mode_values.iloc[0]
            )


print("\n" + "=" * 70)
print("5. MISSING INDICATOR FEATURES")
print("=" * 70)


df_indicator = df_original.copy()


for column in df_original.columns:


    indicator_column = (
        column + "_Missing"
    )


    df_indicator[indicator_column] = (
        df_original[column].isnull()
    ).astype(int)


print("\nMissing Indicator Result:")
print(df_indicator.head())


final_result = df_original.copy()


final_result[
    "Deletion_Row_Removed"
] = deletion_indicator


for column in numeric_columns:


    final_result[
        column + "_Mean_Imputed"
    ] = df_mean[column]


for column in numeric_columns:


    final_result[
        column + "_Median_Imputed"
    ] = df_median[column]


for column in numeric_columns:


    final_result[
        column + "_Model_Imputed"
    ] = df_model[column]


for column in df_original.columns:


    final_result[
        column + "_Missing"
    ] = (
        df_original[column].isnull()
    ).astype(int)


final_result.to_csv(
    output_file,
    index=False
)


print("\n" + "=" * 70)
print("FINAL OUTPUT")
print("=" * 70)


print(final_result.head())


print("\nFinal Dataset Shape:")
print(final_result.shape)


print("\nMissing Values in Mean-Imputed Dataset:")
print(df_mean.isnull().sum())


print("\nMissing Values in Median-Imputed Dataset:")
print(df_median.isnull().sum())


print("\nMissing Values in Model-Imputed Dataset:")
print(df_model.isnull().sum())


df_check = pd.read_csv(input_file)


if df_original.equals(df_check):


    print(
        "\nOriginal dataset was NOT modified."
    )


else:


    print(
        "\nWARNING: Original dataset was modified!"
    )


print("\n" + "=" * 70)
print("PROCESS COMPLETED SUCCESSFULLY")
print("=" * 70)


print("\nInput file:")
print(input_file)


print("\nOutput file:")
print(output_file)


print(
    "\nAll missing-value handling techniques "
    "are stored in ONE CSV file."
)
