from Settings import *

import pandas as pd
import numpy as np


def detect_and_remove_outliers(max_ratio):
    df = pd.read_csv(path_csv_all)

    outlier_report = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_drop = []

    def is_binary(series):
        return set(series.dropna().unique()).issubset({0, 1})

    for col in numeric_cols:
        series = df[col].dropna()

        # Binary kontrolü
        binary_flag = is_binary(series)

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = (series < lower_bound) | (series > upper_bound)
        outlier_count = outliers.sum()
        outlier_ratio = outlier_count / len(series) * 100

        # Renkli kolon etiketi
        if binary_flag:
            col_name = f"🔵 {col}"
        else:
            if outlier_ratio < 5:
                col_name = f"🟢 {col}"
            elif outlier_ratio < 10:
                col_name = f"🟠 {col}"
            else:
                col_name = f"🔴 {col}"

        # Silinecekler listesine ekle
        if not binary_flag and outlier_ratio > max_ratio:
            cols_to_drop.append(col)

        outlier_report[col_name] = {
            "outlier_count": int(outlier_count),
            "outlier_ratio": round(outlier_ratio, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "min": round(series.min(), 2),
            "max": round(series.max(), 2),
            "mean": round(series.mean(), 2),
            "std": round(series.std(), 2)
        }

    # Kolonları kaldır
    if "Close" in cols_to_drop:
        cols_to_drop.remove("Close")  # Close'u silme

    cleaned_df = df.drop(columns=cols_to_drop)

    # Silinen kolonları logla (CSV olarak)
    log_path = path_data / "Ready" / f"Log_Columns_Removed_Outlier_{max_ratio}.csv"
    log_df = pd.DataFrame({"Removed_Column": cols_to_drop})
    log_df.to_csv(log_path, index=False)

    # Kaydet
    cleaned_df.to_csv(path_csv_no_outlier, index=False)
    cleaned_df.tail(5000).to_csv(path_csv_no_outlier_last5000, index=False)
    cleaned_df.iloc[:-5000].to_csv(path_csv_no_outlier_nolast5000, index=False)
    print(f"{len(cols_to_drop)} column(s) removed due to outlier ratio > {max_ratio}%.\n")