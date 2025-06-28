from Settings import *

import pandas as pd
import numpy as np

def remove_corr(threshold):
    remove_from_all = remove_highly_correlated_features(path_csv_all, threshold)
    remove_from_all.to_csv(path_csv_no_corr, index=False)
    remove_from_all.tail(5000).to_csv(path_csv_no_corr_last5000, index=False)
    remove_from_all.iloc[:-5000].to_csv(path_csv_no_corr_nolast5000, index=False)

    remove_from_outlier = remove_highly_correlated_features(path_csv_no_outlier, threshold)
    remove_from_outlier.to_csv(path_csv_cleaned, index=False)
    remove_from_outlier.tail(5000).to_csv(path_csv_cleaned_last5000, index=False)
    remove_from_outlier.iloc[:-5000].to_csv(path_csv_cleaned_nolast5000, index=False)


def remove_highly_correlated_features(data_path, threshold):

    df = pd.read_csv(data_path)

    # Yüzdelik eşiği orana çevir
    corr_threshold = threshold / 100.0

    # Korunacak kolonları ayarla
    protected_cols = ["Time", "ProfitLabel", "TimeFilter", "Close"]

    # Sadece sayısal kolonları al (korunanlar hariç)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cols_to_check = [col for col in numeric_cols if col not in protected_cols]

    # Korelasyon matrisi
    corr_matrix = df[cols_to_check].corr().abs()

    # Üst üçgeni maskele
    upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    corr_matrix_masked = corr_matrix.where(upper)

    # Silinecek kolonları topla
    to_drop = [
        column
        for column in corr_matrix_masked.columns
        if any(corr_matrix_masked[column] > corr_threshold)
    ]

    # Temizlenmiş dataframe
    cleaned_df = df.drop(columns=to_drop)

    # Silinen kolonları logla (CSV olarak)
    log_path = path_data / "Ready" / f"Log_Columns_Removed_Corr_{threshold}.csv"

    # Daha önce varsa oku
    if log_path.exists():
        old_log_df = pd.read_csv(log_path)
        combined = pd.concat([old_log_df, pd.DataFrame({"Removed_Column": to_drop})])
        combined = combined.drop_duplicates().reset_index(drop=True)
    else:
        combined = pd.DataFrame({"Removed_Column": to_drop})

    # Yeni haliyle kaydet
    combined.to_csv(log_path, index=False)

    print(f"{len(to_drop)} kolon, % {threshold} üzeri korelasyon nedeniyle silindi.")

    return cleaned_df
