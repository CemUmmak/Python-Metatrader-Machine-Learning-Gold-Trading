from pathlib import Path
import pandas as pd
import numpy as np
import  os
import sys
import re
from datetime import datetime

symbol = "XAUUSD"


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

path_project = Path(__file__).resolve().parents[0]
path_data = path_project / "Data"

# Ready_All
path_csv_all = path_data / "Ready" / "Ready_All.csv"
path_csv_all_last5000 = path_data / "Ready" / "Ready_All_Last5000.csv"
path_csv_all_nolast5000 = path_data / "Ready" / "Ready_All_NoLast5000.csv"

# Ready_NoCorr
path_csv_no_corr = path_data / "Ready" / "Ready_NoCorr.csv"
path_csv_no_corr_last5000 = path_data / "Ready" / "Ready_NoCorr_Last5000.csv"
path_csv_no_corr_nolast5000 = path_data / "Ready" / "Ready_NoCorr_NoLast5000.csv"

# Ready_NoOutlier
path_csv_no_outlier = path_data / "Ready" / "Ready_NoOutlier.csv"
path_csv_no_outlier_last5000 = path_data / "Ready" / "Ready_NoOutlier_Last5000.csv"
path_csv_no_outlier_nolast5000 = path_data / "Ready" / "Ready_NoOutlier_NoLast5000.csv"

# Ready_Clean (hem outlier hem korelasyon temizlenmiş)
path_csv_cleaned = path_data / "Ready" / "Ready_All_Clean.csv"
path_csv_cleaned_last5000 = path_data / "Ready" / "Ready_Clean_Last5000.csv"
path_csv_cleaned_nolast5000 = path_data / "Ready" / "Ready_Clean_NoLast5000.csv"


def get_data_name_from_path(path: Path) -> str:
    filename = path.name.lower()

    if "clean" in filename:
        return "Clean" + "_Data"
    elif "nooutlier" in filename:
        return "NoOutlier" + "_Data"
    elif "nocorr" in filename:
        return "NoCorr" + "_Data"
    elif "all" in filename:
        return "All" + "_Data"
    else:
        return "Unknown"

def print_inline_progress(current, total, prefix="Progress", bar_length=30):
    percent = current / total
    filled_len = int(bar_length * percent)
    bar = "█" * filled_len + "-" * (bar_length - filled_len)
    sys.stdout.write(f"\r{prefix} |{bar}| {percent*100:.1f}% ({current}/{total})")
    sys.stdout.flush()

import time

def create_lag_features(
    df: pd.DataFrame,
    n_lags: int = 20,
    label_column: str = "ProfitLabel",
    exclude_columns: list = ["Time"]
) -> pd.DataFrame:
    """
    Belirtilen label ve exclude kolonları dışında her feature için geçmiş n_lags kadar lag sütunu ekler.
    Performans için pd.concat kullanılır.
    """
    start = time.time()

    exclude_columns = set(exclude_columns + [label_column])
    features = [col for col in df.columns if col not in exclude_columns]

    lagged_columns = []

    for col in features:
        for lag in range(1, n_lags + 1):
            shifted = df[col].shift(lag)
            shifted.name = f"{col}_lag_{lag}"
            shifted.index = df.index
            lagged_columns.append(shifted)

    df_lags = pd.concat([df] + lagged_columns, axis=1)
    df_lags.dropna(inplace=True)
    df_lags = df_lags.reset_index(drop=True)

    duration = time.time() - start
    print(f"⏱️ Lag feature işlemi tamamlandı. Süre: {duration:.2f} saniye")

    return df_lags

def get_ready_drop_columns():
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Bu script'in bulunduğu klasör
    ready_dir = os.path.join(base_dir, "Data", "Ready")

    files = [
        "Log_Columns_Removed_Corr_90.csv",
        "Log_Columns_Removed_Outlier_15.csv"
    ]

    columns_to_drop = set()

    for file in files:
        file_path = os.path.join(ready_dir, file)
        if os.path.exists(file_path):
            df_removed = pd.read_csv(file_path)
            removed = df_removed["Removed_Column"].dropna().tolist()
            columns_to_drop.update(removed)
        else:
            print(f"⛔ Dosya bulunamadı: {file_path}")

    return list(columns_to_drop)

def parse_time(t):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y.%m.%d %H:%M:%S"):
        try:
            return datetime.strptime(t.strip(), fmt)
        except:
            continue
    print(f"❌ Failed to parse date: {t}")
    return pd.NaT
