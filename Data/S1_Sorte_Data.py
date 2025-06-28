import pandas as pd
from datetime import datetime

raw_folder = "Raw/"
merged_folder = "Merged/"
ready_folder = "Ready/"


def parse_time(t):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y.%m.%d %H:%M:%S"):
        try:
            return datetime.strptime(t.strip(), fmt)
        except:
            continue
    print(f"❌ Failed to parse date: {t}")
    return pd.NaT


def sorte_candles(symbol):
    df = pd.read_csv(raw_folder + f"{symbol}_Indicators.csv")
    df["Time"] = df["Time"].apply(parse_time)
    df = df.sort_values("Time")
    df["Hour"] = df["Time"].dt.hour
    df["Minute"] = df["Time"].dt.minute
    df = df.drop_duplicates()
    sorted_path = f"{merged_folder}{symbol}_Ind_Sorted.csv"
    df.to_csv(sorted_path, index=False)
    print(f"✅ File saved successfully: {sorted_path}")


def sort_orders(symbol):
    df = pd.read_csv(f"{raw_folder}{symbol}_Orders.csv")
    df["OpenTime"] = df["OpenTime"].apply(parse_time)
    df = df.sort_values("OpenTime")
    df = df.drop_duplicates()

    sorted_path = f"{merged_folder}{symbol}_Orders_Sorted.csv"
    df.to_csv(sorted_path, index=False)
    print(f"✅ File saved successfully: {sorted_path}")


def merge_orders_with_candles(symbol):
    df_orders = pd.read_csv(f"{merged_folder}{symbol}_Orders_Sorted.csv")
    df_candles = pd.read_csv(f"{merged_folder}{symbol}_Ind_Sorted.csv")

    # Tarih sütunlarını datetime formatına çeviriyoruz
    df_orders["OpenTime"] = pd.to_datetime(df_orders["OpenTime"])
    df_candles["Time"] = pd.to_datetime(df_candles["Time"])

    # Merge işlemi (LEFT JOIN, yani tüm mumlar kalacak, işlemler varsa eşleşecek)
    merged_df = pd.merge(df_candles, df_orders, how="left", left_on="Time", right_on="OpenTime")

    # Eğer işlem yoksa kar sıfır olacak
    merged_df["Profit"] = merged_df["Profit"].fillna(0.0)
    merged_df["OpenPrice"] = merged_df["OpenPrice"].fillna(0.0)

    # ProfitLabel: sell 1 , buy 2, nötrse 0
    merged_df["ProfitLabel"] = merged_df["Profit"].apply(lambda x: 1 if x < 0 else (2 if x > 0 else 0))

    # OpenTime sütununu artık istemiyoruz (zaten Time var)
    merged_df = merged_df.drop(columns=["OpenTime"])

    # Kolonların sıralamasını ayarlıyoruz
    cols = merged_df.columns.tolist()
    time_idx = cols.index("Time")
    cols = (
        cols[:time_idx + 1] +
        ["Profit", "ProfitLabel"] +
        [col for col in cols if col not in ("Time", "Profit", "ProfitLabel")]
    )
    merged_df = merged_df[cols]

    # Sonuç dosyasını kaydet
    save_path = f"{merged_folder}{symbol}_Merged.csv"
    merged_df.to_csv(save_path, index=False)
    print(f"✅ File saved successfully: {save_path}")
