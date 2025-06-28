import time
import joblib
import numpy as np
from datetime import datetime
import pandas as pd
import os
from filelock import FileLock
from tensorflow.keras.models import load_model
from ML_Ind_V3.Data.Indicators.General import apply_price_features
from ML_Ind_V3.Data.Indicators.RSI import apply_rsi
from ML_Ind_V3.Data.Indicators.MACD import apply_macd
from ML_Ind_V3.Data.Indicators.CCI import apply_cci
from ML_Ind_V3.Data.Indicators.Stoch import apply_stoch
from ML_Ind_V3.Data.Indicators.IndMomentums import apply_ind_momentum
from ML_Ind_V3.Data.Indicators.Ma import apply_ma
from ML_Ind_V3.Data.Indicators.Ichimoku import apply_ichimoku
from ML_Ind_V3.Data.Indicators.BB import apply_bollinger
from ML_Ind_V3.Data.Indicators.Env import apply_envelopes
from ML_Ind_V3.Settings import get_ready_drop_columns, parse_time


# === Ayarlar ===
common_path = r"C:\Users\cem_u\AppData\Roaming\MetaQuotes\Terminal\Common\Files\\"
path_data = f"{common_path}XAUUSD_Live_V3.csv"

MAX_ROWS = 40
model_no = 999
WINDOW_SIZE = 20  # eğitimde kullandığın window size ile aynı olmalı

if MAX_ROWS < WINDOW_SIZE + 5:
    print("Live veri çok az !")
    exit()



last_data_static = 0.0

clean_csv_cols = get_ready_drop_columns()

# === Model ve Scaler Yükle ===
model = load_model(f"Models/Clean_Data/Clean_Data_Keras_Model_{model_no}.h5")
scaler = joblib.load(f"Models/Clean_Data/Clean_Data_Keras_Scaler_{model_no}.pkl")

while True:
    time.sleep(0.25)

    try:
        df = pd.read_csv(path_data)
    except Exception as ex:
        print(f"⚠️ Veri okuma hatası: {ex}")
        continue

    if len(df) > MAX_ROWS:
        start_time = time.time()

        # İndikatör hesaplamaları
        df["Time"] = df["Time"].apply(parse_time)
        df = df.sort_values("Time")
        df["Hour"] = df["Time"].dt.hour
        df["Minute"] = df["Time"].dt.minute

        apply_price_features(df)
        apply_rsi(df)
        apply_macd(df)
        apply_cci(df)
        apply_stoch(df)
        apply_ma(df)
        apply_ichimoku(df)
        apply_bollinger(df)
        apply_envelopes(df)

        apply_ind_momentum(df)

        close = df["Close"].iloc[-1]
        data_time = df["Time"].iloc[-1]

        df.drop(columns=clean_csv_cols + ["Hour", "Minute", "Time", "Close"], inplace=True)

        seq_data = df.iloc[-WINDOW_SIZE:].copy()

        if seq_data.isnull().values.any():
            print(seq_data)
            raise ValueError("❌ seq_data içinde NaN veya NaT değerler var!")

        if last_data_static != close:
            last_data_static = close

            # Giriş formatına dönüştür
            X_seq = seq_data.to_numpy().reshape(1, WINDOW_SIZE, -1)

            for t in range(WINDOW_SIZE):
                X_seq[:, t, :] = scaler.transform(X_seq[:, t, :])

            # Tahmin
            probs = model.predict(X_seq, verbose=0)[0]
            sell_prob = probs[1] * 100
            buy_prob = probs[2] * 100

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # === Tahmin çıktısı
            df_result = pd.DataFrame([{
                "Time": now_str,
                "BuyProb_Model1": round(buy_prob, 1),
                "SellProb_Model1": round(sell_prob, 1),
                "BuyProb_Model2": round(99.9, 1),
                "SellProb_Model2": round(99.9, 1)
            }])

            for i in range(1, 4):
                prediction_csv_path = f"{common_path}Live_V3_{i}.csv"
                lock_path = prediction_csv_path + ".lock"

                with FileLock(lock_path):
                    df_result.to_csv(prediction_csv_path, mode="w", header=True, index=False, encoding='utf-8')
                    print(f"✅ Dosya yazıldı: {prediction_csv_path}")

            print(f"\n📊 Model Prediction : {data_time}")
            print(f"Buy: {buy_prob:.2f}%, Sell: {sell_prob:.2f}%")
            print(f"⏱️ Process Time: {round((time.time() - start_time) * 1000)} ms\n")

            # === Log verisi
            log_data = {
                "LastDataTime": data_time,
                "PredictionTime": now_str,
                "BuyProb": round(buy_prob, 1),
                "SellProb": round(sell_prob, 1)
            }

            log_data.update(seq_data.iloc[-1].to_dict())
            df_log = pd.DataFrame([log_data])
            log_file_path = "Live_Log.csv"

            if not os.path.exists(log_file_path):
                df_log.to_csv(log_file_path, mode="w", header=True, index=False, encoding='utf-8')
            else:
                df_log.to_csv(log_file_path, mode="a", header=False, index=False, encoding='utf-8')

            print(f"📝 Log kaydedildi: {log_file_path}")

        # Son MAX_ROWS kadar veri tut
        for attempt in range(10):
            try:
                df = df.tail(MAX_ROWS).reset_index(drop=True)
                df.to_csv(path_data, index=False)
                break
            except Exception as e:
                print(f"⚠️ Kayıt denemesi {attempt + 1}/10 başarısız: {e}")
                time.sleep(0.25)
