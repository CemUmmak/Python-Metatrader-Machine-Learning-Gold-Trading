from S1_Sorte_Data import merged_folder, ready_folder
from Settings import *

import pandas as pd
from Indicators.General import apply_price_features
from Indicators.RSI import apply_rsi
from Indicators.MACD import apply_macd
from Indicators.CCI import apply_cci
from Indicators.Stoch import apply_stoch
from Indicators.IndMomentums import apply_ind_momentum
from Indicators.Ma import apply_ma
from Indicators.Ichimoku import apply_ichimoku
from Indicators.BB import apply_bollinger
from Indicators.Env import apply_envelopes

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

def calculate_indicators(symbol):
    df = pd.read_csv(merged_folder + f"{symbol}_Merged.csv")

    #Indicators
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

    #
    # df.drop(columns=["Hour", "Minute", "Close", "Profit", "OpenPrice"], inplace=True)
    # df.drop(columns=["Hour", "Minute", "Profit", "OpenPrice", "ProfitLabel"], inplace=True)
    df.drop(columns=["Hour", "Minute", "Profit", "OpenPrice"], inplace=True)

    df.dropna(inplace=True)

    print("Saving..")
    df.to_csv(path_csv_all, index=False)
    df.tail(5000).to_csv(path_csv_all_last5000, index=False)
    df.iloc[:-5000].to_csv(path_csv_all_nolast5000, index=False)
    print("Saved..\n")

# calculate_indicators("XAUUSD")