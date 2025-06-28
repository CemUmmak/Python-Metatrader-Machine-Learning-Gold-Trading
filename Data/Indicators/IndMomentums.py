import numpy as np


def apply_ind_momentum(df):
    rsi_names = {1: "Rsi7", 2: "Rsi14", 3: "Rsi21"}

    for i in range(1, 4):
        rsi_col = rsi_names[i]

        # --- RSI ---
        df[f"{rsi_col}_momentum_1"] = np.where(df[rsi_col] > df[rsi_col].shift(1), 1, 0)
        df[f"{rsi_col}_momentum_2"] = np.where(
            (df[rsi_col] > df[rsi_col].shift(1)) & (df[rsi_col].shift(1) > df[rsi_col].shift(2)), 1, 0)
        df[f"{rsi_col}_momentum_3"] = np.where(
            (df[rsi_col] > df[rsi_col].shift(1)) &
            (df[rsi_col].shift(1) > df[rsi_col].shift(2)) &
            (df[rsi_col].shift(2) > df[rsi_col].shift(3)),
            1, 0)



        # --- MACD Line ---
        df[f"MacdLine{i}_momentum_1"] = np.where(df[f"MacdLine{i}"] > df[f"MacdLine{i}"].shift(1), 1, 0)
        df[f"MacdLine{i}_momentum_2"] = np.where((df[f"MacdLine{i}"] > df[f"MacdLine{i}"].shift(1)) & (
                    df[f"MacdLine{i}"].shift(1) > df[f"MacdLine{i}"].shift(2)), 1, 0)
        df[f"MacdLine{i}_momentum_3"] = np.where(
            (df[f"MacdLine{i}"] > df[f"MacdLine{i}"].shift(1)) &
            (df[f"MacdLine{i}"].shift(1) > df[f"MacdLine{i}"].shift(2)) &
            (df[f"MacdLine{i}"].shift(2) > df[f"MacdLine{i}"].shift(3)),
            1, 0)



        # --- Stochastic Main ---
        df[f"StochMain{i}_momentum_1"] = np.where(df[f"StochMain{i}"] > df[f"StochMain{i}"].shift(1), 1, 0)
        df[f"StochMain{i}_momentum_2"] = np.where((df[f"StochMain{i}"] > df[f"StochMain{i}"].shift(1)) & (
                    df[f"StochMain{i}"].shift(1) > df[f"StochMain{i}"].shift(2)), 1, 0)
        df[f"StochMain{i}_momentum_3"] = np.where(
            (df[f"StochMain{i}"] > df[f"StochMain{i}"].shift(1)) &
            (df[f"StochMain{i}"].shift(1) > df[f"StochMain{i}"].shift(2)) &
            (df[f"StochMain{i}"].shift(2) > df[f"StochMain{i}"].shift(3)),
            1, 0)

    # df.drop(columns=['Rsi7', 'Rsi14', 'Rsi21'], inplace=True, errors='ignore')

    # df.drop(columns=['MacdLine1', 'MacdCandle1', 'MacdLine2', 'MacdCandle2', 'MacdLine3', 'MacdCandle3'],
            # inplace=True, errors='ignore')

    # df.drop(columns=['StochMain1', 'StochSignal1', 'StochMain2', 'StochSignal2', 'StochMain3', 'StochSignal3'],
            # inplace=True, errors='ignore')

    print(f"✅ RSI, MACD Line ve Stochastic momentum (1-2-3 mum) features generated successfully!")
    return df
