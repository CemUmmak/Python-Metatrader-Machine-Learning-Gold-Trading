import numpy as np


def apply_macd(df, digits = 2):
    for i in range(1, 4):
        df[f"MacdHist{i}"] = (df[f"MacdLine{i}"] - df[f"MacdCandle{i}"]).round(digits)
        df[f"MacdCross{i}"] = np.where(df[f"MacdLine{i}"] > df[f"MacdCandle{i}"], 1, 0)
        df[f"MacdAboveZero{i}"] = (df[f"MacdLine{i}"] > 0).astype(int)
        df[f"MacdSlope{i}"] = (df[f"MacdLine{i}"] - df[f"MacdLine{i}"].shift(1)).round(digits)

    # MACD values dropping in IndMomentums

    print(f"✅ MACD features generated successfully!")
    return df
