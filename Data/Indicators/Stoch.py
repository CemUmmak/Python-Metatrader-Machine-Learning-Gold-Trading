import numpy as np

def apply_stoch(df):
    def stoch_region(x):
        if x > 80:
            return 2
        elif x < 20:
            return 0
        else:
            return 1

    for i in range(1, 4):
        df[f"StochCross{i}"] = np.where(df[f"StochMain{i}"] > df[f"StochSignal{i}"], 1, 0)
        df[f"StochRegion{i}"] = df[f"StochMain{i}"].apply(stoch_region)

    # MACD values dropping in IndMomentums

    print(f"✅ Stochastic features generated successfully!")

    return df
