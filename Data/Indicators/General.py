import numpy as np

def apply_price_features(df):
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    print(f"✅ All general features generated successfully!")
