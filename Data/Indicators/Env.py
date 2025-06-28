import pandas as pd

def apply_envelopes(df: pd.DataFrame):

    df['close_env_upper_diff']  = (df['Close'] - df['EnvUpper']).round(2)
    df['close_env_lower_diff']  = (df['Close'] - df['EnvLower']).round(2)

    # Orijinal değerleri istersen kaldır
    df.drop(columns=['EnvUpper', 'EnvLower'], inplace=True, errors='ignore')

    return df
