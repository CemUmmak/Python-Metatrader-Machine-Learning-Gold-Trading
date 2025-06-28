import pandas as pd

def apply_bollinger(df: pd.DataFrame):
    # 1. Band genişliği
    df['boll_bandwidth'] = (df['BollUpper'] - df['BollLower']).round(2)

    # 2-4. Fiyat ile bant ilişkileri
    df['close_boll_upper_diff']  = (df['Close'] - df['BollUpper']).round(2)
    df['close_boll_lower_diff']  = (df['Close'] - df['BollLower']).round(2)
    df['close_boll_middle_diff'] = (df['Close'] - df['BollMiddle']).round(2)

    # 5-6. Bandların birbirine uzaklığı
    df['boll_upper_middle_diff'] = (df['BollUpper'] - df['BollMiddle']).round(2)
    df['boll_lower_middle_diff'] = (df['BollLower'] - df['BollMiddle']).round(2)

    # Orijinal değerleri istersen silebilirsin:
    df.drop(columns=['BollUpper', 'BollLower', 'BollMiddle'], inplace=True, errors='ignore')

    return df
