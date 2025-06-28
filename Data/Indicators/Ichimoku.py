import pandas as pd

def apply_ichimoku(df: pd.DataFrame):

    # 1. Tenkan - Kijun mesafesi
    df['tenkan_kijun_diff'] = (df['TenkanSen'] - df['KijunSen']).round(2)

    # 2. Span A - Span B mesafesi
    df['spanA_spanB_diff'] = (df['SenkouSpanA'] - df['SenkouSpanB']).round(2)

    # 3-6. Fiyat ile Ichimoku bileşenleri
    df['close_tenkan_diff'] = (df['Close'] - df['TenkanSen']).round(2)
    df['close_kijun_diff']  = (df['Close'] - df['KijunSen']).round(2)
    df['close_spanA_diff']  = (df['Close'] - df['SenkouSpanA']).round(2)
    df['close_spanB_diff']  = (df['Close'] - df['SenkouSpanB']).round(2)

    # Orijinal değerleri atmak istersen:
    df.drop(columns=['TenkanSen', 'KijunSen', 'SenkouSpanA', 'SenkouSpanB'], inplace=True, errors='ignore')

    return df
