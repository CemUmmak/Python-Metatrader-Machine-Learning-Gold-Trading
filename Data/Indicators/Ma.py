import pandas as pd





def apply_ma(df: pd.DataFrame):
    # 1. MA Sıralaması
    ma_columns = ['Ma7', 'Ma14', 'Ma21']

    # 2. MA’lar arası mesafe
    df['ma_diff_7_14'] = (df['Ma7'] - df['Ma14']).round(2)
    df['ma_diff_14_21'] = (df['Ma14'] - df['Ma21']).round(2)
    df['ma_diff_7_21'] = (df['Ma7'] - df['Ma21']).round(2)

    # MA ile kapanış fiyatı arası fark
    df['ma7_close_diff'] = (df['Ma7'] - df['Close']).round(2)
    df['ma14_close_diff'] = (df['Ma14'] - df['Close']).round(2)
    df['ma21_close_diff'] = (df['Ma21'] - df['Close']).round(2)

    # Unique olmayanları bırak, MA değerlerini at
    df.drop(columns=ma_columns, inplace=True, errors='ignore')

    return df
