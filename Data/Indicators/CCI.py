def apply_cci(df):
    def cci_region(x):
        if x > 200:
            return 4  # Çok aşırı alım
        elif x > 150:
            return 3  # Aşırı alım
        elif x > 100:
            return 2  # Güçlü alım
        elif x > 0:
            return 1  # Hafif alım
        elif x > -100:
            return 0  # Hafif satım
        elif x > -150:
            return -1  # Güçlü satım
        elif x > -200:
            return -2  # Aşırı satım
        else:
            return -3  # Çok aşırı satım

    for i in range(1, 4):
        df[f"CciRegion{i}"] = df[f"Cci{i}"].apply(cci_region)

    # df.drop(columns=['Cci1', 'Cci2', 'Cci3'], inplace=True, errors='ignore')

    print(f"✅ CCI features generated successfully!")
    return df
