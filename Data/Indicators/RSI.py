

def apply_rsi(df, digits=2):
    # RSI farkları ve oranları
    df["RSI7_minus_RSI14"] = (df["Rsi7"] - df["Rsi14"]).round(digits)
    df["RSI7_minus_RSI21"] = (df["Rsi7"] - df["Rsi21"]).round(digits)
    df["RSI14_minus_RSI21"] = (df["Rsi14"] - df["Rsi21"]).round(digits)

    df["RSI7_div_RSI14"] = (df["Rsi7"] / df["Rsi14"]).round(digits)
    df["RSI7_div_RSI21"] = (df["Rsi7"] / df["Rsi21"]).round(digits)
    df["RSI14_div_RSI21"] = (df["Rsi14"] / df["Rsi21"]).round(digits)

    # RSI bölgesi (5 bölge, numeric)
    def rsi_region(value):
        if value < 20:
            return 0  # Çok aşırı satım
        elif value < 40:
            return 1  # Aşırı satım
        elif value < 60:
            return 2  # Nötr
        elif value < 80:
            return 3  # Aşırı alım
        else:
            return 4  # Çok aşırı alım

    df["RSI7_region"] = df["Rsi7"].apply(rsi_region)
    df["RSI14_region"] = df["Rsi14"].apply(rsi_region)
    df["RSI21_region"] = df["Rsi21"].apply(rsi_region)

    # RSI sıralama (numeric kod)
    def rsi_order_numeric(row):
        rsi_values = {"RSI7": row["Rsi7"], "RSI14": row["Rsi14"], "RSI21": row["Rsi21"]}
        sorted_rsi = sorted(rsi_values.items(), key=lambda x: x[1], reverse=True)
        order = ">".join([name for name, value in sorted_rsi])

        # 6 olasılık
        if order == "RSI7>RSI14>RSI21":
            return 0
        elif order == "RSI7>RSI21>RSI14":
            return 1
        elif order == "RSI14>RSI7>RSI21":
            return 2
        elif order == "RSI14>RSI21>RSI7":
            return 3
        elif order == "RSI21>RSI7>RSI14":
            return 4
        elif order == "RSI21>RSI14>RSI7":
            return 5
        else:
            return -1

    df["RSI_order"] = df.apply(rsi_order_numeric, axis=1)

    # Rsi values dropping in IndMomentums

    print(f"✅ RSI features generated successfully!")
    return df

