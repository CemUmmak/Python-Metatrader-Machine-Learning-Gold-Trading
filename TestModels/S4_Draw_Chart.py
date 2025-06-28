import matplotlib.pyplot as plt
from ML_Ind_V3.Settings import *
import os

# --- Tek model için ayarlar ---
model_name1 = "XGB"
model_no1 = 19
model_prob1 = 65
data_path1 = path_csv_no_corr

# --- Çift model için ayarlar ---
model_name2 = "XGB"
model_no2 = 59
model_prob2 = 65
data_path2 = path_csv_no_corr

data_name1 = get_data_name_from_path(data_path1)
data_name2 = get_data_name_from_path(data_path2)

# === DOSYA YOLU AYARLARI ===
if model_name2 == "":

    file_path = f"{model_name1}/{model_name1}_{data_name1}/{data_name1}_{model_name1}_{model_no1}_{model_prob1}.csv"
    save_path = f"{model_name1}/{model_name1}_{data_name1}/{data_name1}_{model_name1}_{model_no1}_{model_prob1}_Chart.png"
    chart_title = f"{model_name1} | Model: {model_no1} | Prob: {model_prob1}"
else:
    top_folder = f"{model_name1}_{model_name2}"
    sub_folder = f"{model_name1}_{data_name1}___{model_name2}_{data_name2}"
    file_name = (f"{model_name1}_{model_no1}_{model_prob1}_{data_name1}__"
                 f"{model_name2}_{model_no2}_{model_prob2}_{data_name2}.csv")

    file_path = f"{top_folder}/{sub_folder}/{file_name}"
    save_path = f"{top_folder}/{sub_folder}/{file_name.replace('.csv', '_Chart.png')}"
    chart_title = f"{model_name1}({model_no1}/{model_prob1}) vs {model_name2}({model_no2}/{model_prob2})"

# === VERİYİ YÜKLE ===
if not os.path.exists(file_path):
    raise FileNotFoundError(f"⛔ Dosya bulunamadı: {file_path}")

df = pd.read_csv(file_path)

# === LINE CHART: Cumulative Profit ===
plt.figure(figsize=(12, 6))
plt.plot(df["CumulativeProfit"], label="Cumulative Profit", color="green", linewidth=2)
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.title(chart_title, fontsize=14)
plt.xlabel("Trade Index")
plt.ylabel("Cumulative Profit")
plt.grid(True)

# === MAXIMUM DRAWDOWN ===
cumulative_profit = df["CumulativeProfit"]
peak = cumulative_profit.cummax()
drawdown = peak - cumulative_profit
max_drawdown = drawdown.max()

# === METRİKLER ===
order_opened = (df["Action"] != 0).sum()
profit_count = (df["Profit"] > 0).sum()
loss_count = (df["Profit"] < 0).sum()
win_rate = (profit_count / order_opened * 100) if order_opened > 0 else 0

# Maksimum ardışık kayıp
consecutive_loss = 0
max_consecutive_loss = 0
for profit in df["Profit"]:
    if profit < 0:
        consecutive_loss += 1
        max_consecutive_loss = max(max_consecutive_loss, consecutive_loss)
    elif profit > 0:
        consecutive_loss = 0

# Skor hesapla
score = win_rate * (order_opened ** 0.5) / (1 + max_consecutive_loss)

# === TABLO VERİLERİ ===
metrics = {
    "OrderOpened": order_opened,
    "ProfitCount": profit_count,
    "LossCount": loss_count,
    "WinRate (%)": round(win_rate, 2),
    "MaxConsecLoss": max_consecutive_loss,
    "Score": round(score, 2),
    "Cumulative Profit": round(cumulative_profit.iloc[-1], 2),
    "MaxDrawdown": round(max_drawdown, 2)
}

table_data = [[k, str(v)] for k, v in metrics.items()]
plt.table(cellText=table_data,
          colLabels=["Metric", "Value"],
          cellLoc='center',
          colWidths=[0.2, 0.2],
          loc='upper left',
          bbox=[0.01, 0.5, 0.3, 0.45])

# === KAYDET VE GÖSTER ===
plt.grid(False)
plt.tight_layout()
plt.legend(loc="lower right")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"✅ Grafik kaydedildi: {save_path}")
