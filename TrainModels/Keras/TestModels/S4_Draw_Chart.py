import matplotlib.pyplot as plt
import pandas as pd
import os

from ML_Ind_V3.Settings import *

# --- Model ayarları ---
model_name = "Keras"
model_no = 999
model_prob = 95
data_path = path_csv_cleaned_last5000

data_name = get_data_name_from_path(data_path)
file_path = f"{model_name}_{data_name}/{data_name}_{model_name}_{model_no}_{model_prob}.csv"
save_path = f"{model_name}_{data_name}/{data_name}_{model_name}_{model_no}_{model_prob}_Chart.png"
chart_title = f"{model_name} | Model: {model_no} | Prob: {model_prob}"

# === Dosya kontrolü ===
if not os.path.exists(file_path):
    raise FileNotFoundError(f"⛔ Dosya bulunamadı: {file_path}")

df = pd.read_csv(file_path)

# === Çizim: Cumulative Profit ===
plt.figure(figsize=(12, 6))
plt.plot(df["CumulativeProfit"], label="Cumulative Profit", color="green", linewidth=2)
plt.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.title(chart_title, fontsize=14)
plt.xlabel("Trade Index")
plt.ylabel("Cumulative Profit")
plt.grid(True)

# === Max Drawdown Hesabı ===
cumulative_profit = df["CumulativeProfit"]
peak = cumulative_profit.cummax()
drawdown = peak - cumulative_profit
max_drawdown = drawdown.max()

# === Metrikler ===
order_opened = (df["Action"] != 0).sum()
profit_count = (df["Profit"] > 0).sum()
loss_count = (df["Profit"] < 0).sum()
win_rate = (profit_count / order_opened * 100) if order_opened > 0 else 0

consecutive_loss = 0
max_consecutive_loss = 0
for profit in df["Profit"]:
    if profit < 0:
        consecutive_loss += 1
        max_consecutive_loss = max(max_consecutive_loss, consecutive_loss)
    elif profit > 0:
        consecutive_loss = 0

score = win_rate * (order_opened ** 0.5) / (1 + max_consecutive_loss)

# === Tablo ===
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

# === Kaydet ve Göster ===
plt.grid(False)
plt.tight_layout()
plt.legend(loc="lower right")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"✅ Grafik kaydedildi: {save_path}")
