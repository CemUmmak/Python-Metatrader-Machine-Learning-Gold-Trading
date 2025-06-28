import joblib
import time
from ML_Ind_V3.Settings import *

# === Ayarlar ===
model_name = "RF"
model_prob_list = [52, 55]
start_model_no = 1
total_models = 240
custom_list = [888]
data_path = path_csv_cleaned_last5000


data_name = get_data_name_from_path(data_path)

model_folder = f"{path_project}/TrainModels/{model_name}/Models/{data_name}/"

# === Klasör kontrolü ===
output_folder = f"{model_name}/{model_name}_{data_name}"
os.makedirs(output_folder, exist_ok=True)

# === Veriyi oku ===
df = pd.read_csv(data_path)

time_series = df["Time"].reset_index(drop=True)
df.drop(columns=["Time"], inplace=True)

results_all = []

# === Zaman ve test sayaçları ===
start_time = time.time()
total_test = custom_list if len(custom_list) > 0 else range(start_model_no, total_models + 1)
total_tests_count = len(total_test) * len(model_prob_list)
test_counter = 0
data_count = len(df)

# === Model ve eşik döngüsü ===
for model_no in total_test:

    model_results = []

    try:
        model = joblib.load(f"{model_folder}{data_name}_{model_name}_Model_{model_no}.pkl")
        scaler = joblib.load(f"{model_folder}{data_name}_{model_name}_Scaler_{model_no}.pkl")
    except Exception as ex:
        print(ex)
        continue

    for model_prob in model_prob_list:
        order_opened = 0
        profit_count = 0
        loss_count = 0
        consecutive_loss = 0
        max_consecutive_loss = 0
        cumulative_profit = 0
        win_rate = 0

        for i in range(data_count):
            row = df.iloc[[i]].copy()
            true_label = int(row["ProfitLabel"].values[0])
            X = row.drop(columns=["ProfitLabel"])
            X_scaled = scaler.transform(X)
            probabilities = model.predict_proba(X_scaled)[0]

            sell_prob = probabilities[1] * 100
            buy_prob = probabilities[2] * 100

            action = 0
            if sell_prob > model_prob and sell_prob > buy_prob:
                action = 1
            elif buy_prob > model_prob and buy_prob > sell_prob:
                action = 2

            if action == 1:
                if true_label == 1:
                    profit = 10
                    profit_count += 1
                    consecutive_loss = 0
                else:
                    profit = -10
                    loss_count += 1
                    consecutive_loss += 1
                    max_consecutive_loss = max(max_consecutive_loss, consecutive_loss)
                order_opened += 1

            elif action == 2:
                if true_label == 2:
                    profit = 10
                    profit_count += 1
                    consecutive_loss = 0
                else:
                    profit = -10
                    loss_count += 1
                    consecutive_loss += 1
                    max_consecutive_loss = max(max_consecutive_loss, consecutive_loss)
                order_opened += 1
            else:
                profit = 0

            cumulative_profit += profit

            model_results.append({
                "Time": time_series[i],
                "DataName": data_name,
                "ModelName": model_name,
                "ModelNo": model_no,
                "ModelProb": model_prob,
                "Index": i,
                "TrueLabel": true_label,
                "SellProb(%)": round(sell_prob, 2),
                "BuyProb(%)": round(buy_prob, 2),
                "Action": action,
                "Profit": profit,
                "CumulativeProfit": cumulative_profit
            })
            win_rate = (profit_count / order_opened * 100) if order_opened > 0 else 0

            print_inline_progress(i + 1, data_count, prefix=(
                f"🧪 {model_name} {model_no:3d} P:{model_prob:2d} | "
                f"📂 Orders: {int(order_opened):4d} | ✅ Profit: {int(profit_count):4d} | "
                f"❌ Loss: {int(loss_count):4d} | 🎯 WinRate: {win_rate:5.1f}% | "
                f"💥 MaxLoss: {int(max_consecutive_loss):3d}  "
            ))

        score = win_rate * (order_opened ** 0.5) / (1 + max_consecutive_loss)
        score = max(0, int(score))

        results_all.append({
            "ModelNo": model_no,
            "ModelProb": model_prob,
            "OrderOpened": order_opened,
            "ProfitCount": profit_count,
            "LossCount": loss_count,
            "WinRate(%)": round(win_rate, 2),
            "MaxConsecLoss": max_consecutive_loss,
            "CumulativeProfit": cumulative_profit,
            "Score": round(score, 2)
        })

        # 📁 Tek model-prob sonucu CSV
        single_result_df = pd.DataFrame(model_results)
        single_result_path = f"{output_folder}/{data_name}_{model_name}_{model_no}_{model_prob}.csv"
        single_result_df.to_csv(single_result_path, index=True, index_label="Index")

        # 📄 Sonuç dosyasının yolu
        result_path = f"{output_folder}/{data_name}_{model_name}_All_Model_Test_Results.csv"

        # 📊 Yeni sonuç DataFrame'i
        new_results_df = pd.DataFrame(results_all)

        # 🧠 Eski sonuçlar varsa oku, yeniyle birleştir, Score'a göre sırala
        if os.path.exists(result_path):
            old_results_df = pd.read_csv(result_path, index_col="Index")
            combined_df = pd.concat([old_results_df, new_results_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["ModelNo", "ModelProb"])
        else:
            combined_df = new_results_df

        # 🏆 Score'a göre sırala ve kaydet
        combined_df.sort_values(by="Score", ascending=False).to_csv(result_path, index=True, index_label="Index")

        # ⏱️ Süre tahmini
        test_counter += 1
        elapsed = time.time() - start_time
        avg_time = elapsed / test_counter
        remaining_tests = total_tests_count - test_counter
        eta_minutes = (remaining_tests * avg_time) / 60

        sys.stdout.write(f"\r✅ {model_name} {model_no:3d} P:{model_prob:2d} | "
                         f"🏆 Score: {score:3d} | 📂 Orders: {order_opened:4d} | ✅ Profit: {profit_count:4d} | "
                         f"❌ Loss: {loss_count:4d} | 🎯 WinRate: {win_rate:.1f}% |"
                         f" 💥 MaxLoss: {max_consecutive_loss:3d} | "
                         f"🧪 Test: {test_counter:3d}/{total_tests_count} | ⏱️ ETA: {eta_minutes:.1f} Min\n")
        sys.stdout.flush()

# 🏁 Final bilgi
print(f"\n📄 Tüm sonuçlar kaydedildi: {model_name}_All_Model_Test_Results.csv")
