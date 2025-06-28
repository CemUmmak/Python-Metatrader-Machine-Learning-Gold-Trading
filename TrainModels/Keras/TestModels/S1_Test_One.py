
import joblib

from keras.models import load_model
from ML_Ind_V3.Settings import *

# === Ayarlar ===
model_name = "Keras"
model_prob_list = [80, 85, 90, 95] # range(50,100,5)
custom_list = [1005]
start_model_no = 1
total_models = 240
data_path = path_csv_cleaned_last5000
window_size = 20  # LSTM pencere boyutu

# === Dosya yolları ===
data_name = get_data_name_from_path(data_path)
model_folder = f"{path_project}/TrainModels/{model_name}/Models/{data_name}/"
output_folder = f"{model_name}_{data_name}"
os.makedirs(output_folder, exist_ok=True)

# === Veriyi oku ===
df = pd.read_csv(data_path)
time_series = df["Time"].reset_index(drop=True)
df.drop(columns=["Time"], inplace=True)

results_all = []
start_time = time.time()
total_test = custom_list if len(custom_list) > 0 else range(start_model_no, total_models + 1)
total_tests_count = len(total_test) * len(model_prob_list)
test_counter = 0
data_count = len(df)

for model_no in total_test:
    try:
        model = load_model(f"{model_folder}{data_name}_Model_{model_no}.keras")
        scaler = joblib.load(f"{model_folder}{data_name}_Scaler_{model_no}.pkl")
    except Exception as ex:
        print(f"🚫 Model yüklenemedi: {ex}")
        continue

    for model_prob in model_prob_list:
        model_results = []
        order_opened = profit_count = loss_count = consecutive_loss = max_consecutive_loss = cumulative_profit = 0
        win_rate = 0.0

        for i in range(window_size, data_count):
            true_label = int(df.iloc[i]["ProfitLabel"])
            row_seq = df.iloc[i - window_size:i].drop(columns=["ProfitLabel"]).values
            row_scaled = scaler.transform(row_seq)
            row_input = np.expand_dims(row_scaled, axis=0)

            try:
                probabilities = model.predict(row_input, verbose=0)[0]
            except Exception as ex:
                print(f"🚫 Tahmin hatası: {ex}")
                continue

            sell_prob = probabilities[1] * 100
            buy_prob = probabilities[2] * 100

            action = 0
            if sell_prob > model_prob and sell_prob > buy_prob:
                action = 1
            elif buy_prob > model_prob and buy_prob > sell_prob:
                action = 2

            if action in [1, 2]:
                correct = (true_label == action)
                profit = 10 if correct else -10
                profit_count += int(correct)
                loss_count += int(not correct)
                consecutive_loss = 0 if correct else consecutive_loss + 1
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

            win_rate = (profit_count / order_opened * 100) if order_opened else 0
            print_inline_progress(i + 1, data_count, prefix=(
                f"🧪 {model_name} {model_no:3d} P:{model_prob:2d} | "
                f"📂 Orders: {order_opened:4d} | ✅ Profit: {profit_count:4d} | "
                f"❌ Loss: {loss_count:4d} | 🎯 WinRate: {win_rate:5.1f}% | "
                f"💥 MaxLoss: {max_consecutive_loss:3d}  "
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

        pd.DataFrame(model_results).to_csv(f"{output_folder}/{data_name}_{model_name}_{model_no}_{model_prob}.csv", index=True)

        result_path = f"{output_folder}/{data_name}_{model_name}_All_Model_Test_Results.csv"
        new_results_df = pd.DataFrame(results_all)

        if os.path.exists(result_path):
            old_results_df = pd.read_csv(result_path, index_col="Index")
            combined_df = pd.concat([old_results_df, new_results_df], ignore_index=True)
            combined_df.drop_duplicates(subset=["ModelNo", "ModelProb"], inplace=True)
        else:
            combined_df = new_results_df

        combined_df.sort_values(by="WinRate(%)", ascending=False).to_csv(result_path, index=True, index_label="Index")

        test_counter += 1
        elapsed = time.time() - start_time
        avg_time = elapsed / test_counter
        eta_minutes = (total_tests_count - test_counter) * avg_time / 60

        sys.stdout.write(f"\r✅ {model_name} {model_no:3d} P:{model_prob:2d} | "
                         f"🏆 Score: {score:3d} | 📂 Orders: {order_opened:4d} | ✅ Profit: {profit_count:4d} | "
                         f"❌ Loss: {loss_count:4d} | 🎯 WinRate: {win_rate:.1f}% |"
                         f" 💥 MaxLoss: {max_consecutive_loss:3d} | "
                         f"🧪 Test: {test_counter:3d}/{total_tests_count} | ⏱️ ETA: {eta_minutes:.1f} Min\n")
        sys.stdout.flush()

print(f"\n📄 Tüm sonuçlar kaydedildi: {model_name}_All_Model_Test_Results.csv")
