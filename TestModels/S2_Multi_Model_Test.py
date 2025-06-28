import joblib
import time
from ML_Ind_V3.Settings import *

# === MANUEL AYARLAR ===
model1_name = "XGB"
model1_probs = [55, 60]
model1_list = [237, 239]
data_path1 = path_csv_no_corr_last5000

model2_name = "RF"
model2_probs = [51]
model2_list = [888, 999]
data_path2 = path_csv_cleaned_last5000


data_name1 = get_data_name_from_path(data_path1)
data_name2 = get_data_name_from_path(data_path2)

model1_folder = f"{path_project}/TrainModels/{model1_name}/Models/{data_name1}/"
model2_folder = f"{path_project}/TrainModels/{model2_name}/Models/{data_name2}/"

output_folder = f"{model1_name}_{model2_name}/{model1_name}_{data_name1}___{model2_name}_{data_name2}"
os.makedirs(output_folder, exist_ok=True)

# === VERİLERİ YÜKLE ===
df1 = pd.read_csv(data_path1)
df2 = pd.read_csv(data_path2)

# TIME UYUŞMAZSA TEST İPTAL
if not df1["Time"].equals(df2["Time"]):
    print("❌ Time kolonları uyuşmuyor, test atlandı.")
    exit()

time_series = df1["Time"].copy().reset_index(drop=True)
df1.drop(columns=["Time"], inplace=True)
df2.drop(columns=["Time"], inplace=True)

results_all = []
tested_pairs = set()
test_counter = 0

if model1_name == model2_name:
    model_pairs = [
        (a, b) for i, a in enumerate(model1_list)
        for b in model2_list[i:]
        if a != b  # aynı model numarasıyla test yapılmasın
    ]
else:
    model_pairs = [(a, b) for a in model1_list for b in model2_list]

total_tests = len(model_pairs) * len(model1_probs) * len(model2_probs)

print(f"Total Tests : {total_tests}")

start_time = time.time()

for model1_no in model1_list:
    for model2_no in model2_list:

        trades = []

        if model1_name == model2_name and ((model2_no, model1_no) in tested_pairs or model1_no == model2_no):
            continue

        tested_pairs.add((model1_no, model2_no))

        try:
            model1 = joblib.load(f"{model1_folder}{data_name1}_{model1_name}_Model_{model1_no}.pkl")
            scaler1 = joblib.load(f"{model1_folder}{data_name1}_{model1_name}_Scaler_{model1_no}.pkl")

            model2 = joblib.load(f"{model2_folder}{data_name2}_{model2_name}_Model_{model2_no}.pkl")
            scaler2 = joblib.load(f"{model2_folder}{data_name2}_{model2_name}_Scaler_{model2_no}.pkl")

            for prob1 in model1_probs:
                for prob2 in model2_probs:
                    order_opened = 0
                    profit_count = 0
                    loss_count = 0
                    cumulative_profit = 0
                    consecutive_loss = 0
                    max_consec_loss = 0
                    win_rate = 0

                    for i in range(len(df1)):
                        true_label = int(df1.loc[i, "ProfitLabel"])
                        X1 = df1.drop(columns=["ProfitLabel"]).iloc[[i]]
                        X2 = df2.drop(columns=["ProfitLabel"]).iloc[[i]]

                        prob1_pred = model1.predict_proba(scaler1.transform(X1))[0]
                        prob2_pred = model2.predict_proba(scaler2.transform(X2))[0]

                        sell1, buy1 = prob1_pred[1] * 100, prob1_pred[2] * 100
                        sell2, buy2 = prob2_pred[1] * 100, prob2_pred[2] * 100

                        action = 0
                        if sell1 > prob1 and sell2 > prob2:
                            action = 1
                        elif buy1 > prob1 and buy2 > prob2:
                            action = 2

                        if action == 1:
                            profit = 10 if true_label == 1 else -10
                        elif action == 2:
                            profit = 10 if true_label == 2 else -10
                        else:
                            profit = 0

                        if profit < 0:
                            loss_count += 1
                            consecutive_loss += 1
                            max_consec_loss = max(max_consec_loss, consecutive_loss)
                        elif profit > 0:
                            profit_count += 1
                            consecutive_loss = 0

                        if action > 0:
                            order_opened += 1

                        cumulative_profit += profit

                        trades.append({
                            "Time": time_series[i],
                            "Index": i,
                            "Action": action,
                            "TrueLabel": true_label,
                            "Sell1": round(sell1, 2),
                            "Buy1": round(buy1, 2),
                            "Sell2": round(sell2, 2),
                            "Buy2": round(buy2, 2),
                            "Profit": profit,
                            "CumulativeProfit": cumulative_profit
                        })
                        win_rate = (profit_count / order_opened * 100) if order_opened > 0 else 0

                        print_inline_progress(i + 1, len(df1), prefix=(
                            f"🧪 {model1_name} {model1_no:3d} P: {prob1:5.2f} | "
                            f"{model2_name} {model2_no:3d} P: {prob2:5.2f} | "
                            f"📂 Orders: {order_opened:3d} | ✅ Profit: {profit_count:3d} | "
                            f"❌ Loss: {loss_count:3d} | 🎯 WinRate: {win_rate:.2f}%"
                        ))

                    score = win_rate * (order_opened ** 0.5) / (1 + max_consec_loss)
                    score = max(0, int(score))

                    results_all.append({
                        "Model1": model1_no, "Model2": model2_no,
                        "Prob1": prob1, "Prob2": prob2,
                        "Orders": order_opened,
                        "WinRate(%)": round(win_rate, 2),
                        "MaxConsecLoss": max_consec_loss,
                        "ProfitCount": profit_count,
                        "LossCount": loss_count,
                        "CumulativeProfit": cumulative_profit,
                        "Score": score
                    })

                    # CSV Kaydet
                    test_counter += 1
                    avg = (time.time() - start_time) / test_counter
                    eta = (total_tests - test_counter) * avg / 60

                    sys.stdout.write(f"\r✅ {model1_name} {model1_no:3d} P: {prob1:5.2f} | "
                                     f"{model2_name} {model2_no:3d} P: {prob2:5.2f} | "
                                     f"🏆 Score: {score:4d} | 📂 Orders: {order_opened:3d} | ✅ Profit: {profit_count:3d} | "
                                     f"❌ Loss: {loss_count:3d} | 🎯 WinRate: {win_rate:.2f}% | "
                                     f"💥 MaxLoss: {max_consec_loss:2d} | 🧪 Test: {test_counter}/{total_tests} | "
                                     f"⏱️ ETA: {eta:.1f} Min")
                    sys.stdout.flush()

                    # 📄 Detaylı işlem CSV dosyası
                    filename = f"{model1_name}_{model1_no}_{prob1}_{data_name1}__{model2_name}_{model2_no}_{prob2}_{data_name2}.csv"
                    pd.DataFrame(trades).to_csv(f"{output_folder}/{filename}", index=True, index_label="Index")

                    # 📄 Sonuç dosyasının yolu
                    result_path = f"{output_folder}/All_Results_{model1_name}_{data_name1}__{model2_name}_{data_name2}.csv"
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
                    combined_df.sort_values(by="Score", ascending=False).to_csv(result_path, index=True,
                                                                                index_label="Index")

        except Exception as e:
            print(f"❌ Model yükleme hatası: {model1_name}-{model1_no} + {model2_name}-{model2_no}: {e}")
