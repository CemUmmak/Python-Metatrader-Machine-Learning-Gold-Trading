from ML_Ind_V3.Settings import *

# === Ayar: diğer dosyaları sil !!
clear_others = True

# === Ayarlar ===
model1_name = "XGB"
data_path1 = path_csv_no_outlier


model2_name = ""
data_path2 = path_csv_no_corr_last5000


data_name1 = get_data_name_from_path(data_path1)
data_name2 = get_data_name_from_path(data_path2)

is_dual = model2_name != ""

if not is_dual:
    output_folder = f"{model1_name}/{model1_name}_{data_name1}"
    file_path = f"{output_folder}/{data_name1}_{model1_name}_All_Model_Test_Results.csv"
else:
    output_folder = f"{model1_name}_{model2_name}/{model1_name}_{data_name1}___{model2_name}_{data_name2}"

    file_path = f"{output_folder}/All_Results_{model1_name}_{data_name1}__{model2_name}_{data_name2}.csv"

# === CSV'yi oku ===
df = pd.read_csv(file_path, index_col="Index")
# === En iyi 10 model (Score) ===
top_by_score = df.sort_values(by="Score", ascending=False).head(10)

# === En iyi 10 model (WinRate) ===
top_by_winrate = df.sort_values(by="WinRate(%)", ascending=False).head(10)

# === Ortak modeller (index bazında) ===
common_indices = top_by_score.index.intersection(top_by_winrate.index)
common_models = df.loc[common_indices]

# === Fonksiyon: model detaylarını emojili yazdır ===
def print_model_block(df_to_print, title_emoji, title, is_dual=False,
                      model1_name="", model2_name="", data1_name="", data2_name=""):
    if is_dual:
        header_info = f"📊 {model1_name}_{data1_name} ⬄ {model2_name}_{data2_name}"
    else:
        header_info = f"📊 {model1_name}_{data1_name}"

    print(f"\n{title_emoji} {title} | {header_info}\n" + "=" * 160)

    for idx, row in df_to_print.iterrows():
        if is_dual:
            print(
                f"🔢Index: {idx:<4} | "
                f"🧠Model1: {int(row['Model1']):<3} | "
                f"🧠Model2: {int(row['Model2']):<3} | "
                f"🎯Prob1: {int(row['Prob1']):<3} | "
                f"🎯Prob2: {int(row['Prob2']):<3} | "
                f"📈Opened: {int(row['Orders']):<5} | "
                f"✅Profit: {int(row['ProfitCount']):<4} | "
                f"❌Loss: {int(row['LossCount']):<4} | "
                f"🏆WinRate: {row['WinRate(%)']:<6}% | "
                f"📉MaxLoss: {int(row['MaxConsecLoss']):<2} | "
                f"💰ProfitSum: {int(row['CumulativeProfit']):<5} | "
                f"📊Score: {int(row['Score'])}"
            )
        else:
            print(
                f"🔢Index: {idx:<4} | "
                f"🧠ModelNo: {int(row['ModelNo']):<3} | "
                f"🎯Prob: {int(row['ModelProb']):<3} | "
                f"📈Opened: {int(row['OrderOpened']):<5} | "
                f"✅Profit: {int(row['ProfitCount']):<4} | "
                f"❌Loss: {int(row['LossCount']):<4} | "
                f"🏆WinRate: {row['WinRate(%)']:<6}% | "
                f"📉MaxLoss: {int(row['MaxConsecLoss']):<2} | "
                f"💰ProfitSum: {int(row['CumulativeProfit']):<5} | "
                f"📊Score: {int(row['Score'])}"
            )


# === Yazdır ===
print_model_block(top_by_score, "🏅", "En İyi 10 Model (Score'a Göre)",
                  is_dual, model1_name, model2_name, data_name1, data_name2)

print_model_block(top_by_winrate, "🔥", "En İyi 10 Model (WinRate'e Göre)",
                  is_dual, model1_name, model2_name, data_name1, data_name2)

if not common_models.empty:
    print_model_block(common_models, "✅", "Her İki Listeye de Giren Ortak Modeller",
                      is_dual, model1_name, model2_name, data_name1, data_name2)
else:
    print("\n🚫 Ortak model bulunamadı.")

if is_dual:
    top_model_keys = set(
        (int(row["Model1"]), int(row["Prob1"]), int(row["Model2"]), int(row["Prob2"]))
        for _, row in top_by_score.iterrows()
    )
else:
    top_model_keys = set(
        (int(row["ModelNo"]), int(row["ModelProb"]))
        for _, row in top_by_score.iterrows()
    )

if clear_others:
    print(f"\n🧹 Gereksiz detay dosyalar temizleniyor...\n")

    if is_dual:
        print("\n📄 Ana sonuç dosyası da sadeleştiriliyor (ilk 10 model tutulacak)...")
        top10_df = df.sort_values(by="WinRate", ascending=False).head(10)
        top10_df.to_csv(file_path, index=True, index_label="Index")
        print("✅ Ana CSV dosyası temizlendi:", os.path.basename(file_path))

        folder_path = os.path.dirname(file_path)
        for fname in os.listdir(folder_path):
            if not fname.endswith(".csv"): continue
            if fname.startswith("All_Results"): continue

            match = re.match(rf"{model1_name}_(\d+)_(\d+)_.*__{model2_name}_(\d+)_(\d+)_.*\.csv", fname)
            if match:
                m1, p1, m2, p2 = map(int, match.groups())
                if (m1, p1, m2, p2) not in top_model_keys:
                    full_path = os.path.join(folder_path, fname)
                    os.remove(full_path)
                    print(f"🗑️ Silindi: {fname}")

    else:
        folder_path = os.path.dirname(file_path)
        for fname in os.listdir(folder_path):
            if not fname.endswith(".csv"): continue
            if fname.startswith("All_Model_Test_Results"): continue

            match = re.match(rf"{data_name1}_{model1_name}_(\d+)_(\d+)\.csv", fname)
            if match:
                m, p = map(int, match.groups())
                if (m, p) not in top_model_keys:
                    full_path = os.path.join(folder_path, fname)
                    os.remove(full_path)
                    print(f"🗑️ Silindi: {fname}")

        # 🎯 Model/scaler/parametre dosyaları da temizleniyor
        model_folder = f"{path_project}/TrainModels/{model1_name}/Models/{data_name1}/"
        if os.path.exists(model_folder):
            for fname in os.listdir(model_folder):
                if fname.endswith(".pkl") or fname.endswith(".csv"):
                    match = re.match(rf"{data_name1}_{model1_name}_(?:Model|Scaler|Params)_(\d+)", fname)
                    if match:
                        m = int(match.group(1))
                        if (m,) not in [(m1,) for (m1, p1) in top_model_keys]:  # sadece model no ile karşılaştır
                            full_path = os.path.join(model_folder, fname)
                            os.remove(full_path)
                            print(f"🗑️ Model/scaler/parametre silindi: {fname}")