import xgboost as xgb
import joblib
import itertools
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from ML_Ind_V3.Settings import *

# 📂 1. Inputs
data_path = path_csv_no_corr_nolast5000
f1_score_target = 0.6

# 📄 2. Veri Yükle
data_name = get_data_name_from_path(data_path)
df = pd.read_csv(data_path)
# df = create_lag_features(df, n_lags=20, label_column="ProfitLabel")

os.makedirs(os.path.join("Models", data_name), exist_ok=True)

df.drop(columns=["Time"], inplace=True)

X = df.drop("ProfitLabel", axis=1)
y = df["ProfitLabel"]

# 🔀 3. Eğitim ve test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔄 4. Standardizasyon
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔧 5. Parametre kombinasyonları
param_grid = {
    "max_depth": [5, 6],
    "learning_rate": [0.04, 0.05, 0.06],
    "n_estimators": [1250, 1500, 2000],
    "subsample": [0.65, 0.7, 0.75],
    "colsample_bytree": [0.85, 0.9],
    "gamma": [0.0, 0.1, 0.2],
    "reg_alpha": [0.5, 0.6, 0.7],
    "reg_lambda": [1]
}

all_combinations = list(itertools.product(*param_grid.values()))
param_names = list(param_grid.keys())

successful_models = []
model_idx = 1
start_time = time.time()
total_tests = len(all_combinations)

# 🚀 6. Döngü ile her kombinasyonu dene
for test_idx, combo in enumerate(all_combinations, start=1):
    params = dict(zip(param_names, combo))

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        # n_jobs = -1,
        **params
    )

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred, average="macro")

    if f1 > f1_score_target:
        model_path = f"Models/{data_name}/{data_name}_XGB_Model_{model_idx}.pkl"
        scaler_path = f"Models/{data_name}/{data_name}_XGB_Scaler_{model_idx}.pkl"

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        # 📊 Her döngüde başarılı modelleri kaydet
        result_row = {
            "Model_No": model_idx,
            "F1_Score": round(f1, 2),
            **params
        }

        successful_models.append(result_row)

        result_row["Model_No"] = model_idx
        successful_models.append(result_row)


        pd.DataFrame(successful_models).sort_values(by="F1_Score", ascending=False).to_csv(
            f"Models/{data_name}/{data_name}_XGB_Top_Params.csv", index=False
        )

        print(f"✅ Model {model_idx} kaydedildi - F1 Score: {round(f1, 4)}")
        model_idx += 1
    else:
        print(f"❌ F1 Score düşük ({round(f1, 4)}), model kaydedilmedi.")

    # ⏱️ Süre ve kalan tahmini
    elapsed = time.time() - start_time
    avg_time = elapsed / test_idx
    remaining_tests = total_tests - test_idx
    eta_minutes = (remaining_tests * avg_time) / 60

    print(f"⏱️ Ortalama süre/test: {avg_time:.2f} sn | Tahmini kalan süre: {eta_minutes:.1f} dk | "
          f"Test {test_idx}/{total_tests}\n")
