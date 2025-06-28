import time

from Settings import *

import pandas as pd
# import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 📂 1. Inputs
data_path = path_csv_no_corr_nolast5000
model_no = 999

# 2. Read Data
data_name = get_data_name_from_path(data_path)

df = pd.read_csv(data_path)

df = create_lag_features(df, n_lags=20, label_column="ProfitLabel")


os.makedirs(os.path.join("Models", data_name), exist_ok=True)

print(f"✅ Toplam satır sayısı : {df.shape[0]}")
print(f"✅ Toplam feature sayısı : {df.shape[1]}")

df.drop(columns=["Time"], inplace=True)

X = df.drop("ProfitLabel", axis=1)
y = df["ProfitLabel"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


custom_params = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 100,
    "subsample": 0.7,
    "colsample_bytree": 0.9,
    "gamma": 0.1,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0
}

print(time.time())
model = xgb.XGBClassifier(**custom_params)

# 🏋️ 6. Modeli eğit
model.fit(X_train_scaled, y_train)

# 🔮 7. Test verisi üzerinde tahmin yap
y_pred = model.predict(X_test_scaled)

# 📈 8. Performans metriklerini hesapla
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

print("\n=== Model Evaluation Metrics ===", time.time())
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\n🎯 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


joblib.dump(model, f"Models/{data_name}/{data_name}_XGB_Model_{model_no}.pkl")
joblib.dump(scaler, f"Models/{data_name}/{data_name}_XGB_Scaler_{model_no}.pkl")

# 💾 10. Model parametrelerini CSV olarak kaydet
df_params = pd.DataFrame([custom_params])
df_params.to_csv(f"Models/{data_name}/{data_name}_XGB_Params_{model_no}.csv", index=False)

print(f"\n✅ Model, Scaler ve Parametreler 'Models' klasörüne kaydedildi.")
