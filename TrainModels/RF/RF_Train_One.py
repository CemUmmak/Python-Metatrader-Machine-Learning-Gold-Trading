from ML_Ind_V3.Settings import *

import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 📂 1. Inputs
data_path = path_csv_cleaned_nolast5000
model_no = 888

# 2. Read Data
data_name = get_data_name_from_path(data_path)

df = pd.read_csv(data_path)
# df = create_lag_features(df, n_lags=20, label_column="ProfitLabel")

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

"""custom_params = {
    "n_estimators": 300,          # Daha fazla ağaç → daha istikrarlı sonuçlar
    "criterion": "gini",          # Gini impurity
    "max_depth": 6,               # Ağaçların maksimum derinliği: overfitting engeller
    "min_samples_split": 5,       # Bir düğümü bölmek için minimum 5 örnek olsun
    "min_samples_leaf": 3,        # Yaprakta minimum 3 örnek → overfit önler
    "min_weight_fraction_leaf": 0.0,
    "max_features": "sqrt",       # Default zaten bu → her split'te karekök(feature) seçilir
    "max_leaf_nodes": None,       # Bırakıyoruz, gerekirse sonra kısıtlarız
    "bootstrap": True,            # Bootstrap açık → çeşitlilik sağlar
    "oob_score": False,           # Şu anda oob kapalı, dilersen açarız
    "n_jobs": -1,                 # Full CPU kullanımı
    "random_state": 42,           # Sabit random
    "verbose": 0,
    "class_weight": "balanced_subsample"  # ⚡ Class imbalance için küçük ayar (sample içi dengeleme)
}"""

custom_params = {
    "n_estimators": 1200,              # Çok sayıda ağaç → istikrar ve düşük varyans
    "criterion": "entropy",            # Bilgi kazancı odaklı, daha keskin kararlar
    "max_depth": 18,                   # Derinlik yüksek → detaylı öğrenme
    "min_samples_split": 10,           # Daha büyük alt kümeler → emin olmadan bölme
    "min_samples_leaf": 5,             # Yaprakta en az 5 örnek → noise azaltma
    "max_features": 0.3,               # Her split’te daha az feature bak → çeşitlilik
    "bootstrap": True,
    "oob_score": True,
    "n_jobs": -1,
    "random_state": 42,
    "class_weight": "balanced_subsample",
    "verbose": 0
}


model = RandomForestClassifier(**custom_params)

# 🏋️ 6. Modeli eğit
model.fit(X_train_scaled, y_train)

# 🔮 7. Test verisi üzerinde tahmin yap
y_pred = model.predict(X_test_scaled)

# 📈 8. Performans metriklerini hesapla
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")

print("\n=== Model Evaluation Metrics ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\n🎯 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, f"Models/{data_name}/{data_name}_RF_Model_{model_no}.pkl")
joblib.dump(scaler, f"Models/{data_name}/{data_name}_RF_Scaler_{model_no}.pkl")

# 💾 10. Model parametrelerini CSV olarak kaydet
df_params = pd.DataFrame([custom_params])
df_params.to_csv(f"Models/{data_name}/{data_name}_RF_Params_{model_no}.csv", index=False)

print(f"\n✅ Model, Scaler ve Parametreler 'Models' klasörüne kaydedildi.")
