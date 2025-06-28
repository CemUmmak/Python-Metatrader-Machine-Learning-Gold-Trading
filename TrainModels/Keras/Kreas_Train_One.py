import joblib
from ML_Ind_V3.Settings import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional

# 📂 1. Inputs
data_path = path_csv_cleaned_nolast5000
model_no = 1006
window_size = 20 # 999 > 20

# 📄 2. Read Data
data_name = get_data_name_from_path(data_path)
df = pd.read_csv(data_path)
os.makedirs(os.path.join("Models", data_name), exist_ok=True)

df.drop(columns=["Time"], inplace=True)

# 🎯 3. Create sequences
X_raw = df.drop("ProfitLabel", axis=1).values
y_raw = df["ProfitLabel"].values

X_seq, y_seq = [], []
for i in range(window_size, len(X_raw)):
    X_seq.append(X_raw[i - window_size:i])
    y_seq.append(y_raw[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# 🧪 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
)

# ⚖️ 5. Scaling
scaler = StandardScaler()
num_features = X_train.shape[2]

# her timestep için scaler uyguluyoruz (fit sadece train’e)
X_train_scaled = np.empty_like(X_train)
X_test_scaled = np.empty_like(X_test)

for i in range(window_size):
    X_train_scaled[:, i, :] = scaler.fit_transform(X_train[:, i, :])
    X_test_scaled[:, i, :] = scaler.transform(X_test[:, i, :])

# 🔧 6. Model parametreleri
custom_params = {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "dropout_rate": 0.3,
    "lstm_units": [64, 32]
}

# 🧠 7. Model
model = Sequential()
model.add(LSTM(custom_params["lstm_units"][0], return_sequences=True, input_shape=(window_size, num_features)))
model.add(Dropout(custom_params["dropout_rate"]))
model.add(LSTM(custom_params["lstm_units"][1]))
model.add(Dropout(custom_params["dropout_rate"]))
model.add(Dense(len(np.unique(y_seq)), activation='softmax'))



"""model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(window_size, num_features)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.3))
model.add(Dense(len(np.unique(y_seq)), activation='softmax'))"""


optimizer = Adam(learning_rate=custom_params["learning_rate"])
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ⏱️ 8. Eğitim

model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=custom_params["epochs"],
    batch_size=custom_params["batch_size"],
    callbacks=[early_stop],
    verbose=1
)

# 🔮 9. Tahmin
y_pred = model.predict(X_test_scaled)
y_pred_labels = y_pred.argmax(axis=1)

# 📈 10. Performans
accuracy = accuracy_score(y_test, y_pred_labels)
precision = precision_score(y_test, y_pred_labels, average="macro")
recall = recall_score(y_test, y_pred_labels, average="macro")
f1 = f1_score(y_test, y_pred_labels, average="macro")

print("\n=== Model Evaluation Metrics ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\n🎯 Classification Report:")
print(classification_report(y_test, y_pred_labels, digits=4))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_labels))

for attempt in range(1, 11):
    try:
        model.save(f"Models/{data_name}/{data_name}_Model_{model_no}.keras")
        joblib.dump(scaler, f"Models/{data_name}/{data_name}_Scaler_{model_no}.pkl")
        df_params = pd.DataFrame([custom_params])
        df_params.to_csv(f"Models/{data_name}/{data_name}_Params_{model_no}.csv", index=False)

        print(f"✅ Model, Scaler ve Parametreler {attempt}. denemede başarıyla kaydedildi.")
        break
    except Exception as e:
        print(f"⚠️ {attempt}. denemede kayıt başarısız: {e}")
        time.sleep(5)

print(f"\n✅ LSTM model, Scaler ve parametreler 'Models' klasörüne kaydedildi.")
