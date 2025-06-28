import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from Settings import *

TOP_N = 30

# Hedef kolon
LABEL_COL = "ProfitLabel"

df = pd.read_csv(path_csv_all)
df = df.dropna()

X = df.drop(columns=[LABEL_COL, "Time", "TimeFilter"])
y = df[LABEL_COL]

model = xgb.XGBClassifier(eval_metric="logloss")
model.fit(X, y)



importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

# ================================
# 📈 Görselleştir
# ================================

plt.figure(figsize=(12, 8))
sns.barplot(x="Importance", y="Feature", data=importance_df.head(TOP_N))
plt.title(f"XGBoost Feature Importance (Top {TOP_N})")
plt.tight_layout()
plt.show()
