from S1_Sorte_Data import sorte_candles, sort_orders, merge_orders_with_candles
from S2_Create_Feature import  calculate_indicators
from S3_Outlier import detect_and_remove_outliers
from S4_Correlation import remove_corr
from Settings import *


# Step 1 - Sorte Candles
# sorte_candles(symbol)

# Step 2 - Sorte Orders
# sort_orders(symbol)

# Step 3 - Merge Raw Data
# merge_orders_with_candles(symbol)

# Step 4 - Calculate All Indicators
calculate_indicators(symbol)

#Step 5 - Clear Outliers
detect_and_remove_outliers(15)

# Step 6 - Clear Correlation
remove_corr(90)

print(" ")
df = pd.read_csv(path_csv_all)
print(path_csv_all)
print(f"✅ Toplam satır sayısı   : {df.shape[0]}")
print(f"✅ Toplam feature sayısı : {df.shape[1]}")
print(" ")

df = pd.read_csv(path_csv_no_corr)
print(path_csv_no_corr)
print(f"✅ Toplam satır sayısı   : {df.shape[0]}")
print(f"✅ Toplam feature sayısı : {df.shape[1]}")
print(" ")

df = pd.read_csv(path_csv_no_outlier)
print(path_csv_no_outlier)
print(f"✅ Toplam satır sayısı   : {df.shape[0]}")
print(f"✅ Toplam feature sayısı : {df.shape[1]}")
print(" ")

df = pd.read_csv(path_csv_cleaned)
print(path_csv_cleaned)
print(f"✅ Toplam satır sayısı   : {df.shape[0]}")
print(f"✅ Toplam feature sayısı : {df.shape[1]}")
print(" ")



