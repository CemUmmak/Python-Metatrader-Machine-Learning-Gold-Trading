import sys
import  importlib

print(sys.version)

import sklearn
print(sklearn.__version__)
exit()

# 📦 Yüklemek istediğin kütüphaneleri buraya yaz
packages = [
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "xgboost"
    "filelock"
]

for package in packages:
    try:
        importlib.import_module(package)
        print(f"✅ Zaten yüklü: {package}")
    except ImportError:
        print(f"⬇️ Yükleniyor: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])