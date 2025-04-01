import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 產生假資料（模擬交易紀錄）
np.random.seed(42)
n = 1000
X = pd.DataFrame({
    '交易金額': np.random.uniform(5, 500, n),
    '評論數量': np.random.randint(1, 300, n),
    '退貨率': np.random.uniform(0, 0.6, n),
    '價格波動': np.random.uniform(-0.05, 0.05, n),
    '銷售波動性': np.random.uniform(0.1, 0.5, n),
    '評論變化率': np.random.uniform(-0.2, 0.3, n),
    '退貨率異常': np.random.choice([0, 1], size=n),
    '價格波動幅度': np.random.choice([0, 1], size=n)
})

# 模擬標籤（0=正常、1=可疑）
y = np.random.choice([0, 1], size=n, p=[0.8, 0.2])

# 切分訓練測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 儲存模型
joblib.dump(model, 'fraud_model.pkl')
print("✅ 模型已儲存為 fraud_model.pkl")
