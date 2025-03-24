import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# 設定標題
st.title("📊 商家風險數據分析與特徵構造")

# 生成模擬數據
np.random.seed(42)
num_records = 1000
merchant_ids = [f"merchant_{i}" for i in range(1, num_records + 1)]
transaction_amounts = np.random.uniform(5, 500, num_records)
review_counts = np.random.poisson(5, num_records)
return_rates = np.random.uniform(0, 0.2, num_records)
price_fluctuations = np.random.uniform(-0.05, 0.05, num_records)
labels = np.random.choice([0, 1], size=num_records, p=[0.8, 0.2])  # 80% 正常，20% 可疑

# 創建 DataFrame
df = pd.DataFrame({
    "商家 ID": merchant_ids,
    "交易金額": transaction_amounts,
    "評論數量": review_counts,
    "退貨率": return_rates,
    "價格波動": price_fluctuations,
    "風險狀態": labels
})

# 增強異常商家特徵
df.loc[df["風險狀態"] == 1, "退貨率"] = np.random.uniform(0.3, 0.6, df["風險狀態"].sum())
df.loc[df["風險狀態"] == 1, "評論數量"] = np.random.randint(50, 300, df["風險狀態"].sum())

# 數據特徵構造
df["銷售波動性"] = df["交易金額"].rolling(10).std().fillna(0) / df["交易金額"].rolling(10).mean().fillna(1)
df["評論變化率"] = df["評論數量"].pct_change().fillna(0)
df["退貨率異常"] = (df["退貨率"] > 0.25).astype(int)
df["價格波動幅度"] = abs(df["價格波動"]) > 0.03

# 轉換標籤
df["風險狀態"] = df["風險狀態"].map({0: "正常", 1: "可疑"})

# 顯示數據樣本
st.subheader("📋 數據樣本")
st.dataframe(df.head(50))

# 顯示新特徵的統計資訊
st.subheader("📈 特徵統計資訊")
st.write(df.describe())

# 分割數據集
X = df[["交易金額", "評論數量", "退貨率", "價格波動", "銷售波動性", "評論變化率", "退貨率異常"]]
y = (df["風險狀態"] == "可疑").astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.success("✅ 特徵構造完成，數據已準備好進行 AI 模型訓練！")
