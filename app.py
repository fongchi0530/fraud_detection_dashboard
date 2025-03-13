import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 設定標題
st.title("📊 商家風險數據分析儀表板")

# 生成模擬數據
np.random.seed(42)
num_records = 1000  # 數據筆數
merchant_ids = [f"merchant_{i}" for i in range(1, num_records + 1)]
product_ids = [f"product_{i % 100 + 1}" for i in range(num_records)]
transaction_amounts = np.random.uniform(5, 500, num_records)
review_counts = np.random.poisson(5, num_records)
return_rates = np.random.uniform(0, 0.2, num_records)
price_fluctuations = np.random.uniform(-0.05, 0.05, num_records)
labels = np.random.choice([0, 1], size=num_records, p=[0.8, 0.2])  # 80% 正常，20% 可疑

# 創建 DataFrame
df = pd.DataFrame({
    "商家 ID": merchant_ids,
    "商品 ID": product_ids,
    "交易金額": transaction_amounts,
    "評論數量": review_counts,
    "退貨率": return_rates,
    "價格波動": price_fluctuations,
    "風險狀態": labels
})

# 提高可疑商家的異常特徵
df.loc[df["風險狀態"] == 1, "退貨率"] = np.random.uniform(0.3, 0.6, df["風險狀態"].sum())
df.loc[df["風險狀態"] == 1, "評論數量"] = np.random.randint(50, 300, df["風險狀態"].sum())

# 替換數值標籤為文字標籤
df["風險狀態"] = df["風險狀態"].map({0: "正常", 1: "可疑"})

# 定義可疑原因
def get_risk_reason(row):
    reasons = []
    if row["風險狀態"] == "可疑":
        if row["退貨率"] > 0.3:
            reasons.append("高退貨率 (>30%)")
        if row["評論數量"] > 100:
            reasons.append("過多評論數 (>100)")
        if abs(row["價格波動"]) > 0.03:
            reasons.append("價格波動過大 (>±3%)")
    return "，".join(reasons) if reasons else "無"

df["可疑原因"] = df.apply(get_risk_reason, axis=1)

# 顯示數據樣本（展開可疑原因）
st.subheader("📋 數據樣本")
st.dataframe(df.head(50), use_container_width=True)

# 顯示總筆數
st.write(f"📊 數據總量: {df.shape[0]} 筆")

# 繪製退貨率分佈
st.subheader("📈 退貨率分佈圖")
fig, ax = plt.subplots()
sns.histplot(df["退貨率"], bins=30, kde=True, ax=ax)
st.pyplot(fig)

# 商家風險狀態比例圖
st.subheader("📌 商家風險狀態比例")
fig, ax = plt.subplots()
df["風險狀態"].value_counts().plot.pie(autopct="%1.1f%%", labels=["正常商家", "可疑商家"], ax=ax)
st.pyplot(fig)

# 允許使用者查詢特定商家
st.subheader("🔍 查詢商家資料")
merchant_query = st.text_input("輸入商家 ID（例如：merchant_10）", "")
if merchant_query:
    result = df[df["商家 ID"] == merchant_query]
    if not result.empty:
        st.dataframe(result)  # 使用 dataframe 顯示完整內容
    else:
        st.write("❌ 找不到該商家")
