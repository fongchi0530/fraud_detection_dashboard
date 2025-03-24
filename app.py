import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# 設定 Matplotlib 中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Windows 用 Microsoft YaHei
plt.rcParams['axes.unicode_minus'] = False  # 避免負號顯示錯誤

# 設定標題
st.title("📊 商家風險數據分析儀表板")

# 生成模擬數據
np.random.seed(42)
num_records = 1000
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

# 增強異常商家的異常特徵
df.loc[df["風險狀態"] == 1, "退貨率"] = np.random.uniform(0.3, 0.6, df["風險狀態"].sum())
df.loc[df["風險狀態"] == 1, "評論數量"] = np.random.randint(50, 300, df["風險狀態"].sum())

# 數據特徵構造
df["銷售波動性"] = df["交易金額"].rolling(10).std().fillna(0) / df["交易金額"].rolling(10).mean().fillna(1)
df["評論變化率"] = df["評論數量"].pct_change().fillna(0)
df["退貨率異常"] = (df["退貨率"] > 0.25).astype(int)
df["價格波動幅度"] = abs(df["價格波動"]) > 0.03

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

# 顯示數據樣本（可編輯）
st.subheader("📋 數據樣本")
st.data_editor(df.head(50), use_container_width=True, hide_index=True)

# 顯示總筆數
st.markdown(f"📊 **數據總量**: {df.shape[0]} 筆")

# 📈 退貨率分佈圖
st.subheader("📈 退貨率分佈")
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["退貨率"], bins=30, kde=True, ax=ax, color="skyblue")
ax.set_title("退貨率分佈", fontsize=14)
st.pyplot(fig)

# 📊 風險狀態比例圖（修正 Pie Chart 中文錯誤）
st.subheader("📌 商家風險狀態比例")
fig, ax = plt.subplots(figsize=(6, 6))
colors = ["#1f77b4", "#ff7f0e"]
df["風險狀態"].value_counts().plot.pie(
    autopct="%1.1f%%",
    labels=["正常", "可疑"],
    colors=colors,
    startangle=140,
    wedgeprops={'linewidth': 2, 'edgecolor': 'white'},
    ax=ax
)
ax.set_ylabel("")  # 移除 y 標籤
ax.set_title("商家風險狀態比例", fontsize=14)
st.pyplot(fig)

# 🔍 查詢商家資料
st.subheader("🔍 查詢商家資料")
merchant_query = st.text_input("輸入商家 ID（例如：merchant_10）", placeholder="請輸入完整商家 ID")
if merchant_query:
    result = df[df["商家 ID"] == merchant_query]
    if not result.empty:
        st.dataframe(result, use_container_width=True)
    else:
        st.error("❌ 找不到該商家，請確認 ID 是否正確")

# 顯示數據特徵的統計信息，並轉換統計列名稱為中文
st.subheader("📊 數據特徵統計")

# 只選擇數字類型的列進行描述性統計
numeric_columns = df.select_dtypes(include=[np.number]).columns
df_description = df[numeric_columns].describe()

# 顯示 df.describe() 的結構以便檢查
st.write("❓ df.describe() 結構:", df_description)

# 顯示列數與名稱，並進行錯誤處理
num_columns = df_description.shape[1]
st.write(f"統計數據的列數：{num_columns}")
if num_columns == 8:  # 確保列數符合預期
    df_description.columns = [
        "數據筆數", "平均值", "標準差", "最小值", "25百分位", "50百分位", "75百分位", "最大值"
    ]
else:
    st.error(f"❌ 描述性統計列數（{num_columns}）與期望列數不符，請檢查數據結構。")

# 顯示更新後的描述性統計表格
st.dataframe(df_description)
