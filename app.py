import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 載入訓練好的模型
model = joblib.load('fraud_model.pkl')

# 設定標題
st.title("📊 商家風險數據分析儀表板")

# 生成模擬數據（這裡調整成更合理的範圍）
np.random.seed(42)
num_records = 1000
merchant_ids = [f"merchant_{i}" for i in range(1, num_records + 1)]
product_ids = [f"product_{i % 100 + 1}" for i in range(num_records)]
transaction_amounts = np.random.normal(loc=250, scale=80, size=num_records).clip(5, 500)  # 平均 250，標準差 80
review_counts = np.random.poisson(lam=15, size=num_records)  # 平均評論數變多
return_rates = np.random.beta(a=2, b=10, size=num_records) * 0.4  # 偏低退貨率，但上限較真實
price_fluctuations = np.random.normal(loc=0.0, scale=0.02, size=num_records).clip(-0.05, 0.05)
labels = np.random.choice([0, 1], size=num_records, p=[0.8, 0.2])

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

# 增強可疑商家異常特徵
df.loc[df["風險狀態"] == 1, "退貨率"] = np.random.uniform(0.35, 0.6, df["風險狀態"].sum())
df.loc[df["風險狀態"] == 1, "評論數量"] = np.random.randint(100, 300, df["風險狀態"].sum())

# 特徵工程
df["銷售波動性"] = df["交易金額"].rolling(10).std().fillna(0) / df["交易金額"].rolling(10).mean().fillna(1)
df["評論變化率"] = df["評論數量"].pct_change().fillna(0)
df["退貨率異常"] = (df["退貨率"] > 0.25).astype(int)
df["價格波動幅度"] = abs(df["價格波動"]) > 0.03

# 將風險狀態轉為文字
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

# 📋 數據樣本
st.subheader("📋 數據樣本")
st.data_editor(df.head(50), use_container_width=True, hide_index=True)

# 📊 數據筆數
st.markdown(f"📊 **數據總量**: {df.shape[0]} 筆")

# 📈 退貨率分佈圖
st.subheader("📈 退貨率分佈")
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["退貨率"], bins=30, kde=True, ax=ax, color="skyblue")
ax.set_title("退貨率分佈", fontsize=14)
st.pyplot(fig)

# 📊 風險狀態比例
st.subheader("📌 商家風險狀態比例")
fig, ax = plt.subplots(figsize=(6, 6))
colors = ["#28a745", "#dc3545"]  # 綠＝正常，紅＝可疑
df["風險狀態"].value_counts().plot.pie(
    autopct="%1.1f%%",
    labels=["正常", "可疑"],
    colors=colors,
    startangle=140,
    wedgeprops={'linewidth': 2, 'edgecolor': 'white'},
    ax=ax
)
ax.set_ylabel("")
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

# 📊 數據統計資訊
st.subheader("📊 數據特徵統計")
numeric_columns = df.select_dtypes(include=[np.number]).columns
df_description = df[numeric_columns].describe()
st.dataframe(df_description)

# 🔮 詐騙風險檢測表單
st.subheader("🔍 詐騙風險檢測表單")

with st.form("fraud_form"):
    transaction_amount = st.number_input("💵 交易金額", min_value=0.0, max_value=500.0, step=1.0)
    review_count = st.number_input("📝 評論數量", min_value=0, max_value=1000, step=1)
    return_rate = st.slider("📦 退貨率", min_value=0.0, max_value=0.6, step=0.01)
    price_fluctuation = st.slider("💹 價格波動（正負%)", min_value=-0.05, max_value=0.05, step=0.01)

    submit = st.form_submit_button("✨ 預測是否為詐騙商家")

if submit:
    expected_columns = ['交易金額', '評論數量', '退貨率', '價格波動',
                        '銷售波動性', '評論變化率', '退貨率異常', '價格波動幅度']

    input_data = pd.DataFrame({
        '交易金額': [transaction_amount],
        '評論數量': [review_count],
        '退貨率': [return_rate],
        '價格波動': [price_fluctuation],
        '銷售波動性': [np.random.uniform(0.1, 0.4)],  # 更合理的範圍
        '評論變化率': [np.random.uniform(-0.1, 0.3)],
        '退貨率異常': [int(return_rate > 0.25)],
        '價格波動幅度': [abs(price_fluctuation) > 0.03]
    })

    input_data = input_data[expected_columns]

    prediction = model.predict(input_data)[0]
    risk_score = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ 這可能是可疑商家！風險分數：{risk_score:.2f}")
    else:
        st.success(f"✅ 看起來是正常商家，風險分數：{risk_score:.2f}")
