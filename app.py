import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
import pytz
import json
import requests

# ------------------ 模型與標題 ------------------
model = joblib.load('fraud_model.pkl')
st.title("\U0001F4CA 商家風險數據分析儀表板")

# ------------------ 使用者暱稱 ------------------
st.sidebar.title("\U0001F464 使用者資訊")
user_name = st.sidebar.text_input("請輸入你的暱稱（可留空）", placeholder="例如：小美")
if not user_name:
    user_name = "匿名"

# ✅ 測試寫入按鈕
if st.sidebar.button("✍️ 測試寫入一筆記錄"):
    save_chat_to_google_sheet("測試用戶", "這是一條測試訊息", "這是機器人回應")

# ------------------ 模擬數據生成 ------------------
np.random.seed(42)
num_records = 1000
merchant_ids = [f"merchant_{i}" for i in range(1, num_records + 1)]
product_ids = [f"product_{i % 100 + 1}" for i in range(num_records)]
transaction_amounts = np.random.normal(loc=250, scale=80, size=num_records).clip(5, 500)
review_counts = np.random.poisson(lam=15, size=num_records)
return_rates = np.random.beta(a=2, b=10, size=num_records) * 0.4
price_fluctuations = np.random.normal(loc=0.0, scale=0.02, size=num_records).clip(-0.05, 0.05)
labels = np.random.choice([0, 1], size=num_records, p=[0.8, 0.2])

df = pd.DataFrame({
    "商家 ID": merchant_ids,
    "商品 ID": product_ids,
    "交易金額": transaction_amounts,
    "評論數量": review_counts,
    "退貨率": return_rates,
    "價格波動": price_fluctuations,
    "風險狀態": labels
})

# ------------------ 特徵與標記 ------------------
df.loc[df["風險狀態"] == 1, "退貨率"] = np.random.uniform(0.35, 0.6, df["風險狀態"].sum())
df.loc[df["風險狀態"] == 1, "評論數量"] = np.random.randint(100, 300, df["風險狀態"].sum())

df["銷售波動性"] = df["交易金額"].rolling(10).std().fillna(0) / df["交易金額"].rolling(10).mean().fillna(1)
df["評論變化率"] = df["評論數量"].pct_change().fillna(0)
df["退貨率異常"] = (df["退貨率"] > 0.25).astype(int)
df["價格波動幅度"] = abs(df["價格波動"]) > 0.03

df["風險狀態"] = df["風險狀態"].map({0: "正常", 1: "可疑"})

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

# ------------------ 視覺化 ------------------
st.subheader("\U0001F4CB 數據樣本")
st.data_editor(df.head(50), use_container_width=True, hide_index=True)
st.markdown(f"\U0001F4CA **數據總量**: {df.shape[0]} 筆")

st.subheader("\U0001F4C8 退貨率分佈")
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df["退貨率"], bins=30, kde=True, ax=ax, color="skyblue")
ax.set_title("退貨率分佈", fontsize=14)
st.pyplot(fig)

st.subheader("\U0001F4CC 商家風險狀態比例")
fig, ax = plt.subplots(figsize=(6, 6))
colors = ["#28a745", "#dc3545"]
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

st.subheader("\U0001F50D 查詢商家資料")
merchant_query = st.text_input("輸入商家 ID（例如：merchant_10）", placeholder="請輸入完整商家 ID")
if merchant_query:
    result = df[df["商家 ID"] == merchant_query]
    if not result.empty:
        st.dataframe(result, use_container_width=True)
    else:
        st.error("❌ 找不到該商家，請確認 ID 是否正確")

st.subheader("\U0001F4CA 數據特徵統計")
numeric_columns = df.select_dtypes(include=[np.number]).columns
df_description = df[numeric_columns].describe()
st.dataframe(df_description)

# ------------------ 詐騙風險檢測表單 ------------------
# ... (以下略，保留原邏輯)

# ------------------ 函式：寫入 Google Sheet ------------------
def save_chat_to_google_sheet(user_name, user_msg, bot_msg):
    try:
        st.toast("\U0001F4BE 進入儲存函式！")
        st.write(f"🪪 使用者名稱：{user_name or '匿名'}")
        st.write("🛠️ 嘗試寫入 Google Sheet...")

        creds_dict = json.loads(st.secrets["gcp_service_account"])
        st.write("✅ 成功讀取 Google API 金鑰")

        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        st.write("✅ 成功授權 Google Sheets API")

        sheet = client.open("小詐詐聊天紀錄").sheet1
        st.write("✅ 試算表成功打開！")

        taipei_tz = pytz.timezone("Asia/Taipei")
        timestamp = datetime.now(taipei_tz).strftime("%Y-%m-%d %H:%M:%S")
        row_data = [timestamp, user_name, user_msg, bot_msg]
        st.write(f"📤 嘗試寫入數據：{row_data}")
        sheet.append_row(row_data)
        st.write("✅ 成功寫入試算表！")

    except gspread.exceptions.APIError as e:
        st.error(f"⚠️ Google Sheets API 錯誤：{str(e)}")
    except Exception as e:
        st.error(f"⚠️ 其他錯誤：{str(e)}")