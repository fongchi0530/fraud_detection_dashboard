
import streamlit as st
import joblib
import numpy as np
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pytz
import os

st.set_page_config(page_title="詐騙風險評估系統", layout="centered")

# ===== 模型區塊 =====
try:
    cwd = os.getcwd()
    st.info(f"📁 目前工作目錄：{cwd}")

    files = os.listdir(cwd)
    st.info(f"📄 目前資料夾內檔案：{files}")

    model_path = os.path.join(cwd, "slim_fraud_model.pkl")
    model = joblib.load(model_path)
    st.success("✅ 模型已載入成功")

except Exception as e:
    st.error(f"❌ 找不到模型 slim_fraud_model.pkl，請確認模型檔案與本程式放在同一資料夾。\n錯誤細節：{str(e)}")
    st.stop()

# ===== Google Sheet 紀錄函式 =====
def save_chat_to_google_sheet(user_name, user_msg, bot_msg):
    try:
        creds_dict = json.loads(st.secrets["gcp_service_account"])
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("小詐詐聊天紀錄").sheet1

        taipei_tz = pytz.timezone("Asia/Taipei")
        timestamp = datetime.now(taipei_tz).strftime("%Y-%m-%d %H:%M:%S")
        row_data = [timestamp, user_name, user_msg, bot_msg]
        sheet.append_row(row_data)
        return True
    except Exception as e:
        st.error(f"⚠️ 儲存對話錯誤：{str(e)}")
        return False

# 側邊欄頁面切換
page = st.sidebar.selectbox("選擇頁面", ["詐騙風險問卷", "GPT 詐騙諮詢助理"])

# ===== 問卷預測頁面 =====
if page == "詐騙風險問卷":
    st.title("🔍 詐騙風險預測（簡化特徵）")
    st.markdown("請輸入以下欄位進行預測：")

    features = {
        "V14": st.number_input("V14", value=0.0),
        "V17": st.number_input("V17", value=0.0),
        "V12": st.number_input("V12", value=0.0),
        "V10": st.number_input("V10", value=0.0),
        "V25": st.number_input("V25", value=0.0),
        "V27": st.number_input("V27", value=0.0),
        "V2": st.number_input("V2", value=0.0),
        "V22": st.number_input("V22", value=0.0),
    }

    if st.button("進行預測"):
        input_data = np.array(list(features.values())).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"⚠️ 預測結果：詐騙風險高（信心值：{prob:.2%}）")
        else:
            st.success(f"✅ 預測結果：詐騙風險低（信心值：{prob:.2%}）")

# ===== GPT 聊天機器人頁面 =====
else:
    st.title("🤖 小詐詐：防詐騙聊天助理")

    user_name = st.text_input("請輸入您的暱稱", max_chars=20)

    if "chat_openrouter" not in st.session_state:
        st.session_state.chat_openrouter = []

    for msg in st.session_state.chat_openrouter:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("請輸入您的問題...")
    if user_input and user_name:
        st.session_state.chat_openrouter.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            headers = {
                "Authorization": f"Bearer {st.secrets['openrouter_api_key']}",
                "HTTP-Referer": "https://your-app-name.streamlit.app",
                "X-Title": "GPT 防詐諮詢助理"
            }

            body = {
                "model": "gryphe/mythomax-l2-13b",
                "messages": st.session_state.chat_openrouter,
                "temperature": 0.7,
            }

            import requests
            res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
            reply = res.json()["choices"][0]["message"]["content"]

        except Exception as e:
            reply = f"⚠️ 錯誤：{str(e)}"

        st.session_state.chat_openrouter.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

        save_chat_to_google_sheet(user_name, user_input, reply)
    elif user_input and not user_name:
        st.warning("請先輸入暱稱！")
