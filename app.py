import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import json
import requests
import gspread
import pytz
import os
import gdown
TW = pytz.timezone("Asia/Taipei")

def now_tw():
    """取得台灣時區的現在時間（tz-aware）"""
    return datetime.now(TW)

def to_tw(dt):
    """把任何 datetime 轉成台灣時區：
       - tz-aware：直接轉台灣
       - naive：視為 UTC，再轉台灣
    """
    if getattr(dt, "tzinfo", None) is None:
        dt = pytz.utc.localize(dt)
    return dt.astimezone(TW)

def to_tw_str(dt, fmt="%Y-%m-%d %H:%M:%S"):
    """轉為台灣時區後再格式化成字串"""
    return to_tw(dt).strftime(fmt)
from oauth2client.service_account import ServiceAccountCredentials

def ensure_creditcard_csv_local(local_path="creditcard.csv"):
    """
    確保本機有 creditcard.csv；若沒有，就從 Google Drive 下載。
    """
    if os.path.exists(local_path):
        return local_path  # 已存在就直接用

    # 從 secrets 取 Drive 檔案 ID
    file_id = None
    try:
        # 先試環境變數／secrets（Streamlit Cloud）
        file_id = st.secrets.get("GDRIVE_CREDIT_FILE_ID", "").strip()
    except Exception:
        pass

    if not file_id:
        # 退而求其次：你也可以直接把 ID 寫死，但不建議
        raise RuntimeError("找不到 GDRIVE_CREDIT_FILE_ID，請到 Secrets 設定檔案 ID")

    # gdown 下載（可處理 Google Drive 大檔、病毒掃描頁面）
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    # 顯示進度（可選）
    with st.spinner("正在從 Google Drive 下載 creditcard.csv..."):
        gdown.download(url, local_path, quiet=False)

    if not os.path.exists(local_path):
        raise RuntimeError("下載失敗：沒有找到本機的 creditcard.csv")
    return local_path

def prepare_features(input_dict):
    df_tmp = pd.DataFrame([input_dict])
    df_tmp['Amount_log'] = np.log1p(df_tmp['Amount'])
    v_cols = [c for c in df_tmp.columns if c.startswith('V')]
    if v_cols:
        df_tmp['V_mean'] = df_tmp[v_cols].mean(axis=1)
        df_tmp['V_std']  = df_tmp[v_cols].std(axis=1)
        df_tmp['V_max']  = df_tmp[v_cols].max(axis=1)
        df_tmp['V_min']  = df_tmp[v_cols].min(axis=1)
    return df_tmp

def predict_fraud(model, scaler, features, input_data):
    X = prepare_features(input_data)
    X = X[features]                # 對齊訓練時的特徵欄位
    Xs = scaler.transform(X)       # 縮放
    pred = model.predict(Xs)[0]
    prob = model.predict_proba(Xs)[0]
    return pred, prob

st.set_page_config(
    page_title="信用卡交易監測系統",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    .stMetric { 
        background-color: #1e3a5f; 
        padding: 15px; 
        border-radius: 5px; 
        color: white;
    }
    .stMetric > div { color: white !important; }
    .stMetric label { color: #b8c5d6 !important; }
    .stMetric [data-testid="stMetricValue"] { color: white !important; }
    .stMetric [data-testid="stMetricDelta"] { color: #4ade80 !important; }
    .success-box { background-color: #d4edda; padding: 15px; border-radius: 5px; border-left: 5px solid #28a745; }
    .danger-box { background-color: #f8d7da; padding: 15px; border-radius: 5px; border-left: 5px solid #dc3545; }
    .warning-box { background-color: #fff3cd; padding: 15px; border-radius: 5px; border-left: 5px solid #ffc107; }
    div[data-testid="stMetricValue"] { font-size: 2rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        model = joblib.load('fraud_model.pkl')
        scaler = joblib.load('fraud_scaler.pkl')
        features = joblib.load('fraud_features.pkl')
        metrics = joblib.load('model_metrics.pkl')
        return model, scaler, features, metrics
    except:
        return None, None, None, None

@st.cache_data(ttl=24*3600, show_spinner=False)  # 下載/讀檔結果快取 24 小時
def load_data():
    try:
        local_csv = ensure_creditcard_csv_local("creditcard.csv")
        # 直接讀 creditcard.csv
        df = pd.read_csv(local_csv)

        # 你原本的清理邏輯
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)

        # 若欄位名是典型 Kaggle 版（Time, V1~V28, Amount, Class），可在這裡做保險檢查
        needed = {'Amount', 'Class'}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"缺少必要欄位：{missing}")

        return df
    except Exception as e:
        st.error(f"讀取 creditcard.csv 失敗：{e}")
        return None
    
model, scaler, features, metrics = load_models()
df = load_data()

if model is None or df is None:
    st.error("系統初始化失敗：請確認模型檔案和資料集存在")
    st.stop()
with st.sidebar:
    st.title(" 信用卡交易監測系統")
    st.title("👤 使用者資訊")
    user_name = st.text_input("請輸入你的暱稱（可留空）", placeholder="例如：小美")
    
    if not user_name:
        user_name = "匿名使用者"
    
    st.divider()
menu = st.sidebar.selectbox(
    "功能選單",
    ["監控總覽","資料分析","交易檢測","小詐詐聊天","情境腳本"]

)

if menu == "監控總覽":
    st.header("監控總覽")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(df)
    fraud = (df['Class'] == 1).sum()
    normal = (df['Class'] == 0).sum()
    rate = (fraud / total * 100)
    
    with col1:
        st.metric("總交易數", f"{total:,}")
    with col2:
        st.metric("正常交易", f"{normal:,}")
    with col3:
        st.metric("可疑交易", f"{fraud:,}")
    with col4:
        st.metric("風險比例", f"{rate:.3f}%")
    
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("模型效能")
        if metrics:
            perf_df = pd.DataFrame({
                '指標': ['準確率', '精確率', '召回率', 'AUC-ROC'],
                '數值': [
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['auc_roc']:.4f}"
                ]
            })
            st.table(perf_df)
            
            st.subheader("混淆矩陣")
            cm = metrics['confusion_matrix']
            cm_df = pd.DataFrame(
                cm,
                index=['實際:正常', '實際:可疑'],
                columns=['預測:正常', '預測:可疑']
            )
            st.dataframe(cm_df)
    
    with col2:
        st.subheader("模型資訊")
        if hasattr(model, '__class__'):
            model_info = pd.DataFrame({
                '項目': ['模型類型', '特徵數量', '訓練樣本', '測試準確率'],
                '數值': [
                    model.__class__.__name__,
                    f"{len(features)} 個",
                    f"{total:,} 筆",
                    f"{metrics['accuracy']*100:.2f}%" if metrics else "N/A"
                ]
            })
            st.table(model_info)
        
        st.subheader("特徵重要性 (Top 5)")
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(5)
            
            for _, row in importance_df.iterrows():
                st.progress(row['importance'], text=f"{row['feature']}: {row['importance']:.3f}")


elif menu == "資料分析":
    st.header("交易資料分析")
    
    analysis_type = st.selectbox(
        "分析類型",
        ["資料概覽", "特徵分佈", "相關性分析"]
    )
    
    if analysis_type == "資料概覽":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("資料集統計")
            st.dataframe(df.describe(), height=400)
        
        with col2:
            st.subheader("資料資訊")
            info_df = pd.DataFrame({
                '項目': ['總筆數', '特徵數', '可疑比例', '金額中位數', '金額平均值'],
                '數值': [
                    f"{len(df):,}",
                    f"{len(df.columns)}",
                    f"{(df['Class']==1).mean()*100:.3f}%",
                    f"${df['Amount'].median():.2f}",
                    f"${df['Amount'].mean():.2f}"
                ]
            })
            st.table(info_df)
    
    elif analysis_type == "特徵分佈":
        feature = st.selectbox("選擇特徵", df.columns.tolist())
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{feature} 分佈")
            st.bar_chart(
                pd.DataFrame({
                    '正常': df[df['Class']==0][feature].value_counts().head(20),
                    '可疑': df[df['Class']==1][feature].value_counts().head(20)
                })
            )
        
        with col2:
            st.subheader("統計摘要")
            stats_df = pd.DataFrame({
                '統計量': ['平均值', '中位數', '標準差', '最小值', '最大值'],
                '正常交易': [
                    df[df['Class']==0][feature].mean(),
                    df[df['Class']==0][feature].median(),
                    df[df['Class']==0][feature].std(),
                    df[df['Class']==0][feature].min(),
                    df[df['Class']==0][feature].max()
                ],
                '可疑交易': [
                    df[df['Class']==1][feature].mean(),
                    df[df['Class']==1][feature].median(),
                    df[df['Class']==1][feature].std(),
                    df[df['Class']==1][feature].min(),
                    df[df['Class']==1][feature].max()
                ]
            })
            st.dataframe(stats_df)
    
    elif analysis_type == "相關性分析":
        st.subheader("特徵相關性矩陣")
        
        n_features = st.slider("顯示前 N 個特徵", 5, 30, 10)
        selected_features = df.columns[:n_features].tolist()
        
        if 'Class' not in selected_features:
            selected_features.append('Class')
        
        corr_matrix = df[selected_features].corr()
        
        st.dataframe(
            corr_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1),
            height=400
        )
        
        st.subheader("與 Class 相關性最高的特徵")
        class_corr = df.corr()['Class'].abs().sort_values(ascending=False)[1:11]
        corr_df = pd.DataFrame({
            '特徵': class_corr.index,
            '相關係數': class_corr.values
        })
        st.table(corr_df)

elif menu == "交易檢測":
    st.header("信用卡交易檢測")
    
    # 使用單欄設計，更簡潔
    st.subheader("快速風險評估")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        amount = st.number_input(
            "交易金額 (USD)",
            min_value=0.01,
            max_value=10000.0,
            value=100.0,
            step=10.0,
            help="輸入本次刷卡金額"
        )
    
    with col2:
        hour = st.selectbox(
            "發生交易時間 (小時)",
            options=list(range(24)),
            index=12,
            help="選擇交易發生的時間"
        )
    
    with col3:
        merchant = st.selectbox(
            "商家類型",
            ["餐飲", "購物", "交通", "娛樂", "線上服務", "其他"]
        )
    
    # 進階選項
    with st.expander("進階設定（選填）"):
        col_a, col_b = st.columns(2)
        
        with col_a:
            consecutive = st.checkbox("連續多筆交易", help="最近30分鐘內有其他交易")
            international = st.checkbox("跨國交易", help="商家位於國外")
            first_time = st.checkbox("首次交易商家", help="從未在此商家消費過")
        
        with col_b:
            unusual_amount = st.checkbox("異常金額", help="金額遠超平常消費")
            weekend = st.checkbox("週末交易", help="交易發生在週末")
            online = st.checkbox("線上交易", help="非實體店面交易")
    
    # 立即執行檢測按鈕
    if st.button("立即檢測", type="primary", use_container_width=True):
        detected_at = now_tw()  
        # 計算風險分數
        risk_score = 0
        risk_factors = []
        
        # 基礎風險評估
        if hour >= 0 and hour < 6:
            risk_score += 25
            risk_factors.append("深夜交易")
        
        if amount > 500:
            risk_score += 15
            if amount > 1000:
                risk_score += 10
                risk_factors.append("高額交易")
        
        if merchant in ["線上服務", "其他"]:
            risk_score += 10
        
        # 進階風險評估
        if consecutive:
            risk_score += 20
            risk_factors.append("短時間多筆")
        if international:
            risk_score += 15
            risk_factors.append("跨國交易")
        if first_time:
            risk_score += 10
            risk_factors.append("新商家")
        if unusual_amount:
            risk_score += 20
            risk_factors.append("金額異常")
        if weekend and hour in [2, 3, 4, 5]:
            risk_score += 10
        if online:
            risk_score += 5
        
        # 根據風險生成模型輸入
        np.random.seed(int(amount * 100 + hour + len(merchant)))
        input_data = {}
        
        for i in range(1, 29):
            if risk_score > 60:
                input_data[f'V{i}'] = np.random.normal(0, 2.0) * (1 + risk_score/100)
            elif risk_score > 30:
                input_data[f'V{i}'] = np.random.normal(0, 1.5) * (1 + risk_score/200)
            else:
                input_data[f'V{i}'] = np.random.normal(0, 1.0) * max(0.5, 1 - risk_score/100)
        
        input_data['Amount'] = amount
        
        # 執行模型預測
        pred, prob = predict_fraud(model, scaler, features, input_data)
        
        # 顯示結果
        st.divider()
        
        # 結果顯示區
        if pred == 1 or risk_score > 60:
            st.error("⚠️ 高風險交易")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("風險等級", "高", delta="需要確認")
            with col2:
                st.metric("風險分數", f"{min(risk_score, 95)}/100")
            with col3:
                st.metric("正常機率", f"{prob[1]*100:.1f}%")
            
            if risk_factors:
                st.warning("**風險因素：**")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            
            st.info("**建議措施：**\n• 立即聯繫銀行確認\n• 檢查是否為本人交易\n• 必要時凍結卡片")
            
        elif risk_score > 30:
            st.warning("⚠️ 中度風險")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("風險等級", "中", delta="需留意")
            with col2:
                st.metric("風險分數", f"{risk_score}/100")
            with col3:
                st.metric("正常機率", f"{prob[1]*100:.1f}%")
            
            if risk_factors:
                st.info("**注意事項：**")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            
            st.info("**建議措施：**\n• 確認交易明細\n• 留意後續交易")
            
        else:
            st.success("✓ 正常交易")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("風險等級", "低", delta="安全")
            with col2:
                st.metric("風險分數", f"{risk_score}/100")
            with col3:
                st.metric("正常機率", f"{prob[0]*100:.1f}%")
            
            st.success("交易模式正常，無需額外處理")
        
        # 交易摘要
        st.divider()
        st.subheader("交易摘要")
        
        summary_df = pd.DataFrame({
            '項目': ['金額', '時間', '商家', '檢測時間', '模型信心度'],
            '內容': [
                f'${amount:.2f} USD',
                f'{hour:02d}:00',
                merchant,
                detected_at.strftime('%Y-%m-%d %H:%M:%S'),
                f'{max(prob)*100:.1f}%'
            ]
        })
        st.table(summary_df)
        
        # 儲存到session state
        if 'detection_history' not in st.session_state:
            st.session_state['detection_history'] = []
        
        st.session_state['detection_history'].append({
            'time': detected_at,
            'amount': amount,
            'risk': '高' if pred == 1 or risk_score > 60 else ('中' if risk_score > 30 else '低'),
            'score': risk_score
        })
    
    # 顯示檢測歷史
    if 'detection_history' in st.session_state and len(st.session_state['detection_history']) > 0:
        st.divider()
        st.subheader("最近檢測記錄")
        
        history = st.session_state['detection_history'][-5:]  # 顯示最近5筆
        history_df = pd.DataFrame([
            {
                '時間': to_tw_str(h['time'], "%Y-%m-%d %H:%M:%S"), 
                '金額': f"${h['amount']:.2f}",
                '風險': h['risk'],
                '分數': h['score']
            }
            for h in reversed(history)
        ])
        st.dataframe(history_df, hide_index=True, use_container_width=True)

elif menu == "小詐詐聊天":
    st.header(" 小詐詐 GPT 聊天助手")

    with st.expander(" 小詐詐能幫你什麼？", expanded=True):
        st.markdown("""
            **小詐詐是你的防詐騙小助手！**

            你可以向小詐詐諮詢：
            - 判斷是否遇到詐騙情況
            - 分析可疑訊息或行為
            - 提供防詐騙建議
            - 解答詐騙相關問題

            **試試詢問：**
            - "有人叫我加LINE後要我匯款"
            - "接到自稱是我兒子的電話說要錢"
            - "如何辨別假購物網站？"
            - "有人要我安裝遠端軟體"
            """)

    if "chat_openrouter" not in st.session_state:
        st.session_state.chat_openrouter = []

    for msg in st.session_state.chat_openrouter:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("請描述你遇到的情況，例如：有人叫我加 LINE 匯款")

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

    if user_input and user_input.strip():
        st.session_state.chat_openrouter.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        messages = [
            {
                "role": "system",
                "content": (
                    "你是『小詐詐🕵️』，一個警覺又溫柔的防詐小幫手。"
                    "你的任務是協助使用者判斷是否遇到詐騙，口吻自然、親切、真誠。"
                    "請勇敢提醒使用者保護自己：不要轉帳、不給個資、不加陌生人 LINE，必要時報警。"
                    "不要對使用者亂取綽號或名字直接回答問題。"
                )
            }
        ] + st.session_state.chat_openrouter

        headers = {
            "Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
            "HTTP-Referer": "https://your-app-name.streamlit.app",
            "Content-Type": "application/json"
        }

        data = {
            "model": "gryphe/mythomax-l2-13b",
            "messages": messages
        }

        try:
            with st.spinner("小詐詐努力判斷中，請稍候...🧠"):
                response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                         headers=headers, json=data)
                res_json = response.json()
                if "choices" in res_json:
                    reply = res_json["choices"][0]["message"]["content"]
                elif "error" in res_json:
                    reply = f"⚠️ API 錯誤：{res_json['error'].get('message', '未知錯誤')}"
                else:
                    reply = "⚠️ 小詐詐無法取得回應，請稍後再試～"
        except Exception as e:
            reply = f"⚠️ 小詐詐出現例外錯誤：{str(e)}"

        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.chat_openrouter.append({"role": "assistant", "content": reply})
        save_chat_to_google_sheet(user_name, user_input, reply)

elif menu == "情境腳本":
    st.header("情境腳本 / 使用者旅程")
    st.caption("以真實場景引導一般使用者辨識風險並學會正確處置。")

    # === 內建 4 個常見情境 ===
    SCENARIOS = {
        "購物平台假客服退款": [
            ("對方要求加 LINE/Telegram 私下聯繫？", [("沒有", 0), ("有，但尚未加", 2), ("已加並私聊", 5)]),
            ("被要求提供個資/卡號/身分證影本？", [("沒有", 0), ("有，提供部分", 3), ("有，完整提供", 6)]),
            ("被要求操作 ATM/網銀以『解除分期/錯誤扣款』？", [("沒有", 0), ("有，但未操作", 6), ("已操作", 10)]),
            ("是否點擊過不明簡訊/連結或提供 OTP？", [("沒有", 0), ("有", 8)]),
            ("對話是否充滿緊迫性字眼（限時、帳號將被停權）？", [("沒有", 0), ("有", 3)]),
        ],
        "商品交易：面交／貨到付款": [
            ("賣家帳號剛建立或評價很少？", [("否", 0), ("是", 3)]),
            ("價格明顯過低或不合理贈品？", [("否", 0), ("是", 4)]),
            ("要求改變交貨地點／臨時換聯絡方式？", [("否", 0), ("是", 4)]),
            ("堅持貨到付款但拒絕開箱＋退換貨？", [("否", 0), ("是", 5)]),
        ],
        "投資群組／保證獲利": [
            ("對方保證高報酬、零風險？", [("否", 0), ("是", 6)]),
            ("要求加密貨幣/USDT/境外平台入金？", [("否", 0), ("是", 8)]),
            ("展示假對帳單/『老師』帶單成績？", [("否", 0), ("是", 5)]),
        ],
        "親友急需匯款（冒充親友/公務單位）": [
            ("使用陌生號碼/網路電話，聲稱緊急狀況？", [("否", 0), ("是", 6)]),
            ("要求立即轉帳且阻止你掛電話求證？", [("否", 0), ("是", 8)]),
            ("要求提供銀行/個資或遠端協助？", [("否", 0), ("是", 6)]),
        ],
    }

    scenario = st.selectbox("選擇情境", list(SCENARIOS.keys()))

    st.subheader("依序回答以下問題（越符合越高風險）")
    answers = []
    total_max = 0
    for idx, (q, choices) in enumerate(SCENARIOS[scenario], start=1):
        labels = [c[0] for c in choices]
        weights = [c[1] for c in choices]
        choice = st.radio(f"{idx}. {q}", labels, index=0, horizontal=True)
        answers.append(weights[labels.index(choice)])
        total_max += max(weights)

    # --- 計分與等級 ---
    score = int(sum(answers))
    ratio = score / max(1, total_max)
    st.divider()
    st.subheader("風險評估結果")
    st.progress(min(ratio, 1.0), text=f"風險分數 {score} / {total_max}")

    if ratio < 0.3:
        st.success("風險等級：低\n\n此情境目前風險不高，保持警覺即可。")
        level = "低"
    elif ratio < 0.6:
        st.warning("風險等級：中\n\n此情境包含部分可疑特徵，建議停下確認、保留證據。")
        level = "中"
    else:
        st.error("風險等級：高\n\n高度疑似詐騙！請中止互動並採取下列措施。")
        level = "高"

    # --- 使用者旅程（教育視覺） ---
    st.subheader("使用者旅程（建議行為）")
    cols = st.columns(5)
    steps = ["接觸訊息", "辨識可疑點", "停止互動", "蒐證與求證", "回報/阻詐"]
    tips  = [
        "保留對話/截圖，不點陌生連結。",
        "檢查是否要求私下聯繫、緊迫性、要個資/OTP。",
        "不要轉帳、不提供個資、不下載遠端軟體。",
        "改用官方客服/親友本機號碼求證。",
        "165 反詐騙、平台檢舉、銀行凍卡/改密碼。",
    ]
    for i, c in enumerate(cols):
        with c:
            c.markdown(f"**{i+1}. {steps[i]}**")
            c.caption(tips[i])

    # --- 具體建議（依情境提供） ---
    st.subheader("建議下一步")
    if scenario == "購物平台假客服退款":
        st.info("- 只用平台內建客服，不私加 LINE\n- 官方不會要你去 ATM/提供 OTP\n- 立即改密碼、開啟簡訊 OTP")
    elif scenario == "商品交易：面交／貨到付款":
        st.info("- 面交務必當面開箱錄影\n- 價格異常請提高警覺\n- 平台內訊息留底，勿跳 App 聯繫")
    elif scenario == "投資群組／保證獲利":
        st.info("- 任何『保證獲利』都是警訊\n- 勿轉 USDT/境外平台\n- 向 165 或金管會檢舉")
    else:
        st.info("- 掛斷改打親友/機關官方電話\n- 不外洩個資和金融資訊\n- 保留證據並通報 165")

st.sidebar.divider()
st.sidebar.caption("信用卡交易監測系統 v1.0")
st.sidebar.caption(f"© 2025 | 最後更新: {datetime.now().strftime('%m-%d')}")