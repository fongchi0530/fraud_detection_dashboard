import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import requests
import pytz
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import streamlit.components.v1 as components

# 頁面配置設定 - 讓頁面更寬敞
st.set_page_config(
    page_title="商家風險數據分析儀表板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS 樣式
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 載入訓練好的模型
@st.cache_resource
def load_model():
    return joblib.load('fraud_model.pkl')

model = load_model()

# 生成模擬數據（只有在需要時才生成）
@st.cache_data
def generate_data():
    np.random.seed(42)
    num_records = 1000
    merchant_ids = [f"merchant_{i}" for i in range(1, num_records + 1)]
    product_ids = [f"product_{i % 100 + 1}" for i in range(num_records)]
    transaction_amounts = np.random.normal(loc=250, scale=80, size=num_records).clip(5, 500)
    review_counts = np.random.poisson(lam=15, size=num_records)
    return_rates = np.random.beta(a=2, b=10, size=num_records) * 0.4
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
    
    return df

df = generate_data()

# ---------- 頁面佈局 ----------
st.title("📊 商家風險數據分析儀表板")

# 側邊欄設定
with st.sidebar:
    st.title("👤 使用者資訊")
    user_name = st.text_input("請輸入你的暱稱（可留空）", placeholder="例如：小美")
    
    if not user_name:
        user_name = "匿名使用者"
    
    st.divider()
    st.subheader("💡 儀表板說明")
    st.info("""
    此儀表板提供商家風險分析工具，協助您：
    - 查看商家數據統計
    - 分析詐騙風險趨勢
    - 檢測可疑商家
    - 透過小詐詐聊天助手取得協助
    """)

# 主要內容分頁
tabs = st.tabs(["📈 數據總覽", "🔍 風險檢測", "🤖 小詐詐聊天"])

# ---------- 數據總覽頁面 ----------
with tabs[0]:
    # 使用列佈局
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📋 數據樣本")
        st.data_editor(
            df.head(50), 
            use_container_width=True, 
            hide_index=True,
            height=300
        )
        
        # 查詢商家資料
        st.subheader("🔍 查詢商家資料")
        merchant_query = st.text_input("輸入商家 ID（例如：merchant_10）", 
                                      placeholder="請輸入完整商家 ID",
                                      key="merchant_search")
        if merchant_query:
            result = df[df["商家 ID"] == merchant_query]
            if not result.empty:
                st.dataframe(result, use_container_width=True)
                
                # 顯示風險分析卡片
                risk_status = result["風險狀態"].values[0]
                risk_reason = result["可疑原因"].values[0]
                
                if risk_status == "可疑":
                    st.error(f"⚠️ 風險狀態: {risk_status}\n\n可疑原因: {risk_reason}")
                else:
                    st.success(f"✅ 風險狀態: {risk_status}")
            else:
                st.error("❌ 找不到該商家，請確認 ID 是否正確")
        
        # 數據統計資訊
        with st.expander("📊 數據特徵統計"):
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df_description = df[numeric_columns].describe().round(2)
            st.dataframe(df_description, use_container_width=True)
    
    with col2:
        # 使用 Chart.js 製作退貨率分佈直方圖
        st.subheader("📈 退貨率分佈")
        
        # 計算直方圖數據
        hist_data, bin_edges = np.histogram(df["退貨率"], bins=30, range=(0, 0.6))
        bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
        
        # 生成 Chart.js 的 HTML
        hist_chart_html = f"""
        <html>
        <head>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.7.1/chart.min.js"></script>
            <style>
                .chart-container {{
                    height: 300px;
                    width: 100%;
                    margin: 0 auto;
                }}
            </style>
        </head>
        <body>
            <div class="chart-container">
                <canvas id="histChart"></canvas>
            </div>
            <script>
                var ctx = document.getElementById('histChart').getContext('2d');
                var chart = new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: {[round(x*100, 1) for x in bin_centers]},
                        datasets: [{{
                            label: '頻率',
                            data: {list(hist_data)},
                            backgroundColor: 'rgba(54, 162, 235, 0.5)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            title: {{
                                display: true,
                                text: '退貨率分佈 (%)',
                                font: {{
                                    size: 16
                                }}
                            }},
                            legend: {{
                                display: false
                            }}
                        }},
                        scales: {{
                            x: {{
                                title: {{
                                    display: true,
                                    text: '退貨率 (%)'
                                }}
                            }},
                            y: {{
                                beginAtZero: true,
                                title: {{
                                    display: true,
                                    text: '商家數量'
                                }}
                            }}
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
        """
        
        # 顯示 Chart.js 圖表
        components.html(hist_chart_html, height=350)
        
        # 風險狀態比例圓餅圖 (使用 Chart.js)
        st.subheader("📌 商家風險狀態比例")
        
        # 計算風險狀態比例
        risk_counts = df["風險狀態"].value_counts()
        normal_count = risk_counts.get("正常", 0)
        suspicious_count = risk_counts.get("可疑", 0)
        
        # 生成 Chart.js 的圓餅圖 HTML
        pie_chart_html = f"""
        <html>
        <head>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.7.1/chart.min.js"></script>
            <style>
                .chart-container {{
                    height: 300px;
                    width: 100%;
                    margin: 0 auto;
                }}
            </style>
        </head>
        <body>
            <div class="chart-container">
                <canvas id="pieChart"></canvas>
            </div>
            <script>
                var ctx = document.getElementById('pieChart').getContext('2d');
                var chart = new Chart(ctx, {{
                    type: 'pie',
                    data: {{
                        labels: ['正常', '可疑'],
                        datasets: [{{
                            data: [{normal_count}, {suspicious_count}],
                            backgroundColor: [
                                'rgba(40, 167, 69, 0.8)',
                                'rgba(220, 53, 69, 0.8)'
                            ],
                            borderColor: [
                                'rgba(40, 167, 69, 1)',
                                'rgba(220, 53, 69, 1)'
                            ],
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {{
                            title: {{
                                display: true,
                                text: '商家風險狀態比例',
                                font: {{
                                    size: 16
                                }}
                            }},
                            legend: {{
                                position: 'bottom'
                            }},
                            tooltip: {{
                                callbacks: {{
                                    label: function(context) {{
                                        var label = context.label || '';
                                        var value = context.raw;
                                        var total = context.dataset.data.reduce((a, b) => a + b, 0);
                                        var percentage = Math.round((value / total) * 100);
                                        return label + ': ' + value + ' (' + percentage + '%)';
                                    }}
                                }}
                            }}
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
        """
        
        # 顯示 Chart.js 圓餅圖
        components.html(pie_chart_html, height=350)

# ---------- 風險檢測頁面 ----------
with tabs[1]:
    st.subheader("🔮 詐騙風險檢測")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 表單容器
        with st.form("fraud_form"):
            st.write("輸入商家數據進行風險評估:")
            transaction_amount = st.number_input("💵 交易金額", min_value=0.0, max_value=50000.0, value=250.0, step=10.0)
            review_count = st.number_input("📝 評論數量", min_value=0, max_value=10000, value=15, step=5)
            return_rate = st.slider("📦 退貨率", min_value=0.0, max_value=0.6, value=0.1, step=0.01)
            price_fluctuation = st.slider("💹 價格波動（正負%)", min_value=-0.05, max_value=0.05, value=0.01, step=0.01)
            
            submit = st.form_submit_button("✨ 預測風險")
    
    with col2:
        if submit:
            # 創建特徵數據
            expected_columns = ['交易金額', '評論數量', '退貨率', '價格波動',
                              '銷售波動性', '評論變化率', '退貨率異常', '價格波動幅度']
            
            input_data = pd.DataFrame({
                '交易金額': [transaction_amount],
                '評論數量': [review_count],
                '退貨率': [return_rate],
                '價格波動': [price_fluctuation],
                '銷售波動性': [np.random.uniform(0.1, 0.4)],
                '評論變化率': [np.random.uniform(-0.1, 0.3)],
                '退貨率異常': [int(return_rate > 0.25)],
                '價格波動幅度': [abs(price_fluctuation) > 0.03]
            })
            
            input_data = input_data[expected_columns]
            
            # 預測結果
            prediction = model.predict(input_data)[0]
            risk_score = model.predict_proba(input_data)[0][1]
            
            # 計算各個因素的貢獻度（簡化版）
            factor_weights = {
                "退貨率": 0.4 if return_rate > 0.25 else 0.0,
                "評論數量": 0.3 if review_count > 100 else 0.0,
                "價格波動": 0.3 if abs(price_fluctuation) > 0.03 else 0.0
            }
            
            # 結果呈現
            if prediction == 1:
                st.error(f"⚠️ 警告：這可能是可疑商家！")
                st.metric("風險分數", f"{risk_score:.2f}", delta=f"+{risk_score:.2f}", delta_color="inverse")
            else:
                st.success(f"✅ 分析結果：看起來是正常商家")
                st.metric("風險分數", f"{risk_score:.2f}", delta=f"{risk_score:.2f}", delta_color="inverse")
            
            # 生成風險因素圓餅圖 (使用 Chart.js)
            factors = list(factor_weights.keys())
            weights = list(factor_weights.values())
            
            risk_factors_html = f"""
            <html>
            <head>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.7.1/chart.min.js"></script>
                <style>
                    .chart-container {{
                        height: 250px;
                        width: 100%;
                        margin: 0 auto;
                    }}
                </style>
            </head>
            <body>
                <div class="chart-container">
                    <canvas id="factorChart"></canvas>
                </div>
                <script>
                    var ctx = document.getElementById('factorChart').getContext('2d');
                    var chart = new Chart(ctx, {{
                        type: 'radar',
                        data: {{
                            labels: {factors},
                            datasets: [{{
                                label: '風險因素權重',
                                data: {weights},
                                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                                borderColor: 'rgba(255, 99, 132, 1)',
                                borderWidth: 2,
                                pointBackgroundColor: 'rgba(255, 99, 132, 1)'
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                title: {{
                                    display: true,
                                    text: '風險因素分析',
                                    font: {{
                                        size: 16
                                    }}
                                }}
                            }},
                            scales: {{
                                r: {{
                                    beginAtZero: true,
                                    max: 0.5,
                                    ticks: {{
                                        stepSize: 0.1
                                    }}
                                }}
                            }}
                        }}
                    }});
                </script>
            </body>
            </html>
            """
            
            # 顯示 Chart.js 風險因素圖
            st.subheader("📊 風險因素分析")
            components.html(risk_factors_html, height=280)

# ---------- 聊天助手頁面 ----------
with tabs[2]:
    st.subheader("🤖 小詐詐 GPT 聊天助手")
    
    # 初始化聊天歷史紀錄
    if "chat_openrouter" not in st.session_state:
        st.session_state.chat_openrouter = []
    
    # 輔助性說明
    with st.expander("💡 小詐詐能幫你什麼？"):
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

    # 顯示歷史對話訊息
    for msg in st.session_state.chat_openrouter:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # 使用者輸入
    user_input = st.chat_input("請描述你遇到的情況，例如：有人叫我加 LINE 匯款")

    # 函式：寫入 Google Sheet
    def save_chat_to_google_sheet(user_name, user_msg, bot_msg):
        try:
            st.toast("\U0001F4BE 進入儲存函式！")
            st.write(f"🪪 使用者名稱：{user_name or '匿名'}")
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
        # 儲存使用者訊息
        st.session_state.chat_openrouter.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # 系統指令（角色設定）+ 對話歷史
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
            "HTTP-Referer": "https://chihlee-frauddetectiondashboard.streamlit.app",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gryphe/mythomax-l2-13b",
            "messages": messages
        }
        
        try:
            with st.spinner("小詐詐努力判斷中，請稍候...🧠"):
                response = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                                       headers=headers, json=data, timeout=30)
                res_json = response.json()
                
                if "choices" in res_json:
                    reply = res_json["choices"][0]["message"]["content"]
                elif "error" in res_json:
                    reply = f"⚠️ API 錯誤：{res_json['error'].get('message', '未知錯誤')}"
                else:
                    reply = "⚠️ 小詐詐無法取得回應，請稍後再試～"
        
        except Exception as e:
            reply = f"⚠️ 小詐詐出現例外錯誤：{str(e)}"
        
        # 顯示回覆
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.chat_openrouter.append({"role": "assistant", "content": reply})
        
        # 儲存對話記錄
        save_chat_to_google_sheet(user_name, user_input, reply)

# 頁面頁腳
st.divider()
st.caption("© 2025 商家風險數據分析儀表板 - 版權所有")