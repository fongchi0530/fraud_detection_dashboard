import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib

# 讀取數據（假設來自 fraud_detection_dashboard 的數據）
def load_data():
    np.random.seed(42)
    num_records = 1000
    df = pd.DataFrame({
        "商家 ID": [f"merchant_{i}" for i in range(1, num_records + 1)],
        "交易金額": np.random.uniform(5, 500, num_records),
        "評論數量": np.random.poisson(5, num_records),
        "退貨率": np.random.uniform(0, 0.6, num_records),
        "價格波動": np.random.uniform(-0.05, 0.05, num_records),
        "風險狀態": np.random.choice([0, 1], size=num_records, p=[0.8, 0.2])
    })
    return df

# 數據預處理
def preprocess_data(df):
    df = df.drop(columns=["商家 ID"])  # 移除非數值欄位
    X = df.drop(columns=["風險狀態"])  # 特徵值
    y = df["風險狀態"]  # 標籤
    return train_test_split(X, y, test_size=0.3, random_state=42)

# 訓練模型
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# 評估模型
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("AUC-ROC:", roc_auc_score(y_test, y_prob))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# 主流程
if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    joblib.dump(model, "fraud_model.pkl")  # 儲存模型
