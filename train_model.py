import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import joblib

np.random.seed(42)

print("=" * 60)
print("開始訓練詐騙偵測模型")
print("=" * 60)

# 載入資料
print("\n1. 載入資料集...")
df = pd.read_csv('1.csv')
print(f"   資料集大小: {df.shape}")
print(f"   欄位: {df.columns.tolist()}")

# 資料探索
print("\n2. 資料探索...")
print(f"   缺失值數量: {df.isnull().sum().sum()}")
print(f"   類別分佈:")
print(f"   - 正常交易 (Class=0): {(df['Class']==0).sum()} ({(df['Class']==0).sum()/len(df)*100:.2f}%)")
print(f"   - 詐騙交易 (Class=1): {(df['Class']==1).sum()} ({(df['Class']==1).sum()/len(df)*100:.2f}%)")

# 準備特徵和標籤
print("\n3. 準備特徵和標籤...")
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

X = df.drop('Class', axis=1)
y = df['Class']

# 特徵工程
X['Amount_log'] = np.log1p(X['Amount'])
v_columns = [col for col in X.columns if col.startswith('V')]
X['V_mean'] = X[v_columns].mean(axis=1)
X['V_std'] = X[v_columns].std(axis=1)
X['V_max'] = X[v_columns].max(axis=1)
X['V_min'] = X[v_columns].min(axis=1)

print(f"   特徵數量: {X.shape[1]}")
print(f"   樣本數量: {X.shape[0]}")

# 資料分割 70/30
print("\n4. 分割資料集 (70% 訓練, 30% 測試)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"   訓練集大小: {X_train.shape}")
print(f"   測試集大小: {X_test.shape}")

# 標準化特徵
print("\n5. 標準化特徵...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 處理類別不平衡
print("\n6. 處理類別不平衡 (SMOTE)...")
print(f"   原始訓練集類別分佈:")
print(f"   - Class 0: {(y_train==0).sum()}")
print(f"   - Class 1: {(y_train==1).sum()}")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"   平衡後訓練集類別分佈:")
print(f"   - Class 0: {(y_train_balanced==0).sum()}")
print(f"   - Class 1: {(y_train_balanced==1).sum()}")

# 訓練模型
print("\n7. 訓練模型...")

print("\n   訓練 Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_balanced, y_train_balanced)

print("   訓練 Logistic Regression...")
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='liblinear'
)
lr_model.fit(X_train_balanced, y_train_balanced)

# 模型評估
print("\n8. 模型評估...")

def evaluate_model(model, X_test, y_test, model_name):
    print(f"\n   {model_name} 評估結果:")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n   分類報告:")
    print(classification_report(y_test, y_pred, 
                              target_names=['正常', '詐騙'],
                              digits=4))
    
    cm = confusion_matrix(y_test, y_pred)
    print("\n   混淆矩陣:")
    print(f"   [[TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}]")
    print(f"    [FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}]]")
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\n   AUC-ROC 分數: {auc_score:.4f}")
    
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    
    print(f"\n   關鍵指標:")
    print(f"   - 準確率 (Accuracy): {accuracy:.4f}")
    print(f"   - 精確率 (Precision): {precision:.4f}")
    print(f"   - 召回率 (Recall): {recall:.4f}")
    
    return auc_score

rf_auc = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")
lr_auc = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")

# 選擇最佳模型
print("\n9. 選擇最佳模型...")
if rf_auc >= lr_auc:
    best_model = rf_model
    print(f"   選擇 Random Forest (AUC: {rf_auc:.4f})")
else:
    best_model = lr_model
    print(f"   選擇 Logistic Regression (AUC: {lr_auc:.4f})")

# 特徵重要性
if isinstance(best_model, RandomForestClassifier):
    print("\n10. 特徵重要性 (Top 10):")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"    {row['feature']:15s}: {row['importance']:.4f}")

# 儲存模型評估結果
print("\n11. 儲存模型評估結果...")
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
cm = confusion_matrix(y_test, y_pred)

model_metrics = {
    'accuracy': (cm[0,0] + cm[1,1]) / cm.sum(),
    'precision': cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0,
    'recall': cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0,
    'auc_roc': roc_auc_score(y_test, y_pred_proba),
    'confusion_matrix': cm.tolist(),
    'total_samples': len(y_test),
    'fraud_samples': (y_test == 1).sum(),
    'normal_samples': (y_test == 0).sum()
}

joblib.dump(model_metrics, 'model_metrics.pkl')
print("   模型指標已儲存為 model_metrics.pkl")

# 儲存模型和相關物件
print("\n12. 儲存模型...")
joblib.dump(best_model, 'fraud_model.pkl')
print("   模型已儲存為 fraud_model.pkl")

joblib.dump(scaler, 'fraud_scaler.pkl')
print("   Scaler 已儲存為 fraud_scaler.pkl")

feature_names = X.columns.tolist()
joblib.dump(feature_names, 'fraud_features.pkl')
print("   特徵名稱已儲存為 fraud_features.pkl")

print("\n" + "=" * 60)
print("訓練完成！")
print("=" * 60)
print("\n模型檔案:")
print("1. fraud_model.pkl - 訓練好的模型")
print("2. fraud_scaler.pkl - 特徵標準化器")
print("3. fraud_features.pkl - 特徵名稱列表")
print("4. model_metrics.pkl - 模型評估指標")