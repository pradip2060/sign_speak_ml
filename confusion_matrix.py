import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# 既存の設定をインポート
from asl_config import ASL_CLASSES, VIDEO_DIR, MODEL_DIR, GESTURE_MODEL, T, LAND_DIM, SEED
# 学習時と同じロード関数が必要なため、train.pyからインポートするか、ここに定義します
from train_gesture import load_dataset 

def main():
    # 1. モデルの読み込み
    model_path = os.path.join(MODEL_DIR, GESTURE_MODEL)
    if not os.path.exists(model_path):
        print(f"❌ モデルが見つかりません: {model_path}")
        return
    
    model = tf.keras.models.load_model(model_path)
    print(f"✅ モデルをロードしました: {GESTURE_MODEL}")

    # 2. データの読み込み（学習時と同じシード値で分割）
    print("📦 データを読み込んでいます...")
    X, y, _ = load_dataset(VIDEO_DIR, ASL_CLASSES, T=T)
    
    # 学習時と全く同じ分割を行い、検証データ(val)のみを取り出す
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # 3. 予測の実行
    print("🧠 予測を実行中...")
    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 4. 混同行列の計算
    cm = confusion_matrix(y_val, y_pred)
    
    # 5. 可視化
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=ASL_CLASSES, yticklabels=ASL_CLASSES)
    plt.title(f'Confusion Matrix: {GESTURE_MODEL}')
    plt.ylabel('Actual Label (正解)')
    plt.xlabel('Predicted Label (予測)')
    
    # 画像として保存
    save_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"📊 グラフを保存しました: {save_path}")
    
    # 詳細なレポート（適合率、再現率など）も表示
    print("\n📝 分類レポート:")
    print(classification_report(y_val, y_pred, target_names=ASL_CLASSES))
    
    plt.show()

if __name__ == "__main__":
    main()