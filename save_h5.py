import tensorflow as tf
import os

# --- 1. 設計図の手動定義 ---
def build_asl_model(T=40, land_dim=225, num_classes=9):
    inputs = tf.keras.Input(shape=(T, land_dim))
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    x = tf.keras.layers.LSTM(256, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(128)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)

# --- 2. 実行部分 ---
model = build_asl_model() # 器を作る

# ステップ1で作った重みファイルを読み込む
weights_path = 'models/asl_weights_only.weights.h5'

if os.path.exists(weights_path):
    model.load_weights(weights_path)
    # Colabでエラーが出ない「旧形式(H5)」で丸ごと保存
    model.save('models/asl_model_final.h5') 
    print("🎉 'asl_model_final.h5' が完成しました！これをColabにアップしてください")
else:
    print("❌ 重みファイルが見つかりません。先にステップ1をやってください")