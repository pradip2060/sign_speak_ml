import sys
from unittest.mock import MagicMock

# tensorflow_decision_forests がなくてもエラーにならないようにダミーを作成
sys.modules["tensorflow_decision_forests"] = MagicMock()

import tensorflow as tf
import tensorflowjs as tfjs
import os

# --- 以降は前のコードと同じ ---
model_path = 'models/asl_lstm_landmarks.keras'
model = tf.keras.models.load_model(model_path)

output_dir = 'tfjs_model'
tfjs.converters.save_keras_model(model, output_dir)
print(f"✅ Conversion complete! Saved to: {output_dir}")