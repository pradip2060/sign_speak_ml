ASL_CLASSES = [
    "Hello",
    "I_Love_You",
    "Nothing",
    "Thank_You",
    "YES",
    "NO",
    "SORRY",
    "HELP",
    "PEACE",
]

# 表示用マッピング
LABEL_MAP = {
    "Hello": "Hello",
    "I_Love_You": "I Love You",
    "Nothing": "Nothing",
    "Thank_You": "Thank You",
    "YES": "Yes",
    "NO": "No",
    "SORRY": "Sorry",
    "HELP": "Help",
    "PEACE": "Peace",
}

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

HAND_CONFIDENCE = 0.5   # 手のランドマークの接続線の太さ
POSE_CONFIDENCE = 0.65  # ポーズのランドマークの接続線の太さ

T = 40                            # 固定フレーム長（学習に合わせる）
POSE_DIM = 33 * 3                 # 99
HANDS_DIM = 21 * 2 * 3            # 126
LAND_DIM = POSE_DIM + HANDS_DIM   # 225

# 指の比重デフォルト（少し下げて安定させる）
HAND_WEIGHT = 1.3  

# z座標のスケールを縮小（0.3〜0.5 推奨）
Z_SCALE = 0.3                     

# train_gesture2.py で使用
HAND_WEIGHT_MAP = {
    "YES": 1.5,
    "NO": 1.5,
    "PEACE": 1.8,
    "I_Love_You": 1.8,
}

# train_gesture2.py で使用
IMPORTANT_LM_WEIGHTS = {
    4:  1.5,  # 親指
    8:  1.3,  # 人差し指
    20: 1.5,  # 小指
    12: 1.2,  # 中指
}

BATCH_SIZE = 8
EPOCHS = 60
SEED = 42

PROB_THRESH = 0.7          # 予測確率の閾値
PRED_SMOOTH = 5            # 予測平滑化フレーム

START_MOV_THRESH = 0.008   # 動作開始閾値
STOP_MOV_THRESH = 0.003    # 動作停止閾値

START_FRAMES = 5          # 動作開始フレーム数
STOP_FRAMES = 10          # 動作停止フレーム数


GESTURE_MODEL="asl_lstm_landmarks.keras"
USE_MODEL="asl_multimodal_model_hands"
ALPHABET_DIR = "asl_alphabet_train"
VIDEO_DIR = "asl_videos"
DATA_DIR = "asl_words_train"
MODEL_DIR = "models"
TEST_DIR ="asl_words_test"
EXTENTION = "keras"
IMAGE_SIZE = (64, 64)
MARGIN = 100
