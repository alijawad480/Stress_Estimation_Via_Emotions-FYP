# =============================================================================
# config.py — Central Configuration
# =============================================================================
import os

DATA_PATH      = "/content/drive/MyDrive/Dataset/mydata.parquet"
CHECKPOINT_DIR = "/content/drive/MyDrive/checkpoints"
RESULTS_DIR    = "/content/drive/MyDrive/results"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)

TEXT_COL   = "text_emb"
AUDIO_COL  = "audio_emb"
VISUAL_COL = "visual_emb"

TARGET_COLS   = ["label_anger","label_fear","label_happy",
                 "label_sad","label_surprise","label_disgust"]
EMOTION_NAMES = ["anger","fear","happy","sad","surprise","disgust"]
N_EMOTIONS    = 6

TEXT_DIM   = 300
AUDIO_DIM  = 74
VISUAL_DIM = 713
COMMON_DIM = 128

BATCH_SIZE    = 32
EPOCHS        = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
DROPOUT       = 0.3
PATIENCE      = 7
LR_PATIENCE   = 3
LR_FACTOR     = 0.5
RANDOM_SEED   = 42

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20

RUSSELL_COORDS = {
    "anger":    {"valence":-0.60,"arousal": 0.80},
    "fear":     {"valence":-0.70,"arousal": 0.70},
    "happy":    {"valence": 0.80,"arousal": 0.40},
    "sad":      {"valence":-0.70,"arousal":-0.20},
    "surprise": {"valence": 0.40,"arousal": 0.70},
    "disgust":  {"valence":-0.60,"arousal": 0.20},
}

STRESS_LOW_MAX      = 0.33
STRESS_MODERATE_MAX = 0.67

EMOTION_PRESENCE_THRESHOLD = 1.0

LLM_MODEL_NAME     = "mistralai/Mistral-7B-Instruct-v0.3"
LLM_MAX_NEW_TOKENS = 200
LLM_TEMPERATURE    = 0.75

N_ACTIONS          = 4
RL_GAMMA           = 0.99
RL_LR_ACTOR        = 1e-3
RL_LR_CRITIC       = 2e-3
RL_LR_DQN          = 5e-4
RL_EPISODES        = 3000
RL_ENTROPY_COEF    = 0.05
RL_VALUE_COEF      = 0.5
RL_MAX_GRAD_NORM   = 1.0

DQN_EPSILON_START  = 1.0
DQN_EPSILON_END    = 0.05
DQN_EPSILON_DECAY  = 800
DQN_MEMORY_SIZE    = 20000
DQN_BATCH_SIZE     = 256
DQN_TARGET_UPDATE  = 25
DQN_PER_ALPHA      = 0.6
DQN_PER_BETA_START = 0.4
DQN_PER_BETA_END   = 1.0

MODEL_NAMES       = ["MulT","TFN","LMF","MFN","SelfMM"]
FUSION_STRATEGIES = ["early","late","cross_modal"]
