"""
Central settings for HandChord Studio.
Everything you may want to customize (gestures, chords, sensitivities) lives here.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REC_DIR = ROOT / "recordings"
SESSION_DIR = ROOT / "sessions"
DATASET_PATH = DATA_DIR / "dataset.csv"
MODEL_PATH = MODEL_DIR / "gesture_model.joblib"

# ---------------------------------------------------------------------------
# Gestures -> scale degree (0..6).  None means "silence".
# In a major scale: 0=I  1=ii  2=iii  3=IV  4=V  5=vi  6=vii(dim)
# You can rename gestures or add new ones; afterwards, collect new data
# and retrain the model.
# ---------------------------------------------------------------------------
GESTURE_DEGREES = {
    "fist": None,       # closed fist            -> silence
    "one": 0,           # one finger             -> I
    "two": 3,           # two fingers            -> IV
    "three": 4,         # three fingers          -> V
    "four": 5,          # four fingers           -> vi
    "open": 1,          # open palm              -> ii
    "thumbs_up": 2,     # thumbs up              -> iii
    "rock": 6,          # index + pinky ("rock") -> vii
}

# Rule-based mode (no model): number of extended fingers -> gesture name
RULE_LABELS = ["fist", "one", "two", "three", "four", "open"]

# Left hand: number of extended fingers -> chord quality
QUALITY_BY_LEFT_FINGERS = {0: "triad", 1: "7th", 2: "sus2", 3: "sus4"}

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44100
BLOCK_SIZE = 512
DEFAULT_BPM = 110

# ---------------------------------------------------------------------------
# Detection and stabilization
# ---------------------------------------------------------------------------
CONF_THRESHOLD = 0.60     # below this confidence the gesture counts as "unknown"
VOTE_WINDOW = 7           # number of recent frames used for voting
MIN_VOTES = 4             # minimum votes needed to change the gesture
SWAP_HANDEDNESS = False   # set True if left/right are detected swapped

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
