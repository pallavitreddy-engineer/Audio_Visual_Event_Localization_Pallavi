"""
Central configuration for the AVE (Audio-Visual Event) localization project.
All hyperparameters, paths, and constants in one place.
"""

import os
import torch

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "AVE_Dataset")
VIDEO_DIR = os.path.join(DATASET_ROOT, "AVE")
ANNOTATIONS_FILE = os.path.join(DATASET_ROOT, "Annotations.txt")
TRAIN_SET_FILE = os.path.join(DATASET_ROOT, "trainSet.txt")
VAL_SET_FILE = os.path.join(DATASET_ROOT, "valSet.txt")
TEST_SET_FILE = os.path.join(DATASET_ROOT, "testSet.txt")

# Directory for pre-extracted features (saves re-encoding every epoch)
FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

# ──────────────────────────────────────────────
# Audio settings
# ──────────────────────────────────────────────
AUDIO_SAMPLE_RATE = 16000          # VGGish expects 16 kHz
AUDIO_ORIGINAL_SR = 44100          # Native sample rate of AVE videos
AUDIO_NUM_CHANNELS = 1             # Mono
CLIP_DURATION = 10                 # Each clip is 10 seconds
NUM_SEGMENTS = 10                  # 10 one-second segments per clip
SAMPLES_PER_SEGMENT = AUDIO_SAMPLE_RATE  # 16,000 samples per 1-second chunk

# ──────────────────────────────────────────────
# Video settings
# ──────────────────────────────────────────────
VIDEO_FPS = 8                      # Frames per second to extract per 1-sec chunk
VIDEO_FRAME_SIZE = (112, 112)      # R(2+1)D input resolution
VIDEO_NUM_FRAMES_PER_SEGMENT = VIDEO_FPS  # 8 frames per 1-second segment

# ──────────────────────────────────────────────
# Model dimensions
# ──────────────────────────────────────────────
AUDIO_EMBED_DIM = 128              # VGGish output dimension
VIDEO_FEATURE_DIM = 512            # R(2+1)D last conv layer output channels
VIDEO_SPATIAL_SIZE = 7             # 7x7 spatial grid from R(2+1)D
VIDEO_NUM_REGIONS = VIDEO_SPATIAL_SIZE ** 2  # 49 spatial regions

LSTM_HIDDEN_DIM = 128              # Hidden size for each direction in Bi-LSTM
LSTM_NUM_LAYERS = 1                # Number of LSTM layers
LSTM_OUTPUT_DIM = LSTM_HIDDEN_DIM * 2  # 256 (bidirectional)

FC_HIDDEN_DIM = 128                # Hidden dim for classification MLP
# 29-class problem: 28 real event categories + Background (Guide §1.2 and §4.4)
NUM_CLASSES = 29
BACKGROUND_IDX = 28               # Index of the Background pseudo-class

# ──────────────────────────────────────────────
# Training hyperparameters
# ──────────────────────────────────────────────
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5

# Modality dropout probability (risk mitigation for modality dominance)
MODALITY_DROPOUT_PROB = 0.1        # 10% chance of dropping one modality

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────
# Event categories (28 categories from AVE dataset)
# ──────────────────────────────────────────────
CATEGORIES = [
    "Accordion",
    "Acoustic guitar",
    "Baby cry, infant cry",
    "Banjo",
    "Bark",
    "Bus",
    "Cat",
    "Chainsaw",
    "Church bell",
    "Clock",
    "Female speech, woman speaking",
    "Fixed-wing aircraft, airplane",
    "Flute",
    "Frying (food)",
    "Goat",
    "Helicopter",
    "Horse",
    "Male speech, man speaking",
    "Mandolin",
    "Motorcycle",
    "Race car, auto racing",
    "Rodents, rats, mice",
    "Shofar",
    "Toilet flush",
    "Train horn",
    "Truck",
    "Ukulele",
    "Violin, fiddle",
]

CATEGORY_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}
IDX_TO_CATEGORY = {idx: cat for idx, cat in enumerate(CATEGORIES)}
NUM_CATEGORIES = len(CATEGORIES)

# Full 29-class list (28 events + Background) used by the model output layer
ALL_CLASSES = CATEGORIES + ["Background"]
ALL_CLASS_TO_IDX = {cat: idx for idx, cat in enumerate(ALL_CLASSES)}

# ──────────────────────────────────────────────
# MLflow
# ──────────────────────────────────────────────
MLFLOW_TRACKING_URI = f"file:{os.path.join(PROJECT_ROOT, 'mlruns')}"
MLFLOW_EXPERIMENT_NAME = "AVE_audio_visual_localisation"
