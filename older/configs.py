# ========================
# DATASET SETTINGS
# ========================

# Your CT/MRI dataset directories
TRAIN_IMAGES_PATH = "dataset/train/"
VALIDATION_IMAGES_PATH = "dataset/val/"
TEST_IMAGES_PATH = "dataset/test/"

# ========================
# MODEL INPUT / OUTPUT
# ========================

# Clinical images are grayscale → 1 channel
IMAGE_SIZE = (256, 256, 1)      # height, width, channels

# Watermark is 16x16 → flattened vector of 256 numbers
WATERMARK_SIZE = (16 * 16,)     # 256 bits

# ========================
# TRAINING HYPERPARAMETERS
# ========================

EPOCHS = 100       # 60 million is wrong! Use a normal number
BATCH_SIZE = 10
LEARNING_RATE = 0.001
ATTACK_MAX_ID = 6

# Loss weight balancing
IMAGE_LOSS_WEIGHT = 33.0
WATERMARK_LOSS_WEIGHT = 0.2

# Where to save trained models
MODEL_OUTPUT_PATH = "pure_wavelet/"
