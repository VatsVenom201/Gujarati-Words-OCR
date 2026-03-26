import torch
import os

BASE_DIR = r"d:\PyCharm\Gujarati_text"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

TRAIN_GT = os.path.join(TRAIN_DIR, "train_gt.txt")
VAL_GT = os.path.join(VAL_DIR, "val_gt.txt")
TEST_GT = os.path.join(TEST_DIR, "test_gt.txt")

IMG_HEIGHT = 32
MAX_WIDTH = 256 # Optional max width for memory control
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
HIDDEN_SIZE = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
