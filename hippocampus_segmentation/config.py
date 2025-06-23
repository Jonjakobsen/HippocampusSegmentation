import torch

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_IMG_DIR = "data/train/images"
TRAIN_MASK_DIR = "data/train/masks"
