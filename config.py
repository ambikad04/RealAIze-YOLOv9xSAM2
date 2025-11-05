import os
import torch
from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
EMBEDDINGS_DIR = ROOT_DIR / "embeddings"

# Create directories if they don't exist
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Model paths
YOLO_WEIGHTS = WEIGHTS_DIR / "yolov8m-oiv7.pt"
SAM_WEIGHTS = WEIGHTS_DIR / "sam_vit_h_4b8939.pth"

# Device settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Detection settings
CONFIDENCE_THRESHOLD = 0.1
IOU_THRESHOLD = 0.45

# Visualization settings
SHOW_BOXES = True
SHOW_LABELS = True
SHOW_CONFIDENCE = True
SHOW_SEGMENTATION = True

# Colors for visualization (BGR format)
COLORS = {
    "box": (0, 255, 0),  # Green
    "text": (255, 255, 255),  # White
    "mask": (0, 255, 0),  # Green
}

# Text settings
FONT = 0
FONT_SCALE = 0.5
FONT_THICKNESS = 1
TEXT_PADDING = 5

# Model settings
YOLO_IMG_SIZE = 640
SAM_IMG_SIZE = 1024

# Embedding settings
DINO_MODEL = "facebook/dinov2-base"
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "faiss_index.bin"
METADATA_PATH = EMBEDDINGS_DIR / "metadata.json"
SIMILARITY_THRESHOLD = 0.8
MAX_SIMILAR_OBJECTS = 5 