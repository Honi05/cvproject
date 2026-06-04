from __future__ import annotations
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
OUTPUTS_DIR = REPO_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
OPTUNA_DB = f"sqlite:///{OUTPUTS_DIR / 'optuna.db'}"

KAGGLE_DATASET = "koryakinp/chess-positions"

IMG_SIZE = 400
GRID = 8
CELL = IMG_SIZE // GRID  # 50

CNN_CLASSES = [
    "empty",
    "wP", "wN", "wB", "wR", "wQ", "wK",
    "bP", "bN", "bB", "bR", "bQ", "bK",
]
YOLO_CLASSES = CNN_CLASSES[1:]

FEN_TO_CNN = {
    "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
    "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12,
}
CNN_TO_YOLO = {i: i - 1 for i in range(1, 13)}  # cnn idx -> yolo idx

MEMORY_FRACTION = 0.9

def ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, CACHE_DIR, OUTPUTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
