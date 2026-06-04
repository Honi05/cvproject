from __future__ import annotations
import shutil
from pathlib import Path

import kagglehub

from cvchess.config import KAGGLE_DATASET, RAW_DIR
from cvchess.logging_utils import get_logger

log = get_logger("download")


def is_cached(target_dir: Path) -> bool:
    """True if both train/ and test/ exist and contain at least one image."""
    target_dir = Path(target_dir)
    train, test = target_dir / "train", target_dir / "test"
    if not (train.is_dir() and test.is_dir()):
        return False
    has_train = any(train.glob("*.png")) or any(train.glob("*.jpeg"))
    has_test = any(test.glob("*.png")) or any(test.glob("*.jpeg"))
    return has_train and has_test


def ensure_dataset(target_dir: Path | None = None) -> Path:
    """Download koryakinp/chess-positions unless already cached on disk."""
    target_dir = Path(target_dir) if target_dir else RAW_DIR / "chess-positions"
    if is_cached(target_dir):
        log.info("Dataset cache hit at %s — skipping download", target_dir)
        return target_dir
    log.info("Downloading %s via kagglehub ...", KAGGLE_DATASET)
    src = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    log.info("Downloaded to %s", src)
    target_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("train", "test"):
        s = _find_subdir(src, sub)
        if s and s.resolve() != (target_dir / sub).resolve():
            if (target_dir / sub).exists():
                shutil.rmtree(target_dir / sub)
            shutil.copytree(s, target_dir / sub)
    if not is_cached(target_dir):
        raise RuntimeError(f"Download did not yield train/ and test/ in {target_dir}")
    return target_dir


def _find_subdir(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.is_dir():
        return direct
    matches = list(root.rglob(name))
    return matches[0] if matches else None
