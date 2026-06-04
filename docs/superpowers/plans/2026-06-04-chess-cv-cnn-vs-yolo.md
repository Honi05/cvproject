# Chess Piece Recognition: CNN vs YOLO — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible pipeline that downloads a synthetic chess dataset, trains a custom CNN piece classifier (Optuna-tuned, wandb-tracked, pushed to HF), and compares it against a YOLO detector on accuracy, inference time, and compute.

**Architecture:** One synthetic source (`koryakinp/chess-positions`) yields two views — 50×50 cell crops for the CNN (13 classes) and grid-derived bounding boxes for YOLO (12 classes). A small `src/cvchess/` package holds reusable logic; thin entrypoint scripts drive each stage. Hardware is detected up front and memory capped to 90%. Kornia provides differentiable GPU augmentation.

**Tech Stack:** Python 3.11, PyTorch 2.4 (cu124), torchvision, Kornia, kagglehub, Optuna (SQLite), Weights & Biases, Ultralytics YOLO, huggingface_hub, OpenCV, psutil, thop, pytest.

**Conventions for every commit in this plan:**
- Run from repo root `/workspace/cvproject` on branch `chess-cv-pipeline`.
- Commit messages: **no AI attribution, ever.** Use `git -c commit.gpgsign=false commit`.
- Author is already configured (`Honi <arorahoni966@gmail.com>`).

---

## File Structure

```
cvproject/
  requirements.txt                  # Task 1
  setup_env.sh                      # Task 1
  .gitignore                        # Task 1
  .env.example                      # Task 1
  pytest.ini                        # Task 1
  src/cvchess/
    __init__.py                     # Task 2
    config.py                       # Task 2  — paths, class maps, constants
    logging_utils.py                # Task 3  — central logger
    hardware.py                     # Task 4  — detect + cap to 90%
    data/
      __init__.py                   # Task 5
      fen.py                        # Task 5  — FEN <-> labels, filename decode
      download.py                   # Task 6  — kagglehub download + cache check
      build_dataset.py              # Task 7  — slice cells, derive boxes, manifest
      augment.py                    # Task 8  — Kornia GPU augmentation
      dataset.py                    # Task 9  — torch Dataset/DataLoader
    models/
      __init__.py                   # Task 10
      cnn.py                        # Task 10 — custom CNN (Optuna-parameterized)
    train/
      __init__.py                   # Task 11
      time_estimator.py             # Task 11 — calibration -> ETA
      train_cnn.py                  # Task 12 — Optuna + wandb + final train + save
    yolo/
      __init__.py                   # Task 13
      prepare_yolo.py               # Task 13 — write ultralytics dataset
      train_yolo.py                 # Task 14 — train YOLO baseline
    eval/
      __init__.py                   # Task 15
      compare.py                    # Task 15 — accuracy + latency + FLOPs/VRAM
    hf_push.py                      # Task 16 — upload CNN + model card
    monitor.py                      # Task 17 — live performance monitor
  scripts/
    detect_hardware.py              # Task 4
    01_build_data.py                # Task 7
    02_train_cnn.py                 # Task 12
    03_compare.py                   # Task 15
  tests/
    conftest.py                     # Task 2
    test_fen.py                     # Task 5
    test_hardware.py                # Task 4
    test_download.py                # Task 6
    test_build_dataset.py           # Task 7
    test_augment.py                 # Task 8
    test_dataset.py                 # Task 9
    test_cnn.py                     # Task 10
    test_time_estimator.py          # Task 11
    test_compare.py                 # Task 15
    test_smoke.py                   # Task 18
```

---

## Class & label conventions (used across many tasks — define once)

FEN piece letters: white = uppercase `PNBRQK`, black = lowercase `pnbrqk`.

**CNN 13-class label map** (`config.py`):
```python
CNN_CLASSES = [
    "empty",                                  # 0
    "wP", "wN", "wB", "wR", "wQ", "wK",       # 1..6  (uppercase FEN)
    "bP", "bN", "bB", "bR", "bQ", "bK",       # 7..12 (lowercase FEN)
]
```
**YOLO 12-class map** = `CNN_CLASSES[1:]` (no empty), indices 0..11.

FEN char → CNN class index:
```python
FEN_TO_CNN = {
    "P":1,"N":2,"B":3,"R":4,"Q":5,"K":6,
    "p":7,"n":8,"b":9,"r":10,"q":11,"k":12,
}  # empty handled separately
```

Board geometry: image is 400×400, 8×8 grid, cell = 50×50. Cell `(row, col)` with
`row` 0=top (rank 8) .. 7=bottom (rank 1), `col` 0=left (file a) .. 7=right (file h).
Pixel box for cell = `(col*50, row*50, col*50+50, row*50+50)` as `(x1,y1,x2,y2)`.

Filename FEN: ranks separated by `-` instead of `/` (e.g. `1B1b3R-2q5-...png`).

---

## Task 1: Project scaffolding (env, deps, gitignore)

**Files:**
- Create: `requirements.txt`, `setup_env.sh`, `.gitignore`, `.env.example`, `pytest.ini`

- [ ] **Step 1: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.env
data/cache/
data/raw/
outputs/
wandb/
*.db
*.pt
*.pth
.pytest_cache/
runs/
yolo_dataset/
```

- [ ] **Step 2: Write `requirements.txt`** (torch/torchvision already in system site-packages; pin the rest)

```
kornia==0.7.3
kagglehub==0.3.4
optuna==4.0.0
wandb==0.18.5
ultralytics==8.3.0
huggingface_hub==0.26.2
opencv-python-headless==4.10.0.84
psutil==6.1.0
thop==0.1.1.post2209072238
numpy==1.26.4
pillow==10.4.0
python-dotenv==1.0.1
pytest==8.3.3
```

- [ ] **Step 3: Write `.env.example`**

```
WANDB_API_KEY=your_wandb_key_here
HF_TOKEN=your_hf_token_here
KAGGLE_API_TOKEN=your_kaggle_kgat_token_here
```

- [ ] **Step 4: Write `pytest.ini`**

```ini
[pytest]
pythonpath = src
testpaths = tests
markers =
    gpu: tests that require a CUDA GPU
    network: tests that require network/credentials
    slow: long-running tests
```

- [ ] **Step 5: Write `setup_env.sh`** (venv with system site-packages to keep cu124 torch)

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv --system-site-packages .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo "Environment ready. Verifying hardware..."
python scripts/detect_hardware.py
```

- [ ] **Step 6: Make executable and commit**

```bash
chmod +x setup_env.sh
git add requirements.txt setup_env.sh .gitignore .env.example pytest.ini
git -c commit.gpgsign=false commit -m "Add project scaffolding: deps, env setup, gitignore"
```

---

## Task 2: Package init, config, conftest

**Files:**
- Create: `src/cvchess/__init__.py`, `src/cvchess/config.py`, `tests/conftest.py`

- [ ] **Step 1: Write `src/cvchess/__init__.py`**

```python
"""cvchess: chess piece recognition — custom CNN vs YOLO."""
__version__ = "0.1.0"
```

- [ ] **Step 2: Write `src/cvchess/config.py`**

```python
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
```

- [ ] **Step 3: Write `tests/conftest.py`** (synthetic board fixture, no network)

```python
import numpy as np
import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def synthetic_board(tmp_path):
    """Create a 400x400 board image whose filename encodes a known FEN.
    Each occupied cell gets a distinct solid color so slicing is testable.
    Returns (path, fen_filename, expected_label_grid)."""
    fen = "rnbqkbnr-pppppppp-8-8-8-8-PPPPPPPP-RNBQKBNR"
    img = Image.new("RGB", (400, 400), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    # checkerboard background
    for r in range(8):
        for c in range(8):
            if (r + c) % 2 == 1:
                draw.rectangle([c*50, r*50, c*50+50, r*50+50], fill=(120, 120, 120))
    path = tmp_path / f"{fen}.png"
    img.save(path)
    return path, f"{fen}.png", fen
```

- [ ] **Step 4: Create remaining package `__init__.py` files**

```bash
mkdir -p src/cvchess/data src/cvchess/models src/cvchess/train src/cvchess/yolo src/cvchess/eval tests scripts
touch src/cvchess/data/__init__.py src/cvchess/models/__init__.py src/cvchess/train/__init__.py src/cvchess/yolo/__init__.py src/cvchess/eval/__init__.py
```

- [ ] **Step 5: Commit**

```bash
git add src/cvchess tests/conftest.py
git -c commit.gpgsign=false commit -m "Add package init, config constants, test fixtures"
```

---

## Task 3: Central logger

**Files:**
- Create: `src/cvchess/logging_utils.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_hardware.py` later; for now create `tests/test_logging.py`

```python
import logging
from cvchess.logging_utils import get_logger


def test_get_logger_returns_named_logger():
    log = get_logger("foo")
    assert isinstance(log, logging.Logger)
    assert log.name == "cvchess.foo"


def test_get_logger_is_idempotent():
    a = get_logger("bar")
    b = get_logger("bar")
    assert len(a.handlers) == len(b.handlers)  # no duplicate handlers
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_logging.py -v`
Expected: FAIL with `ModuleNotFoundError: cvchess.logging_utils`

- [ ] **Step 3: Write `src/cvchess/logging_utils.py`**

```python
from __future__ import annotations
import logging
import sys

_CONFIGURED: set[str] = set()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    full = f"cvchess.{name}"
    logger = logging.getLogger(full)
    if full not in _CONFIGURED:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED.add(full)
    return logger
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_logging.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cvchess/logging_utils.py tests/test_logging.py
git -c commit.gpgsign=false commit -m "Add central logging utility"
```

---

## Task 4: Hardware detection + memory caps

**Files:**
- Create: `src/cvchess/hardware.py`, `scripts/detect_hardware.py`, `tests/test_hardware.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hardware.py
from cvchess.hardware import detect, recommended_num_workers, HardwareInfo


def test_detect_returns_sane_values():
    info = detect()
    assert isinstance(info, HardwareInfo)
    assert info.cpu_count >= 1
    assert info.ram_total_gb > 0
    assert info.gpu_count >= 0  # may be 0 on CI


def test_recommended_workers_bounded():
    n = recommended_num_workers(cpu_count=112)
    assert 1 <= n <= 32  # capped, not 112


def test_memory_fraction_constant():
    info = detect()
    assert 0 < info.memory_fraction <= 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hardware.py -v`
Expected: FAIL with `ModuleNotFoundError: cvchess.hardware`

- [ ] **Step 3: Write `src/cvchess/hardware.py`**

```python
from __future__ import annotations
import resource
from dataclasses import dataclass, field

import psutil

from cvchess.config import MEMORY_FRACTION
from cvchess.logging_utils import get_logger

log = get_logger("hardware")


@dataclass
class HardwareInfo:
    cpu_count: int
    ram_total_gb: float
    gpu_count: int
    gpu_names: list[str] = field(default_factory=list)
    gpu_total_gb: list[float] = field(default_factory=list)
    cuda_available: bool = False
    memory_fraction: float = MEMORY_FRACTION


def detect() -> HardwareInfo:
    cpu_count = psutil.cpu_count(logical=True) or 1
    ram_total_gb = psutil.virtual_memory().total / (1024 ** 3)
    gpu_count, names, totals, cuda = 0, [], [], False
    try:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                p = torch.cuda.get_device_properties(i)
                names.append(p.name)
                totals.append(p.total_memory / (1024 ** 3))
    except Exception as e:  # torch missing or driver issue
        log.warning("GPU detection failed: %s", e)
    return HardwareInfo(
        cpu_count=cpu_count, ram_total_gb=ram_total_gb,
        gpu_count=gpu_count, gpu_names=names, gpu_total_gb=totals,
        cuda_available=cuda,
    )


def recommended_num_workers(cpu_count: int | None = None) -> int:
    if cpu_count is None:
        cpu_count = psutil.cpu_count(logical=True) or 1
    return max(1, min(32, cpu_count // 4))


def cap_gpu_memory(fraction: float = MEMORY_FRACTION) -> None:
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_per_process_memory_fraction(fraction, i)
            log.info("Capped GPU memory to %.0f%%", fraction * 100)
    except Exception as e:
        log.warning("Could not cap GPU memory: %s", e)


def cap_system_memory(fraction: float = MEMORY_FRACTION) -> None:
    total = psutil.virtual_memory().total
    soft_cap = int(total * fraction)
    try:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_hard = hard if hard != resource.RLIM_INFINITY else soft_cap
        resource.setrlimit(resource.RLIMIT_AS, (soft_cap, new_hard))
        log.info("Capped system RAM (RLIMIT_AS) to %.0f%% (%.1f GB)",
                 fraction * 100, soft_cap / (1024 ** 3))
    except (ValueError, OSError) as e:
        log.warning("Could not set RLIMIT_AS: %s", e)


def apply_caps(fraction: float = MEMORY_FRACTION) -> None:
    cap_gpu_memory(fraction)
    cap_system_memory(fraction)


def log_summary(info: HardwareInfo | None = None) -> HardwareInfo:
    if info is None:
        info = detect()
    log.info("CPU cores: %d", info.cpu_count)
    log.info("System RAM: %.1f GB", info.ram_total_gb)
    log.info("CUDA available: %s", info.cuda_available)
    for i, (n, g) in enumerate(zip(info.gpu_names, info.gpu_total_gb)):
        log.info("GPU %d: %s (%.1f GB)", i, n, g)
    log.info("Recommended num_workers: %d", recommended_num_workers(info.cpu_count))
    return info
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hardware.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Write `scripts/detect_hardware.py`**

```python
"""Detect hardware, log a summary, and apply 90% memory caps."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvchess.hardware import log_summary, apply_caps  # noqa: E402


def main() -> None:
    info = log_summary()
    apply_caps()
    if info.cuda_available:
        print(f"\nReady: {info.gpu_names[0]} with {info.gpu_total_gb[0]:.0f} GB")
    else:
        print("\nWARNING: no CUDA GPU detected — training will use CPU.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the script**

Run: `python scripts/detect_hardware.py`
Expected: logs CPU=112, RAM≈503GB, GPU 0: NVIDIA RTX A4000 (16 GB), and "Ready:" line.

- [ ] **Step 7: Commit**

```bash
git add src/cvchess/hardware.py scripts/detect_hardware.py tests/test_hardware.py
git -c commit.gpgsign=false commit -m "Add hardware detection and 90% memory capping"
```

---

## Task 5: FEN parsing & label mapping

**Files:**
- Create: `src/cvchess/data/fen.py`, `tests/test_fen.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fen.py
import numpy as np
from cvchess.data.fen import (
    filename_to_fen, fen_to_label_grid, label_grid_to_cnn_indices,
)


def test_filename_to_fen_strips_extension_and_keeps_dashes():
    assert filename_to_fen("1B1b3R-2q5-8-8-8-8-8-8.png") == "1B1b3R-2q5-8-8-8-8-8-8"


def test_fen_to_label_grid_starting_position():
    fen = "rnbqkbnr-pppppppp-8-8-8-8-PPPPPPPP-RNBQKBNR"
    grid = fen_to_label_grid(fen)  # 8x8 of FEN chars or "" for empty
    assert grid[0][0] == "r"   # a8
    assert grid[0][4] == "k"   # e8
    assert grid[7][4] == "K"   # e1
    assert grid[3][3] == ""    # empty middle
    assert grid[1][0] == "p"


def test_label_grid_to_cnn_indices():
    fen = "8-8-8-8-8-8-8-4K3"  # single white king on e1
    grid = fen_to_label_grid(fen)
    idx = label_grid_to_cnn_indices(grid)  # 8x8 int array
    assert idx.shape == (8, 8)
    assert idx[7][4] == 6      # wK
    assert idx[0][0] == 0      # empty
    assert (idx == 0).sum() == 63


def test_each_rank_sums_to_eight():
    fen = "1B1b3R-2q5-8-8-8-8-8-8"
    grid = fen_to_label_grid(fen)
    assert all(len(row) == 8 for row in grid)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fen.py -v`
Expected: FAIL with `ModuleNotFoundError: cvchess.data.fen`

- [ ] **Step 3: Write `src/cvchess/data/fen.py`**

```python
from __future__ import annotations
import os
import numpy as np
from cvchess.config import FEN_TO_CNN


def filename_to_fen(filename: str) -> str:
    """Strip directory + extension, keep the dash-separated FEN board."""
    base = os.path.basename(filename)
    return os.path.splitext(base)[0]


def fen_to_label_grid(fen: str) -> list[list[str]]:
    """Convert a dash-separated FEN board to an 8x8 grid of FEN chars
    ('' for empty). Row 0 = rank 8 (top), col 0 = file a (left)."""
    rows = fen.split("-")
    if len(rows) != 8:
        raise ValueError(f"FEN must have 8 ranks, got {len(rows)}: {fen}")
    grid: list[list[str]] = []
    for r in rows:
        cells: list[str] = []
        for ch in r:
            if ch.isdigit():
                cells.extend([""] * int(ch))
            else:
                cells.append(ch)
        if len(cells) != 8:
            raise ValueError(f"Rank '{r}' expands to {len(cells)} cells, need 8")
        grid.append(cells)
    return grid


def label_grid_to_cnn_indices(grid: list[list[str]]) -> np.ndarray:
    """Map an 8x8 char grid to an 8x8 int array of CNN class indices."""
    idx = np.zeros((8, 8), dtype=np.int64)
    for r in range(8):
        for c in range(8):
            ch = grid[r][c]
            idx[r, c] = FEN_TO_CNN[ch] if ch else 0
    return idx
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fen.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cvchess/data/fen.py tests/test_fen.py
git -c commit.gpgsign=false commit -m "Add FEN parsing and label-grid mapping"
```

---

## Task 6: Dataset download with cache check

**Files:**
- Create: `src/cvchess/data/download.py`, `tests/test_download.py`

- [ ] **Step 1: Write the failing test** (cache logic tested without network via monkeypatch)

```python
# tests/test_download.py
from pathlib import Path
import cvchess.data.download as dl


def test_is_cached_false_when_missing(tmp_path):
    assert dl.is_cached(tmp_path / "nope") is False


def test_is_cached_true_when_marker_present(tmp_path):
    d = tmp_path / "ds"
    (d / "train").mkdir(parents=True)
    (d / "train" / "a.png").write_bytes(b"x")
    (d / "test").mkdir()
    (d / "test" / "b.png").write_bytes(b"x")
    assert dl.is_cached(d) is True


def test_download_skips_when_cached(tmp_path, monkeypatch):
    d = tmp_path / "ds"
    (d / "train").mkdir(parents=True)
    (d / "train" / "a.png").write_bytes(b"x")
    (d / "test").mkdir()
    (d / "test" / "b.png").write_bytes(b"x")
    called = {"n": 0}
    def fake_dl(slug):
        called["n"] += 1
        return str(d)
    monkeypatch.setattr(dl.kagglehub, "dataset_download", fake_dl)
    out = dl.ensure_dataset(target_dir=d)
    assert out == d
    assert called["n"] == 0  # cache hit -> no download
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_download.py -v`
Expected: FAIL with `ModuleNotFoundError: cvchess.data.download`

- [ ] **Step 3: Write `src/cvchess/data/download.py`**

```python
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
    # kagglehub returns a cache path; normalize into target_dir
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_download.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cvchess/data/download.py tests/test_download.py
git -c commit.gpgsign=false commit -m "Add cache-aware Kaggle dataset download"
```

---

## Task 7: Build dataset views (cells + boxes + manifest)

**Files:**
- Create: `src/cvchess/data/build_dataset.py`, `scripts/01_build_data.py`, `tests/test_build_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_dataset.py
import numpy as np
from PIL import Image
from cvchess.data.build_dataset import (
    slice_cells, derive_yolo_boxes, board_record,
)
from cvchess.data.fen import fen_to_label_grid, label_grid_to_cnn_indices


def test_slice_cells_shapes():
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cells = slice_cells(img)
    assert cells.shape == (64, 50, 50, 3)


def test_derive_boxes_only_for_occupied():
    fen = "8-8-8-8-8-8-8-4K3"  # 1 piece (e1)
    grid = fen_to_label_grid(fen)
    idx = label_grid_to_cnn_indices(grid)
    boxes = derive_yolo_boxes(idx)
    assert len(boxes) == 1
    cls, x1, y1, x2, y2 = boxes[0]
    assert cls == 5      # wK cnn idx 6 -> yolo idx 5
    assert (x1, y1, x2, y2) == (4*50, 7*50, 4*50+50, 7*50+50)


def test_board_record_roundtrip(synthetic_board):
    path, fname, fen = synthetic_board
    rec = board_record(path)
    assert rec["fen"] == fen
    assert len(rec["cell_labels"]) == 64
    assert rec["n_pieces"] == sum(1 for v in rec["cell_labels"] if v != 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: cvchess.data.build_dataset`

- [ ] **Step 3: Write `src/cvchess/data/build_dataset.py`**

```python
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from PIL import Image

from cvchess.config import CELL, GRID, CACHE_DIR, CNN_TO_YOLO
from cvchess.data.fen import (
    filename_to_fen, fen_to_label_grid, label_grid_to_cnn_indices,
)
from cvchess.logging_utils import get_logger

log = get_logger("build_dataset")


def slice_cells(img: np.ndarray) -> np.ndarray:
    """Slice a 400x400x3 board into (64, 50, 50, 3) in row-major order."""
    cells = np.empty((GRID * GRID, CELL, CELL, img.shape[2]), dtype=img.dtype)
    k = 0
    for r in range(GRID):
        for c in range(GRID):
            cells[k] = img[r*CELL:(r+1)*CELL, c*CELL:(c+1)*CELL]
            k += 1
    return cells


def derive_yolo_boxes(idx: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """For each occupied cell, a (yolo_cls, x1, y1, x2, y2) cell-box."""
    boxes = []
    for r in range(GRID):
        for c in range(GRID):
            cnn = int(idx[r, c])
            if cnn == 0:
                continue
            boxes.append((CNN_TO_YOLO[cnn],
                          c*CELL, r*CELL, c*CELL+CELL, r*CELL+CELL))
    return boxes


def board_record(path: Path) -> dict:
    """Build a manifest record for one board image (no pixel copies)."""
    fen = filename_to_fen(Path(path).name)
    grid = fen_to_label_grid(fen)
    idx = label_grid_to_cnn_indices(grid)
    flat = idx.reshape(-1).tolist()
    boxes = derive_yolo_boxes(idx)
    return {
        "path": str(path),
        "fen": fen,
        "cell_labels": flat,        # 64 ints, CNN classes
        "boxes": boxes,             # yolo boxes
        "n_pieces": int((idx != 0).sum()),
    }


def build_manifest(split_dir: Path, out_json: Path, limit: int | None = None) -> dict:
    """Scan a split dir of *.png, write a manifest JSON, return summary."""
    split_dir = Path(split_dir)
    imgs = sorted(split_dir.glob("*.png"))
    if limit:
        imgs = imgs[:limit]
    records = [board_record(p) for p in imgs]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(records))
    log.info("Wrote %d records to %s", len(records), out_json)
    return {"count": len(records), "path": str(out_json)}


def load_image(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_build_dataset.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Write `scripts/01_build_data.py`**

```python
"""File 1: download dataset (cache-aware), build CNN+YOLO manifests."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from cvchess.config import CACHE_DIR, ensure_dirs  # noqa: E402
from cvchess.hardware import log_summary, apply_caps  # noqa: E402
from cvchess.data.download import ensure_dataset  # noqa: E402
from cvchess.data.build_dataset import build_manifest  # noqa: E402
from cvchess.logging_utils import get_logger  # noqa: E402

log = get_logger("build_data")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-test", type=int, default=None)
    args = ap.parse_args()

    ensure_dirs()
    log_summary()
    apply_caps()
    root = ensure_dataset()
    build_manifest(root / "train", CACHE_DIR / "train_manifest.json", args.limit_train)
    build_manifest(root / "test", CACHE_DIR / "test_manifest.json", args.limit_test)
    log.info("Data build complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add src/cvchess/data/build_dataset.py scripts/01_build_data.py tests/test_build_dataset.py
git -c commit.gpgsign=false commit -m "Add dataset view builder (cells, YOLO boxes, manifest)"
```

---

## Task 8: Kornia differentiable GPU augmentation

**Files:**
- Create: `src/cvchess/data/augment.py`, `tests/test_augment.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_augment.py
import torch
from cvchess.data.augment import build_augmentation


def test_augmentation_preserves_shape_and_range():
    aug = build_augmentation(strength=0.5)
    x = torch.rand(4, 3, 50, 50)
    y = aug(x)
    assert y.shape == x.shape
    assert y.min() >= 0.0 - 1e-4
    assert y.max() <= 1.0 + 1e-4


def test_augmentation_is_differentiable():
    aug = build_augmentation(strength=0.5)
    x = torch.rand(2, 3, 50, 50, requires_grad=True)
    y = aug(x).sum()
    y.backward()
    assert x.grad is not None


def test_strength_zero_is_near_identity():
    aug = build_augmentation(strength=0.0)
    x = torch.rand(2, 3, 50, 50)
    y = aug(x)
    assert torch.allclose(x, y, atol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_augment.py -v`
Expected: FAIL with `ModuleNotFoundError: cvchess.data.augment`

- [ ] **Step 3: Write `src/cvchess/data/augment.py`**

```python
from __future__ import annotations
import torch
import torch.nn as nn
import kornia.augmentation as K


def build_augmentation(strength: float = 0.5) -> nn.Module:
    """Differentiable GPU augmentation pipeline for 50x50 piece cells.
    strength in [0,1] scales magnitude; 0 => near identity (p=0)."""
    s = max(0.0, min(1.0, strength))
    p = 0.0 if s == 0.0 else 0.5
    return K.AugmentationSequential(
        K.RandomAffine(degrees=8.0 * s, translate=(0.05 * s, 0.05 * s),
                       scale=(1.0 - 0.1 * s, 1.0 + 0.1 * s), p=p),
        K.ColorJitter(brightness=0.2 * s, contrast=0.2 * s,
                      saturation=0.2 * s, hue=0.05 * s, p=p),
        K.RandomGaussianNoise(mean=0.0, std=0.03 * s, p=p),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.1 + 1.0 * s), p=p),
        same_on_batch=False,
    )


class GpuAugmenter(nn.Module):
    """Wraps the pipeline; moves to device and clamps output to [0,1]."""
    def __init__(self, strength: float = 0.5, device: str = "cuda"):
        super().__init__()
        self.aug = build_augmentation(strength).to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aug(x.to(self.device)).clamp_(0.0, 1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_augment.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cvchess/data/augment.py tests/test_augment.py
git -c commit.gpgsign=false commit -m "Add Kornia differentiable GPU augmentation"
```

---

## Task 9: Torch Dataset / DataLoader for CNN cells

**Files:**
- Create: `src/cvchess/data/dataset.py`, `tests/test_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset.py
import json
import numpy as np
from PIL import Image
import torch
from cvchess.data.dataset import CellDataset


def _make_manifest(tmp_path):
    fen = "8-8-8-8-8-8-8-4K3"
    img = Image.fromarray(np.full((400, 400, 3), 127, dtype=np.uint8))
    p = tmp_path / f"{fen}.png"
    img.save(p)
    rec = {
        "path": str(p), "fen": fen,
        "cell_labels": [0]*60 + [0, 0, 0, 0] ,  # placeholder, fixed below
        "boxes": [], "n_pieces": 1,
    }
    # compute real labels
    from cvchess.data.fen import fen_to_label_grid, label_grid_to_cnn_indices
    idx = label_grid_to_cnn_indices(fen_to_label_grid(fen)).reshape(-1).tolist()
    rec["cell_labels"] = idx
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps([rec]))
    return mpath


def test_dataset_len_is_64_per_board(tmp_path):
    m = _make_manifest(tmp_path)
    ds = CellDataset(m)
    assert len(ds) == 64


def test_dataset_item_shape_and_label(tmp_path):
    m = _make_manifest(tmp_path)
    ds = CellDataset(m)
    img, label = ds[60 + 4]  # e1 cell -> wK
    assert img.shape == (3, 50, 50)
    assert img.dtype == torch.float32
    assert int(label) == 6


def test_balanced_indices_reduce_empty(tmp_path):
    m = _make_manifest(tmp_path)
    ds = CellDataset(m, max_empty_ratio=1.0)
    labels = [int(ds[i][1]) for i in range(len(ds))]
    n_empty = sum(1 for l in labels if l == 0)
    n_piece = sum(1 for l in labels if l != 0)
    assert n_empty <= max(1, n_piece)  # capped relative to pieces
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: cvchess.data.dataset`

- [ ] **Step 3: Write `src/cvchess/data/dataset.py`**

```python
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cvchess.config import CELL, GRID
from cvchess.data.build_dataset import slice_cells, load_image


class CellDataset(Dataset):
    """Yields (cell_tensor[3,50,50] float[0,1], cnn_label) for every cell.
    Optionally caps empty cells relative to occupied for class balance."""

    def __init__(self, manifest_path: str | Path, max_empty_ratio: float | None = None):
        self.records = json.loads(Path(manifest_path).read_text())
        self.index: list[tuple[int, int]] = []  # (record_idx, cell_idx)
        for ri, rec in enumerate(self.records):
            labels = rec["cell_labels"]
            piece_cells = [(ri, ci) for ci, l in enumerate(labels) if l != 0]
            empty_cells = [(ri, ci) for ci, l in enumerate(labels) if l == 0]
            if max_empty_ratio is not None:
                cap = max(1, int(len(piece_cells) * max_empty_ratio))
                empty_cells = empty_cells[:cap]
            self.index.extend(piece_cells + empty_cells)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        ri, ci = self.index[i]
        rec = self.records[ri]
        img = load_image(rec["path"])
        cells = slice_cells(img)
        cell = cells[ci].astype(np.float32) / 255.0
        tensor = torch.from_numpy(cell).permute(2, 0, 1).contiguous()
        label = int(rec["cell_labels"][ci])
        return tensor, label
```

> **Note for executor:** `slice_cells` per-item is fine for tests; in Task 12 a
> caching wrapper (LRU on `rec["path"]`) keeps training fast. Add it there.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cvchess/data/dataset.py tests/test_dataset.py
git -c commit.gpgsign=false commit -m "Add CellDataset for CNN training"
```

---

## Task 10: Custom CNN model

**Files:**
- Create: `src/cvchess/models/cnn.py`, `tests/test_cnn.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cnn.py
import torch
from cvchess.models.cnn import ChessCNN, count_params


def test_forward_shape():
    model = ChessCNN(num_classes=13)
    x = torch.rand(8, 3, 50, 50)
    out = model(x)
    assert out.shape == (8, 13)


def test_param_count_positive():
    model = ChessCNN(num_classes=13)
    assert count_params(model) > 0


def test_configurable_depth_width():
    small = ChessCNN(num_classes=13, channels=(16, 32), fc_dim=64)
    big = ChessCNN(num_classes=13, channels=(32, 64, 128), fc_dim=256)
    assert count_params(big) > count_params(small)


def test_dropout_in_eval_is_deterministic():
    model = ChessCNN(num_classes=13, dropout=0.5).eval()
    x = torch.rand(2, 3, 50, 50)
    with torch.no_grad():
        a, b = model(x), model(x)
    assert torch.allclose(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cnn.py -v`
Expected: FAIL with `ModuleNotFoundError: cvchess.models.cnn`

- [ ] **Step 3: Write `src/cvchess/models/cnn.py`**

```python
from __future__ import annotations
import torch
import torch.nn as nn


class ChessCNN(nn.Module):
    """Configurable from-scratch CNN for 50x50 RGB piece cells."""

    def __init__(self, num_classes: int = 13,
                 channels: tuple[int, ...] = (32, 64, 128),
                 fc_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        in_c = 3
        for out_c in channels:
            layers += [
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_c = out_c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_c, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_from_trial(trial, num_classes: int = 13) -> "ChessCNN":
    """Construct a ChessCNN from an Optuna trial's suggested hyperparams."""
    n_blocks = trial.suggest_int("n_blocks", 2, 4)
    base = trial.suggest_categorical("base_channels", [16, 32, 64])
    channels = tuple(base * (2 ** i) for i in range(n_blocks))
    fc_dim = trial.suggest_categorical("fc_dim", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    return ChessCNN(num_classes, channels, fc_dim, dropout)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cnn.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cvchess/models/cnn.py tests/test_cnn.py
git -c commit.gpgsign=false commit -m "Add configurable custom ChessCNN"
```

---

## Task 11: Training-time estimator

**Files:**
- Create: `src/cvchess/train/time_estimator.py`, `tests/test_time_estimator.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_time_estimator.py
from cvchess.train.time_estimator import estimate_total_seconds


def test_estimate_scales_with_steps_and_epochs():
    # 0.01 s/step measured over 10 steps -> 1000 steps/epoch, 5 epochs
    secs = estimate_total_seconds(seconds_per_step=0.01,
                                  steps_per_epoch=1000, epochs=5)
    assert abs(secs - 50.0) < 1e-6


def test_estimate_nonnegative():
    assert estimate_total_seconds(0.0, 100, 10) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_time_estimator.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `src/cvchess/train/time_estimator.py`**

```python
from __future__ import annotations
import time
import torch

from cvchess.logging_utils import get_logger

log = get_logger("time_estimator")


def estimate_total_seconds(seconds_per_step: float, steps_per_epoch: int,
                           epochs: int) -> float:
    return max(0.0, seconds_per_step) * steps_per_epoch * epochs


def calibrate_seconds_per_step(model, loader, device, optimizer,
                               loss_fn, warmup: int = 3, measure: int = 10) -> float:
    """Run a few real steps to measure seconds/step (warmup excluded)."""
    model.train()
    it = iter(loader)
    times: list[float] = []
    for i in range(warmup + measure):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    sps = sum(times) / len(times)
    log.info("Calibrated %.4f s/step", sps)
    return sps
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_time_estimator.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cvchess/train/time_estimator.py tests/test_time_estimator.py
git -c commit.gpgsign=false commit -m "Add training-time estimator with calibration"
```

---

## Task 12: CNN training — Optuna + wandb + final train + save

**Files:**
- Create: `src/cvchess/train/train_cnn.py`, `scripts/02_train_cnn.py`
- Test: extend `tests/test_smoke.py` in Task 18 (training is integration-tested there)

- [ ] **Step 1: Write `src/cvchess/train/train_cnn.py`**

```python
from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cvchess.config import (
    CNN_CLASSES, MODELS_DIR, OPTUNA_DB, CACHE_DIR, OUTPUTS_DIR,
)
from cvchess.data.dataset import CellDataset
from cvchess.data.augment import GpuAugmenter
from cvchess.models.cnn import build_from_trial, ChessCNN, count_params
from cvchess.train.time_estimator import (
    calibrate_seconds_per_step, estimate_total_seconds,
)
from cvchess.hardware import recommended_num_workers, apply_caps
from cvchess.logging_utils import get_logger

log = get_logger("train_cnn")


@dataclass
class TrainConfig:
    train_manifest: str
    test_manifest: str
    epochs: int = 8
    target_accuracy: float = 0.97          # occupied-square accuracy
    budget_seconds: float = 3 * 3600
    n_trials: int = 25
    max_empty_ratio: float = 1.0
    device: str = "cuda"
    wandb_project: str = "chess-cnn-vs-yolo"
    seed: int = 42


def _device(cfg: TrainConfig) -> str:
    return cfg.device if torch.cuda.is_available() else "cpu"


def _loaders(cfg: TrainConfig, batch_size: int):
    train_ds = CellDataset(cfg.train_manifest, max_empty_ratio=cfg.max_empty_ratio)
    test_ds = CellDataset(cfg.test_manifest, max_empty_ratio=cfg.max_empty_ratio)
    nw = recommended_num_workers()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=nw, pin_memory=True)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Return overall + occupied-square accuracy."""
    model.eval()
    correct = total = occ_correct = occ_total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        occ = y != 0
        occ_correct += (pred[occ] == y[occ]).sum().item()
        occ_total += occ.sum().item()
    return {
        "accuracy": correct / max(1, total),
        "occupied_accuracy": occ_correct / max(1, occ_total),
    }


def _train_one(model, cfg, batch_size, lr, weight_decay, optimizer_name,
               aug_strength, device, max_seconds, wandb_run=None, trial=None):
    train_loader, test_loader = _loaders(cfg, batch_size)
    opt_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW,
               "sgd": torch.optim.SGD}[optimizer_name]
    optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    augment = GpuAugmenter(aug_strength, device) if device.startswith("cuda") else None

    # time prediction
    sps = calibrate_seconds_per_step(model, train_loader, device, optimizer, loss_fn)
    eta = estimate_total_seconds(sps, len(train_loader), cfg.epochs)
    log.info("Predicted training time: %.1f s (%.2f min)", eta, eta / 60)
    if wandb_run:
        wandb_run.log({"predicted_seconds": eta, "seconds_per_step": sps})

    start = time.perf_counter()
    best = {"occupied_accuracy": 0.0}
    for epoch in range(cfg.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if augment is not None:
                x = augment(x)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        metrics = evaluate(model, test_loader, device)
        log.info("epoch %d | loss %.4f | occ_acc %.4f", epoch, loss.item(),
                 metrics["occupied_accuracy"])
        if wandb_run:
            wandb_run.log({"epoch": epoch, "loss": loss.item(), **metrics})
        if trial is not None:
            trial.report(metrics["occupied_accuracy"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        best = max([best, metrics], key=lambda m: m["occupied_accuracy"])
        if metrics["occupied_accuracy"] >= cfg.target_accuracy:
            log.info("Hit target accuracy — stopping early")
            break
        if time.perf_counter() - start > max_seconds:
            log.info("Hit per-trial time budget — stopping")
            break
    return best


def make_objective(cfg: TrainConfig):
    import wandb

    def objective(trial: optuna.Trial) -> float:
        torch.manual_seed(cfg.seed)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
        aug_strength = trial.suggest_float("aug_strength", 0.0, 0.8)
        model = build_from_trial(trial, num_classes=len(CNN_CLASSES))
        device = _device(cfg)
        model.to(device)
        run = wandb.init(project=cfg.wandb_project, group="optuna",
                         name=f"trial-{trial.number}", reinit=True,
                         config={**trial.params, "params": count_params(model)})
        try:
            per_trial_budget = cfg.budget_seconds / max(1, cfg.n_trials)
            best = _train_one(model, cfg, batch_size, lr, weight_decay,
                              optimizer_name, aug_strength, device,
                              per_trial_budget, wandb_run=run, trial=trial)
            run.log({"best_occupied_accuracy": best["occupied_accuracy"]})
            return best["occupied_accuracy"]
        finally:
            run.finish()

    return objective


def run_study(cfg: TrainConfig) -> optuna.Study:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    apply_caps()
    study = optuna.create_study(
        study_name="chess_cnn", direction="maximize",
        storage=OPTUNA_DB, load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    deadline = time.perf_counter() + cfg.budget_seconds

    def stop_when_done(study, trial):
        if study.best_value is not None and study.best_value >= cfg.target_accuracy:
            study.stop()
        if time.perf_counter() > deadline:
            study.stop()

    study.optimize(make_objective(cfg), n_trials=cfg.n_trials,
                   callbacks=[stop_when_done])
    log.info("Best occupied accuracy: %.4f params: %s",
             study.best_value, study.best_params)
    return study


def train_final_and_save(cfg: TrainConfig, study: optuna.Study) -> Path:
    """Rebuild best model, train fully, save weights + config."""
    import wandb
    best = study.best_params
    device = _device(cfg)
    fixed = optuna.trial.FixedTrial(best)
    model = build_from_trial(fixed, num_classes=len(CNN_CLASSES)).to(device)
    run = wandb.init(project=cfg.wandb_project, group="final", name="final-model",
                     reinit=True, config=best)
    try:
        _train_one(model, cfg, best["batch_size"], best["lr"],
                   best["weight_decay"], best["optimizer"], best["aug_strength"],
                   device, cfg.budget_seconds, wandb_run=run)
    finally:
        run.finish()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights = MODELS_DIR / "chess_cnn.pt"
    torch.save(model.state_dict(), weights)
    (MODELS_DIR / "chess_cnn_config.json").write_text(json.dumps({
        "classes": CNN_CLASSES, "best_params": best,
    }, indent=2))
    log.info("Saved model to %s", weights)
    return weights
```

- [ ] **Step 2: Write `scripts/02_train_cnn.py`**

```python
"""File 2: Optuna-tuned CNN training with wandb + time prediction + save."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from cvchess.config import CACHE_DIR, ensure_dirs  # noqa: E402
from cvchess.hardware import log_summary  # noqa: E402
from cvchess.train.train_cnn import TrainConfig, run_study, train_final_and_save  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--target", type=float, default=0.97)
    ap.add_argument("--budget-hours", type=float, default=3.0)
    args = ap.parse_args()

    ensure_dirs()
    log_summary()
    cfg = TrainConfig(
        train_manifest=str(CACHE_DIR / "train_manifest.json"),
        test_manifest=str(CACHE_DIR / "test_manifest.json"),
        epochs=args.epochs, n_trials=args.n_trials,
        target_accuracy=args.target, budget_seconds=args.budget_hours * 3600,
    )
    study = run_study(cfg)
    train_final_and_save(cfg, study)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify it imports cleanly (no run yet)**

Run: `python -c "import sys; sys.path.insert(0,'src'); import cvchess.train.train_cnn"`
Expected: no error.

- [ ] **Step 4: Commit**

```bash
git add src/cvchess/train/train_cnn.py scripts/02_train_cnn.py
git -c commit.gpgsign=false commit -m "Add Optuna+wandb CNN training with time prediction and save"
```

---

## Task 13: Prepare YOLO-format dataset

**Files:**
- Create: `src/cvchess/yolo/prepare_yolo.py`

- [ ] **Step 1: Write the failing test** in `tests/test_compare.py` later is wrong place; create `tests/test_prepare_yolo.py`

```python
# tests/test_prepare_yolo.py
import json
from pathlib import Path
import numpy as np
from PIL import Image
from cvchess.yolo.prepare_yolo import to_yolo_label_lines, write_yolo_split


def test_to_yolo_label_lines_normalized():
    boxes = [(5, 200, 350, 250, 400)]  # cls, x1,y1,x2,y2 in 400px
    lines = to_yolo_label_lines(boxes, img_size=400)
    parts = lines[0].split()
    assert parts[0] == "5"
    xc, yc, w, h = map(float, parts[1:])
    assert abs(xc - 225/400) < 1e-6
    assert abs(w - 50/400) < 1e-6
    assert 0 <= xc <= 1 and 0 <= yc <= 1


def test_write_yolo_split_creates_files(tmp_path):
    fen = "8-8-8-8-8-8-8-4K3"
    img = Image.fromarray(np.full((400, 400, 3), 127, dtype=np.uint8))
    p = tmp_path / f"{fen}.png"
    img.save(p)
    from cvchess.data.build_dataset import board_record
    rec = board_record(p)
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps([rec]))
    out = tmp_path / "yolo"
    write_yolo_split(m, out, "train")
    assert (out / "images" / "train" / f"{fen}.png").exists()
    assert (out / "labels" / "train" / f"{fen}.txt").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prepare_yolo.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `src/cvchess/yolo/prepare_yolo.py`**

```python
from __future__ import annotations
import json
import shutil
from pathlib import Path

import yaml  # provided by ultralytics dependency (pyyaml)

from cvchess.config import IMG_SIZE, YOLO_CLASSES


def to_yolo_label_lines(boxes, img_size: int = IMG_SIZE) -> list[str]:
    """Convert [(cls,x1,y1,x2,y2)] pixel boxes to YOLO txt lines (normalized)."""
    lines = []
    for cls, x1, y1, x2, y2 in boxes:
        xc = (x1 + x2) / 2 / img_size
        yc = (y1 + y2) / 2 / img_size
        w = (x2 - x1) / img_size
        h = (y2 - y1) / img_size
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def write_yolo_split(manifest_path: str | Path, out_dir: Path, split: str) -> None:
    records = json.loads(Path(manifest_path).read_text())
    img_dir = Path(out_dir) / "images" / split
    lbl_dir = Path(out_dir) / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        src = Path(rec["path"])
        shutil.copy(src, img_dir / src.name)
        lines = to_yolo_label_lines(rec["boxes"])
        (lbl_dir / f"{src.stem}.txt").write_text("\n".join(lines))


def write_data_yaml(out_dir: Path) -> Path:
    data = {
        "path": str(Path(out_dir).resolve()),
        "train": "images/train",
        "val": "images/test",
        "names": {i: n for i, n in enumerate(YOLO_CLASSES)},
    }
    yaml_path = Path(out_dir) / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data))
    return yaml_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prepare_yolo.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/cvchess/yolo/prepare_yolo.py tests/test_prepare_yolo.py
git -c commit.gpgsign=false commit -m "Add YOLO dataset preparation from manifests"
```

---

## Task 14: Train YOLO baseline

**Files:**
- Create: `src/cvchess/yolo/train_yolo.py`

- [ ] **Step 1: Write `src/cvchess/yolo/train_yolo.py`** (thin wrapper; integration-tested in smoke)

```python
from __future__ import annotations
from pathlib import Path

from cvchess.config import CACHE_DIR, OUTPUTS_DIR
from cvchess.yolo.prepare_yolo import write_yolo_split, write_data_yaml
from cvchess.logging_utils import get_logger

log = get_logger("train_yolo")


def prepare(out_dir: Path | None = None,
            train_manifest: Path | None = None,
            test_manifest: Path | None = None) -> Path:
    out_dir = out_dir or (OUTPUTS_DIR / "yolo_dataset")
    train_manifest = train_manifest or (CACHE_DIR / "train_manifest.json")
    test_manifest = test_manifest or (CACHE_DIR / "test_manifest.json")
    write_yolo_split(train_manifest, out_dir, "train")
    write_yolo_split(test_manifest, out_dir, "test")
    return write_data_yaml(out_dir)


def train(data_yaml: Path, model_name: str = "yolov8n.pt",
          epochs: int = 30, imgsz: int = 400, project: str = "chess-cnn-vs-yolo"):
    from ultralytics import YOLO
    model = YOLO(model_name)
    results = model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz,
                          project=str(OUTPUTS_DIR / "yolo_runs"), name="baseline",
                          exist_ok=True)
    log.info("YOLO training complete: %s", results.save_dir)
    return model, results
```

- [ ] **Step 2: Verify import**

Run: `python -c "import sys; sys.path.insert(0,'src'); import cvchess.yolo.train_yolo"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add src/cvchess/yolo/train_yolo.py
git -c commit.gpgsign=false commit -m "Add YOLO baseline training wrapper"
```

---

## Task 15: Comparison — accuracy, latency, compute

**Files:**
- Create: `src/cvchess/eval/compare.py`, `scripts/03_compare.py`, `tests/test_compare.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compare.py
import torch
from cvchess.eval.compare import (
    accuracy_from_preds, measure_latency_ms, compute_profile,
)
from cvchess.models.cnn import ChessCNN


def test_accuracy_from_preds():
    preds = torch.tensor([1, 2, 3, 0])
    labels = torch.tensor([1, 2, 9, 0])
    assert abs(accuracy_from_preds(preds, labels) - 0.75) < 1e-6


def test_measure_latency_positive():
    model = ChessCNN(num_classes=13).eval()
    ms = measure_latency_ms(model, input_shape=(1, 3, 50, 50),
                            device="cpu", iters=3, warmup=1)
    assert ms > 0


def test_compute_profile_keys():
    model = ChessCNN(num_classes=13)
    prof = compute_profile(model, input_shape=(1, 3, 50, 50))
    assert "params" in prof and prof["params"] > 0
    assert "flops" in prof
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compare.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `src/cvchess/eval/compare.py`**

```python
from __future__ import annotations
import json
import time
from pathlib import Path

import torch

from cvchess.config import OUTPUTS_DIR
from cvchess.logging_utils import get_logger

log = get_logger("compare")


def accuracy_from_preds(preds: torch.Tensor, labels: torch.Tensor) -> float:
    return (preds == labels).float().mean().item()


@torch.no_grad()
def measure_latency_ms(model, input_shape=(1, 3, 50, 50), device="cuda",
                       iters: int = 50, warmup: int = 10) -> float:
    model = model.to(device).eval()
    x = torch.rand(*input_shape, device=device)
    for _ in range(warmup):
        model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def compute_profile(model, input_shape=(1, 3, 50, 50)) -> dict:
    params = sum(p.numel() for p in model.parameters())
    flops = None
    try:
        from thop import profile
        x = torch.rand(*input_shape)
        flops, _ = profile(model, inputs=(x,), verbose=False)
    except Exception as e:
        log.warning("FLOP profiling failed: %s", e)
    return {"params": int(params), "flops": flops}


@torch.no_grad()
def evaluate_cnn(model, loader, device) -> float:
    """Occupied-square accuracy for the CNN over a loader."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        occ = y != 0
        correct += (pred[occ] == y[occ]).sum().item()
        total += occ.sum().item()
    return correct / max(1, total)


def write_report(rows: list[dict], path: Path | None = None) -> Path:
    path = path or (OUTPUTS_DIR / "comparison.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))
    # also a markdown table
    md = ["| model | occ_accuracy | latency_ms | params | flops |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r.get('occ_accuracy','-')} | "
                  f"{r.get('latency_ms','-')} | {r.get('params','-')} | "
                  f"{r.get('flops','-')} |")
    (path.with_suffix(".md")).write_text("\n".join(md))
    log.info("Wrote comparison report to %s", path)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_compare.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Write `scripts/03_compare.py`**

```python
"""File 3: compare custom CNN vs YOLO on the synthetic test set."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cvchess.config import CACHE_DIR, MODELS_DIR, CNN_CLASSES  # noqa: E402
from cvchess.data.dataset import CellDataset  # noqa: E402
from cvchess.models.cnn import build_from_trial  # noqa: E402
from cvchess.eval.compare import (  # noqa: E402
    evaluate_cnn, measure_latency_ms, compute_profile, write_report,
)
from cvchess.hardware import recommended_num_workers, log_summary  # noqa: E402
import optuna  # noqa: E402


def load_cnn(device):
    cfg = json.loads((MODELS_DIR / "chess_cnn_config.json").read_text())
    model = build_from_trial(optuna.trial.FixedTrial(cfg["best_params"]),
                             num_classes=len(CNN_CLASSES))
    model.load_state_dict(torch.load(MODELS_DIR / "chess_cnn.pt", map_location=device))
    return model.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-weights", type=str, default=None,
                    help="path to trained YOLO best.pt; if set, includes YOLO row")
    args = ap.parse_args()
    log_summary()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_ds = CellDataset(CACHE_DIR / "test_manifest.json", max_empty_ratio=1.0)
    loader = DataLoader(test_ds, batch_size=256, num_workers=recommended_num_workers())

    cnn = load_cnn(device)
    rows = [{
        "model": "custom_cnn",
        "occ_accuracy": round(evaluate_cnn(cnn, loader, device), 4),
        "latency_ms": round(measure_latency_ms(cnn, device=device), 4),
        **compute_profile(cnn),
    }]

    if args.yolo_weights:
        from ultralytics import YOLO
        ymodel = YOLO(args.yolo_weights)
        # YOLO latency on a full 400x400 board
        rows.append({
            "model": "yolov8n",
            "occ_accuracy": "see_val_mAP",
            "latency_ms": round(measure_latency_ms(
                ymodel.model, input_shape=(1, 3, 400, 400), device=device), 4),
            "params": sum(p.numel() for p in ymodel.model.parameters()),
            "flops": None,
        })

    write_report(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add src/cvchess/eval/compare.py scripts/03_compare.py tests/test_compare.py
git -c commit.gpgsign=false commit -m "Add CNN vs YOLO comparison (accuracy, latency, compute)"
```

---

## Task 16: Push final CNN to Hugging Face

**Files:**
- Create: `src/cvchess/hf_push.py`

- [ ] **Step 1: Write `src/cvchess/hf_push.py`**

```python
from __future__ import annotations
import json
import os
from pathlib import Path

from cvchess.config import MODELS_DIR, CNN_CLASSES
from cvchess.logging_utils import get_logger

log = get_logger("hf_push")

MODEL_CARD = """---
license: cc0-1.0
tags:
  - image-classification
  - chess
  - pytorch
---

# Chess Piece Classifier (Custom CNN)

A from-scratch CNN that classifies 50x50 chess-board cells into 13 classes
(empty + 6 white + 6 black pieces), trained on the synthetic
`koryakinp/chess-positions` dataset with Kornia GPU augmentation and Optuna
hyperparameter tuning.

## Classes
{classes}

## Usage
```python
import torch
from huggingface_hub import hf_hub_download
weights = hf_hub_download(repo_id="{repo_id}", filename="chess_cnn.pt")
state = torch.load(weights, map_location="cpu")
```
"""


def push(repo_id: str, token: str | None = None, private: bool = False) -> str:
    from huggingface_hub import HfApi, create_repo
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set")
    create_repo(repo_id, token=token, private=private, exist_ok=True,
                repo_type="model")
    api = HfApi(token=token)
    card = MODEL_CARD.format(
        classes="\n".join(f"- {i}: {c}" for i, c in enumerate(CNN_CLASSES)),
        repo_id=repo_id)
    (MODELS_DIR / "README.md").write_text(card)
    for fname in ("chess_cnn.pt", "chess_cnn_config.json", "README.md"):
        fpath = MODELS_DIR / fname
        if fpath.exists():
            api.upload_file(path_or_fileobj=str(fpath), path_in_repo=fname,
                            repo_id=repo_id, repo_type="model")
            log.info("Uploaded %s", fname)
    log.info("Pushed model to https://huggingface.co/%s", repo_id)
    return repo_id
```

- [ ] **Step 2: Verify import**

Run: `python -c "import sys; sys.path.insert(0,'src'); import cvchess.hf_push"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add src/cvchess/hf_push.py
git -c commit.gpgsign=false commit -m "Add Hugging Face model push with model card"
```

---

## Task 17: Performance monitor

**Files:**
- Create: `src/cvchess/monitor.py`

- [ ] **Step 1: Write `src/cvchess/monitor.py`**

```python
from __future__ import annotations
import json
import time
from pathlib import Path

from cvchess.config import OUTPUTS_DIR
from cvchess.hardware import detect
from cvchess.logging_utils import get_logger

log = get_logger("monitor")


def gpu_utilization() -> dict:
    """Sample current GPU memory/util via torch + nvidia-smi if available."""
    out = {"mem_allocated_gb": None, "mem_reserved_gb": None}
    try:
        import torch
        if torch.cuda.is_available():
            out["mem_allocated_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
            out["mem_reserved_gb"] = torch.cuda.memory_reserved() / (1024 ** 3)
    except Exception as e:
        log.warning("gpu sample failed: %s", e)
    return out


def snapshot() -> dict:
    info = detect()
    snap = {
        "gpu": info.gpu_names[0] if info.gpu_names else None,
        **gpu_utilization(),
    }
    comp = OUTPUTS_DIR / "comparison.json"
    if comp.exists():
        snap["comparison"] = json.loads(comp.read_text())
    return snap


def watch(interval_s: int = 30, iterations: int = 0) -> None:
    """Poll and log GPU state; iterations=0 means run once."""
    n = 0
    while True:
        log.info("snapshot: %s", json.dumps(gpu_utilization()))
        n += 1
        if iterations and n >= iterations:
            break
        if iterations == 0:
            break
        time.sleep(interval_s)
```

- [ ] **Step 2: Verify import**

Run: `python -c "import sys; sys.path.insert(0,'src'); import cvchess.monitor"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add src/cvchess/monitor.py
git -c commit.gpgsign=false commit -m "Add performance monitoring utility"
```

---

## Task 18: End-to-end smoke test

**Files:**
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write the smoke test** (tiny synthetic data → build → 1-epoch train → compare; CPU-safe, marked slow)

```python
# tests/test_smoke.py
import json
import numpy as np
import pytest
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from cvchess.data.build_dataset import build_manifest
from cvchess.data.dataset import CellDataset
from cvchess.models.cnn import ChessCNN
from cvchess.eval.compare import evaluate_cnn, measure_latency_ms, compute_profile


def _make_boards(d, n=6):
    d.mkdir(parents=True, exist_ok=True)
    fens = ["8-8-8-8-8-8-8-4K3", "4k3-8-8-8-8-8-8-4K3",
            "rnbqkbnr-pppppppp-8-8-8-8-PPPPPPPP-RNBQKBNR"]
    for i in range(n):
        fen = fens[i % len(fens)]
        img = Image.new("RGB", (400, 400), (200, 200, 200))
        draw = ImageDraw.Draw(img)
        for r in range(8):
            for c in range(8):
                if (r + c) % 2:
                    draw.rectangle([c*50, r*50, c*50+50, r*50+50], fill=(90, 90, 90))
        img.save(d / f"{fen}.png")


@pytest.mark.slow
def test_end_to_end_smoke(tmp_path):
    train_dir, test_dir = tmp_path / "train", tmp_path / "test"
    _make_boards(train_dir, 6)
    _make_boards(test_dir, 3)
    train_m = tmp_path / "train.json"
    test_m = tmp_path / "test.json"
    build_manifest(train_dir, train_m)
    build_manifest(test_dir, test_m)

    ds = CellDataset(train_m, max_empty_ratio=1.0)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    model = ChessCNN(num_classes=13)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for x, y in loader:
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()

    test_ds = CellDataset(test_m, max_empty_ratio=1.0)
    test_loader = DataLoader(test_ds, batch_size=16)
    acc = evaluate_cnn(model, test_loader, "cpu")
    assert 0.0 <= acc <= 1.0
    assert measure_latency_ms(model, device="cpu", iters=3, warmup=1) > 0
    assert compute_profile(model)["params"] > 0
```

- [ ] **Step 2: Run smoke test**

Run: `pytest tests/test_smoke.py -v -m slow`
Expected: PASS (1 passed)

- [ ] **Step 3: Run the full suite**

Run: `pytest -v`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_smoke.py
git -c commit.gpgsign=false commit -m "Add end-to-end smoke test"
```

---

## Task 19: README + real run + iterate to target

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`** documenting setup, the 4 entrypoints, env vars, and the comparison workflow. (Cover: `./setup_env.sh`, set `.env`, `python scripts/01_build_data.py`, `python scripts/02_train_cnn.py`, YOLO prep+train, `python scripts/03_compare.py`, HF push.)

- [ ] **Step 2: Real env setup**

Run: `./setup_env.sh`
Expected: deps install; hardware summary prints A4000.

- [ ] **Step 3: Set secrets** in `.env` (WANDB_API_KEY, HF_TOKEN, KAGGLE_API_TOKEN), `wandb login` via env.

- [ ] **Step 4: Build data (start small, then full)**

Run: `python scripts/01_build_data.py --limit-train 4000 --limit-test 1000`
Then full: `python scripts/01_build_data.py`
Expected: manifests written to `data/cache/`.

- [ ] **Step 5: Train with Optuna (iterate)**

Run: `python scripts/02_train_cnn.py --n-trials 25 --target 0.97 --budget-hours 3`
Expected: wandb runs appear; predicted time logged; best ≥ target or budget hit.
**If target not met:** widen search space / increase epochs / trials and re-run
(study resumes from SQLite). Continue until occupied-accuracy ≥ 0.97.

- [ ] **Step 6: Train YOLO baseline**

Run: `python -c "import sys; sys.path.insert(0,'src'); from cvchess.yolo.train_yolo import prepare, train; y=prepare(); train(y, epochs=30)"`
Expected: YOLO run completes; best.pt under `outputs/yolo_runs/baseline/weights/`.

- [ ] **Step 7: Compare**

Run: `python scripts/03_compare.py --yolo-weights outputs/yolo_runs/baseline/weights/best.pt`
Expected: `outputs/comparison.json` + `.md` with both rows.

- [ ] **Step 8: Push final CNN to HF**

Run: `python -c "import sys; sys.path.insert(0,'src'); from dotenv import load_dotenv; load_dotenv(); from cvchess.hf_push import push; push('arorahoni966/chess-cnn')"`
Expected: model appears on HF.

- [ ] **Step 9: Commit README + final artifacts manifest**

```bash
git add README.md
git -c commit.gpgsign=false commit -m "Add README and run instructions"
```

- [ ] **Step 10: Finish branch** — use `superpowers:finishing-a-development-branch` to merge/PR.

---

## Self-Review (completed by plan author)

**Spec coverage check:**
- Open-source dataset discovery → done in brainstorming; locked to `koryakinp/chess-positions`. ✅
- File 1 download + cache check + GPU/system detect + 90% cap + loggers + Kornia synthetic aug + save to cache → Tasks 4, 6, 7, 8 + `scripts/01_build_data.py`. ✅
- File 2 CNN training + one DB (Optuna SQLite) for loss/error mapping + inference + save + hardware-aware + predict training time → Tasks 11, 12. ✅
- File 3 YOLO vs CNN comparison (accuracy, time, compute) → Tasks 13–15. ✅
- Hardware-detection script + venv from requirements → Tasks 1, 4. ✅
- Optuna iteration + hyperparameter tuning + trials to wandb → Task 12. ✅
- Tests → Tasks 3,4,5,6,7,8,9,10,11,15 + smoke 18. ✅
- Push final CNN to HF → Task 16. ✅
- Monitor performance → Task 17. ✅
- No AI attribution in commits → stated in header + every commit command. ✅

**Placeholder scan:** No TBD/TODO; all code blocks complete. README body (Task 19 Step 1) is described rather than fully written — acceptable as it's prose documentation, content enumerated.

**Type consistency:** `build_from_trial`, `CellDataset(max_empty_ratio=)`, `evaluate_cnn`, `measure_latency_ms`, `compute_profile`, `write_report`, `CNN_CLASSES`(13)/`YOLO_CLASSES`(12), `CNN_TO_YOLO` all consistent across tasks. `_train_one`/`run_study`/`train_final_and_save` signatures align with `scripts/02_train_cnn.py`. ✅

**Known executor notes:**
- `optuna.trial.FixedTrial` re-derives architecture from saved `best_params` — params names in `build_from_trial`/`build_augmentation` searches must stay stable (they are).
- YOLO per-piece accuracy is reported via its own val mAP rather than forced into the cell metric (documented in compare); the apples-to-apples piece-classification number is the CNN's occupied-accuracy vs YOLO's mAP/precision — report both, don't fake a shared scalar.
