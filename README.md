# Chess Piece Recognition — Custom CNN vs YOLO

A reproducible computer-vision pipeline that recognizes chess pieces from board
images and **compares a custom CNN classifier against YOLO object detectors** on
**accuracy, inference time, and compute cost**, using a single shared synthetic
dataset.

- **Dataset:** [`koryakinp/chess-positions`](https://www.kaggle.com/datasets/koryakinp/chess-positions)
  — 100k rendered 400×400 boards, FEN in filename, clean 8×8 grid (50×50 cells).
- **Two derived views from one source:** 50×50 **cell crops** for the CNN
  (13 classes: empty + 6 white + 6 black) and **grid-derived bounding boxes**
  for YOLO (12 piece classes).
- **Custom CNN:** small from-scratch conv net, Optuna-tuned, wandb-tracked.
- **YOLO:** `yolov8n` baseline, a width-scaled **pico** (~CNN-comparable size),
  and `yolov8s` (larger), trained on the same derived boxes.
- **Trained CNN** is published to the Hugging Face Hub.

## Hardware (this run)

NVIDIA RTX A4000 (16 GB) · 112 CPU · 503 GB RAM · CUDA 12.4 · PyTorch 2.4 (cu124).
GPU memory is capped to 90%; system RAM is monitored via a soft guard (we
deliberately do **not** cap virtual address space — that breaks CUDA).

## Setup

```bash
./setup_env.sh                 # venv (--system-site-packages) + deps + hardware check
cp .env.example .env           # then fill in your tokens:
#   WANDB_API_KEY, HF_TOKEN, KAGGLE_API_TOKEN
```

Kaggle's newer `KGAT_` token goes in `KAGGLE_API_TOKEN` (also written to
`~/.kaggle/access_token` for the kaggle client).

## Pipeline

```bash
# 1. Download (cache-aware) + build cell/box manifests + memmap crop cache
python scripts/01_build_data.py --limit-train 20000 --limit-test 4000

# 2. Optuna-tuned CNN training (wandb per trial, SQLite study, time prediction)
python scripts/02_train_cnn.py --epochs 8 --n-trials 20 --target 0.999 --budget-hours 1

# 3a. Train YOLO variants (pico / n / s)
python scripts/train_all_yolo.py

# 3b. Consolidated comparison (per-piece accuracy + deployment stats + wandb)
python scripts/full_compare.py

# Hardware-only check + live GPU monitor
python scripts/detect_hardware.py
python -c "import sys;sys.path.insert(0,'src');from cvchess.monitor import watch; watch(iterations=0)"
```

## Architecture

```
src/cvchess/
  config.py            paths, class maps, constants
  hardware.py          GPU/CPU/RAM detect, 90% GPU cap, soft RAM guard
  logging_utils.py     central logger
  data/
    fen.py             FEN <-> labels, filename decode
    download.py        cache-aware Kaggle download
    build_dataset.py   slice cells, derive YOLO boxes, memmap crop cache
    augment.py         Kornia differentiable GPU augmentation
    dataset.py         CellDataset (memmap-backed)
  models/cnn.py        configurable ChessCNN (Optuna-parameterized)
  train/
    time_estimator.py  calibration -> ETA
    train_cnn.py       Optuna + wandb + final train + save
  yolo/
    prepare_yolo.py    manifests -> ultralytics dataset
    train_yolo.py      YOLO baseline wrapper
  eval/compare.py      accuracy + latency + params/FLOPs
  hf_push.py           upload CNN + model card to HF
  monitor.py           live GPU/perf monitor
scripts/               detect_hardware, 01_build_data, 02_train_cnn,
                       03_compare, train_all_yolo, full_compare
```

## Results

See [`docs/REPORT.md`](docs/REPORT.md) for the full comparison. Summary table is
written to `outputs/comparison.md` / `outputs/comparison.json` by
`full_compare.py`. Trained CNN: **https://huggingface.co/honi05/chess-piece-cnn**.

## Tests

```bash
pytest -q          # unit + smoke tests
```

## License

Code: see `LICENSE`. Dataset: CC0 (`koryakinp/chess-positions`). Research use.
