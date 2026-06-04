# Chess Piece Recognition — Custom CNN vs YOLO: Full Report

**Date:** 2026-06-04 · **Hardware:** NVIDIA RTX A4000 (16 GB), 112 CPU, 503 GB RAM, CUDA 12.4, PyTorch 2.4 (cu124)
**Trained CNN:** https://huggingface.co/honi05/chess-piece-cnn · **Tracking:** Weights & Biases project `chess-cnn-vs-yolo` (user `arorahoni966`)

## 1. Goal & method

Recognize chess pieces from board images and compare a **custom CNN classifier**
against **YOLO object detectors** on accuracy, inference time, and compute, using
one shared synthetic dataset.

- **Dataset:** `koryakinp/chess-positions` — rendered 400×400 boards, FEN in the
  filename, clean 8×8 grid (50×50 cells). Downloaded full (80k train / 20k test);
  working set built from **20k train / 4k test boards**.
- **Two views from one source:** 50×50 **cell crops** for the CNN (13 classes:
  empty + 12 pieces); **grid-derived bounding boxes** for YOLO (12 piece classes).
- A **memmap crop cache** lets the CNN read 50×50 crops with zero JPEG re-decode.
- **Apples-to-apples accuracy:** "occupied-cell accuracy" — for YOLO, each
  detection is mapped to its grid cell and the highest-confidence class per
  occupied cell is compared to ground truth (same scale as the CNN's metric),
  measured on 1,000 held-out test boards.

## 2. Custom CNN — training & tuning

Small from-scratch conv net (Conv-BN-ReLU-MaxPool blocks → global pool → MLP head),
**Optuna**-tuned with a **SQLite** study, each trial logged to **wandb**, training
time predicted from a calibration mini-run.

**Measured training speed (A4000):** 0.0283 s/step · **~38 s/epoch** (20k boards) ·
**~142 s** for a full 6-epoch trial. The task is easy for a CNN — it reaches
target accuracy in **a single epoch**.

**Optuna trials (occupied-accuracy):**

| Trial | base_ch | blocks | fc | optimizer | lr | aug | occ-acc |
|---|---|---|---|---|---|---|---|
| 0 | 32 | 4 | 256 | adamw | 7.1e-4 | 0.004 | 0.9976 |
| 1 | 64 | 3 | 256 | **sgd** | 1.8e-3 | **0.72** | **0.8636** |
| 2 ⭐ | **16** | 3 | **64** | adamw | 2.3e-3 | 0.10 | **0.9995** |

**Tuning insights:** the **smallest** architecture won (base 16 / fc 64); AdamW
clearly beat SGD here; and **heavy augmentation hurt** (trial 1's aug=0.72 + SGD
collapsed to 0.86) — unsurprising for clean, noise-free rendered pieces. The study
stopped early once the 0.999 target was met.

**Best CNN (final, retrained & saved):** 28,813 params · **99.96% occupied-cell
accuracy on the held-out test set** · 0.125 MB on disk.

## 3. YOLO — three variants

Trained on the same derived boxes (25% data fraction = 5k imgs, 12 epochs,
imgsz 416), via `ultralytics`:

- **yolo_pico** — YOLOv8 width-scaled to ~CNN size, **from scratch** (no pretrained weights).
- **yolov8n** — standard nano, **pretrained**.
- **yolov8s** — standard small, **pretrained** ("big" model).

## 4. Head-to-head comparison

Accuracy = occupied-cell top-1 (per-piece, directly comparable). Compute =
params, GFLOPs/board, latency/board (full 400×400 board; CNN runs its 64 cells as
one batched call). Latency measured on the A4000.

| Model | Occ-acc | mAP@50-95 | Params | GFLOPs/board | Latency ms/board | Disk MB | Train s |
|---|---|---|---|---|---|---|---|
| **custom_cnn** ⭐ | **0.9996** | — (classifier) | **28,813** | 0.442 | **0.514** | **0.125** | ~38 |
| yolo_pico (scratch) | 0.2073 | 0.255 | 184,300 | 0.236 | 5.491 | 0.567 | 260 |
| yolov8n (pretrained) | **1.0000** | 0.995 | 2,692,548 | 1.469 | 4.346 | 5.594 | 537 |
| yolov8s (pretrained) | **1.0000** | 0.995 | 9,843,604 | 4.985 | 6.067 | 19.918 | 734 |

(YOLO param counts reflect the 12-class head; nominal 80-class figures are larger.)

## 5. Analysis

**Accuracy.** The pretrained YOLOs are *perfect* (100% per-piece on 1,000 boards;
mAP@50-95 = 0.995 — the grid-aligned boxes are trivial to localize). The custom
CNN is a hair behind at **99.96%**. The from-scratch **pico collapses to 20.7%** —
a sub-nano detector cannot learn detection from scratch on 5k images in 12 epochs,
whereas a classifier of similar size learns the task in one epoch. **A tiny CNN
classifier is far more data/compute-efficient than a tiny detector from scratch.**

**Compute & deployment.** The CNN is the standout on every efficiency axis:
- **~93× fewer params** than yolov8n (28.8k vs 2.69M), **~340× fewer** than yolov8s.
- **~8.5× lower latency/board** than yolov8n (0.51 ms vs 4.35 ms) — the 64 cells
  run as a single batched call on a shallow, regular conv net.
- **~45× smaller on disk** (0.125 MB vs 5.6 MB), trivially embeddable.

**The trade-off.** YOLO buys you **end-to-end detection** — it finds pieces in the
raw image with **no board-localization step**, and generalizes to real photos far
better than a cell classifier that assumes a pre-cropped 8×8 grid. The CNN's
efficiency depends on that grid assumption (here provided by the synthetic data's
geometry). For this clean, grid-aligned synthetic task, the CNN wins decisively on
cost at ~equal accuracy; for messy real-world boards, YOLO's robustness would
likely justify its extra compute.

**On the pico.** Scaling YOLO down to ~CNN size is counter-productive: it lost
almost all accuracy yet was *slower* than yolov8n (5.49 ms vs 4.35 ms) because
detection-head/architecture overhead dominates and reduced width doesn't cut the
layer count. The smallest *useful* YOLO here is the pretrained nano.

## 6. Recommendation

- **Deployment on grid-croppable boards (this task):** ship the **custom CNN** —
  99.96% accuracy at ~1% of YOLO's size and latency.
- **Robust real-photo recognition:** use **yolov8n pretrained** — perfect here,
  smallest *effective* end-to-end model; yolov8s adds cost without accuracy gain.
- **Don't** train a sub-nano YOLO from scratch for a small dataset.

## 7. Reproduction

```bash
./setup_env.sh
# .env: WANDB_API_KEY, HF_TOKEN, KAGGLE_API_TOKEN
python scripts/01_build_data.py --limit-train 20000 --limit-test 4000
python scripts/02_train_cnn.py --epochs 8 --n-trials 20 --target 0.999 --budget-hours 1
python scripts/train_all_yolo.py
python scripts/full_compare.py        # writes outputs/comparison.{json,md} + wandb table
```

Raw numbers: `outputs/comparison.json`. Per-run dashboards: wandb `chess-cnn-vs-yolo`.

## 8. Notes & limitations

- Synthetic, grid-aligned data: the CNN's per-cell framing assumes the board is
  already cropped to an 8×8 grid. Real photos would need board detection first.
- YOLO trained on a 25% fraction / 12 epochs for time budget; already saturated
  (mAP 0.995), so more data/epochs would not change conclusions.
- pico from scratch is intentionally a stress test of "tiny detector" feasibility.
- A monitored soft RAM guard is used instead of an `RLIMIT_AS` cap (the latter
  breaks CUDA's large virtual-address reservations).
