# Chess Piece Recognition — Custom CNN vs YOLO
## Comprehensive Technical Report (ML · Data · Models · Hyperparameter Tuning · Results)

**Date:** 2026-06-04
**Authors' hardware:** NVIDIA RTX A4000 (16 GB GDDR6), 112 vCPU, 503 GB RAM, CUDA 12.4, driver 550
**Software:** Python 3.11, PyTorch 2.4.1+cu124, Kornia 0.7.3, Optuna 4.0, Ultralytics 8.3, Weights & Biases 0.27, kagglehub/kaggle, thop
**Artifacts:**
- Code: `github.com/Honi05/cvproject` (branch `chess-cv-pipeline`)
- Custom CNN: https://huggingface.co/honi05/chess-piece-cnn
- YOLO models: https://huggingface.co/honi05/chess-piece-yolo
- Derived dataset: https://huggingface.co/datasets/honi05/chess-positions-cv
- Experiment tracking: W&B project `chess-cnn-vs-yolo` (user `arorahoni966`)

---

# 1. Objective & experimental design

**Question.** For recognizing chess pieces on a (synthetic) board image, how does a
**small custom CNN classifier** compare to **YOLO object detectors** of various
sizes — in **accuracy**, **inference latency**, and **compute/footprint**?

**Design.** A single shared dataset is processed into two views so both model
families see identical underlying data:

| View | Consumer | Unit | Label space |
|---|---|---|---|
| 50×50 cell crops | Custom CNN | one board cell | 13 classes (empty + 12 pieces) |
| Cell bounding boxes | YOLO detectors | one full board image | 12 piece classes |

To compare fairly across paradigms, we define **occupied-cell accuracy**: a
per-piece top-1 metric computed identically for both families. For YOLO, each
predicted box is mapped to its 8×8 grid cell and the highest-confidence class per
occupied cell is scored against ground truth — putting detection on the same scale
as the classifier.

**Engineering methodology.** The whole system was built **test-driven** (38 unit +
smoke tests) across 19 planned tasks, each implemented by a fresh agent and passed
through spec-compliance + code-quality review. Two latent bugs were caught in a
pre-run integration review and fixed (see §8).

---

# 2. DATA

## 2.1 Source dataset
[`koryakinp/chess-positions`](https://www.kaggle.com/datasets/koryakinp/chess-positions)
(Kaggle, **CC0**). 100,000 procedurally-rendered chess boards (80k train / 20k
test), each a **400×400 JPEG**. The board fills the frame as a perfect 8×8 grid, so
every cell is exactly **50×50 px**. There are **no annotation files** — the label is
the **FEN position encoded in the filename**, with rank separators `/` replaced by
`-` (e.g. `1B1B1K2-3p1N2-6k1-R7-5P2-4q3-7R-1B6.jpeg`).

**Download.** Handled by `cvchess.data.download` (cache-aware: skips re-download if
`train/` and `test/` already hold images). Kaggle's newer `KGAT_` API token is
written to `~/.kaggle/access_token`; the full 4.0 GB archive pulled and unzipped in
**162 s**. Full set: 80,000 train + 20,000 test images.

## 2.2 Label derivation (FEN → grids → two views)
`cvchess.data.fen` + `cvchess.data.build_dataset`:

1. **Filename → FEN** (`filename_to_fen`): strip path/extension, keep the dashed board.
2. **FEN → 8×8 char grid** (`fen_to_label_grid`): digits expand to empty cells;
   validates 8 ranks × 8 files; row 0 = rank 8 (top), col 0 = file a (left).
3. **Char grid → 13-class index grid** (`label_grid_to_cnn_indices`): map
   `{P:1,N:2,B:3,R:4,Q:5,K:6, p:7,n:8,b:9,r:10,q:11,k:12}`, empty = 0.
4. **CNN view** — `slice_cells` cuts the 400×400 board into `(64, 50, 50, 3)`
   row-major; each cell inherits its grid label.
5. **YOLO view** — `derive_yolo_boxes` emits, for each *occupied* cell, a box
   `(yolo_cls, x1, y1, x2, y2)` = the exact 50×50 cell rectangle, with
   `yolo_cls = cnn_idx − 1` (12-class, empty has no box). Boxes are written in
   normalized YOLO `cls xc yc w h` format by `prepare_yolo`.

Each board becomes a JSON manifest record `{path, fen, cell_labels[64], boxes,
n_pieces}`. Because both views derive from the *same* cells, the comparison is
internally consistent.

## 2.3 Working set & class distribution
For the time-boxed run we built manifests for **20,000 train / 4,000 test boards**.

| | Value |
|---|---|
| Train boards | 20,000 |
| Train cells (64×boards) | 1,280,000 |
| Piece cells | 214,631 |
| Empty cells | 1,065,369 (**83.2%**) |
| Avg pieces / board | 10.73 |

**Per-class cell counts (train):**

| Class | Count | % | | Class | Count | % |
|---|---|---|---|---|---|---|
| empty | 1,065,369 | 83.23 | | wK / bK | 20,000 / 20,000 | 1.56 / 1.56 |
| wP / bP | 18,503 / 18,589 | 1.45 / 1.45 | | wN / bN | 19,579 / 19,536 | 1.53 / 1.53 |
| wB / bB | 19,927 / 19,617 | 1.56 / 1.53 | | wR / bR | 19,565 / 19,608 | 1.53 / 1.53 |
| wQ / bQ | 9,848 / 9,859 | 0.77 / 0.77 | | | | |

Notable: **kings appear exactly once per board** (20,000 each — always present);
**queens are rarest** (~0.77%, frequently captured); **empty dominates at 83%**.

## 2.4 Class balancing
With 83% empties, a naïve classifier could score 83% by always predicting "empty".
`CellDataset(max_empty_ratio=1.0)` caps empties to ≈ the number of piece cells per
board, yielding **≈429,000 balanced training items** (~50/50 empty vs piece).
Accuracy is then reported as **occupied-cell accuracy** (piece cells only), which is
immune to the empty-class prior.

## 2.5 Performance engineering — memmap crop cache
The naïve `Dataset.__getitem__` would re-decode a full 400×400 JPEG and re-slice all
64 cells *for every cell returned* — 64× redundant work that dominates wall-clock on
100k images. We precompute a **`numpy` memmap crop cache** (`build_crop_cache`):
`(N,64,50,50,3) uint8` cells + `(N,64) int64` labels, written once in manifest
order. Training then reads a 50×50 crop directly from the memmap with **zero JPEG
decode**. Build was ~40 s; cache sizes **9.0 GB train / 1.8 GB test**.

## 2.6 GPU augmentation (Kornia, differentiable)
`cvchess.data.augment` builds a `kornia.augmentation.AugmentationSequential`
(RandomAffine, ColorJitter, RandomGaussianNoise, RandomGaussianBlur), all magnitudes
scaled by a single `strength∈[0,1]` hyperparameter and applied **on-GPU, batched,
differentiable**, with output clamped to [0,1]. `strength=0` returns a true identity
(kornia's `p=0` is unreliable). Augmentation strength is itself an Optuna-tuned
hyperparameter for the CNN.

---

# 3. MODELS

## 3.1 Custom CNN (`cvchess.models.cnn.ChessCNN`)
A compact, fully-configurable from-scratch classifier. Architecture is parameterized
by `(channels, fc_dim, dropout)`; the **Optuna-selected best** configuration is:

```
Input  3 × 50 × 50
[Conv 3→16, 3×3 p1 → BN → ReLU → MaxPool 2]   → 16 × 25 × 25
[Conv 16→32, 3×3 p1 → BN → ReLU → MaxPool 2]  → 32 × 12 × 12
[Conv 32→64, 3×3 p1 → BN → ReLU → MaxPool 2]  → 64 × 6 × 6
AdaptiveAvgPool2d(1)                          → 64
Flatten → Dropout(0.314) → Linear 64→64 → ReLU → Dropout(0.314) → Linear 64→13
```

**28,813 parameters · 0.0069 GFLOPs/cell · 125 KB on disk.** Batch-norm gives
deterministic eval; global average pooling keeps the head tiny.

## 3.2 YOLO variants (Ultralytics YOLOv8)
| Variant | Origin | Width scale | Params (12-cls) | Notes |
|---|---|---|---|---|
| **yolo_pico** | custom YAML, **from scratch** | 0.0625 (¼ of nano) | 184,300 | sub-nano stress test |
| **yolov8n** | `yolov8n.pt`, **pretrained** | 0.25 | 2,692,548 | standard nano |
| **yolov8s** | `yolov8s.pt`, **pretrained** | 0.50 | 9,843,604 | "big" model |

The pico variant was produced by cloning Ultralytics' `yolov8.yaml` and overriding
the `n` scale to `[0.33, 0.0625, 256]` (`configs/yolov8n.yaml`), giving a detector
whose parameter count is in the CNN's order of magnitude. All YOLOs use a 12-class
detect head (no "empty" — empty = absence of a box).

---

# 4. TRAINING & HYPERPARAMETER-TUNING STRATEGY

## 4.1 Custom CNN — Optuna study
- **Framework:** Optuna 4.0, **SQLite RDB storage** (`sqlite:///outputs/optuna.db`,
  `load_if_exists=True` so studies resume), direction = **maximize occupied-accuracy**.
- **Pruner:** `MedianPruner(n_warmup_steps=1)` — under-performing trials are pruned
  via `trial.report()` at each epoch.
- **Per-trial logging:** a dedicated **W&B run per trial** records the sampled
  params, per-epoch loss/accuracy, predicted vs actual time, and the pruning signal.
- **Search space:**

  | Hyperparameter | Range / choices | Type |
  |---|---|---|
  | `batch_size` | {128, 256, 512} | categorical |
  | `lr` | 1e-4 … 5e-3 | log-uniform |
  | `weight_decay` | 1e-6 … 1e-3 | log-uniform |
  | `optimizer` | {adam, adamw, sgd} | categorical |
  | `aug_strength` | 0.0 … 0.8 | uniform |
  | `n_blocks` | 2 … 4 | int |
  | `base_channels` | {16, 32, 64} | categorical |
  | `fc_dim` | {64, 128, 256} | categorical |
  | `dropout` | 0.0 … 0.5 | uniform |

- **Loss / optim:** `CrossEntropyLoss`; optimizer per-trial; GPU augmentation applied
  in the train loop.
- **Stopping strategy (dual gate):** a trial stops early once it reaches the target
  accuracy *or* exhausts its per-trial slice of the global time budget; the **study**
  stops when the best value ≥ target **or** the wall-clock budget elapses
  (`stop_when_done` callback). This satisfies the "target-OR-budget" criterion.
- **Time prediction:** before each trial, `calibrate_seconds_per_step` runs a few
  real steps (warmup excluded) and `estimate_total_seconds` extrapolates the ETA,
  logged up front (measured **0.0283 s/step → ~38 s/epoch → ~142 s/6-epoch trial**).
- **Final model:** `train_final_and_save` rebuilds the best trial via Optuna
  `FixedTrial(best_params)`, trains it, and saves `chess_cnn.pt` + a config JSON.

### Two-phase tuning narrative
We first ran with `target=0.995`; **trial 0 hit 99.76% in one epoch**, instantly
satisfying the target and stopping the study — *too easy* to drive exploration. We
**raised the target to 0.999** and resumed the same SQLite study to force genuine
search. Trial results:

| Trial | base_ch | n_blocks | fc | optimizer | lr | aug | occ-acc | takeaway |
|---|---|---|---|---|---|---|---|---|
| 0 | 32 | 4 | 256 | adamw | 7.1e-4 | 0.004 | 0.9976 | large net, light aug — strong |
| 1 | 64 | 3 | 256 | **sgd** | 1.8e-3 | **0.72** | **0.8636** | SGD + heavy aug — **bad** |
| 2 ⭐ | **16** | 3 | **64** | adamw | 2.3e-3 | 0.10 | **0.9995** | **smallest net wins** |

**Tuning insights.** (a) **AdamW ≫ SGD** for this short-horizon, small-net regime.
(b) **Heavy augmentation hurts** — clean, noise-free rendered pieces don't benefit
from aggressive affine/color/noise jitter; light aug (~0.10) is optimal. (c) **Model
capacity is not the bottleneck** — the *smallest* searched architecture
(16/32/64 channels, 64-d head) achieved the best accuracy, so we ship a 28.8k-param
network. The study auto-stopped at the 0.999 gate after finding trial 2.

## 4.2 YOLO — training strategy
- **Transfer learning:** yolov8n and yolov8s start from **pretrained COCO weights**;
  yolo_pico trains **from scratch** (no pretrained sub-nano weights exist).
- **Config:** `data.yaml` (val → test split), **imgsz 416**, **12 epochs**,
  **fraction 0.25** (5,000 train imgs) to bound time, batch 64, 16 dataloader
  workers, Ultralytics' `optimizer=auto` (selected **AdamW, lr≈6.25e-4**), default
  YOLO mosaic/HSV augmentation, `plots=False`.
- **Per-model timing:** pico 260 s, yolov8n 537 s, yolov8s 734 s (train only).
- **Evaluation:** post-training `model.val()` for mAP, plus our **occupied-cell**
  remap on 1,000 test boards for the apples-to-apples per-piece number.
- **Why not tune YOLO further?** Pretrained n/s saturated at **mAP@50-95 = 0.995** in
  12 epochs — there is no accuracy headroom to tune toward; additional epochs/data
  would not change conclusions. The pico's failure is architectural (sub-nano from
  scratch), not a tuning artifact.

---

# 5. RESULTS

## 5.1 Master comparison
Accuracy = occupied-cell top-1 (per-piece, 1,000 test boards). Latency = ms per full
400×400 board on the A4000 (CNN runs its 64 cells as one batched call). Compute =
params, GFLOPs/board, on-disk size.

| Model | Occ-acc | mAP@50-95 | Params | GFLOPs/board | Latency ms/board | Disk MB | Train s |
|---|---|---|---|---|---|---|---|
| **custom_cnn** ⭐ | **0.9996** | — (classifier) | **28,813** | 0.442 | **0.514** | **0.125** | ~38 |
| yolo_pico (scratch) | 0.2073 | 0.255 | 184,300 | 0.236 | 5.491 | 0.567 | 260 |
| yolov8n (pretrained) | **1.0000** | 0.995 | 2,692,548 | 1.469 | 4.346 | 5.594 | 537 |
| yolov8s (pretrained) | **1.0000** | 0.995 | 9,843,604 | 4.985 | 6.067 | 19.918 | 734 |

## 5.2 Efficiency ratios (vs custom CNN)
| Metric | yolov8n / CNN | yolov8s / CNN |
|---|---|---|
| Parameters | **93×** | **342×** |
| Latency / board | **8.5×** | **11.8×** |
| Disk size | **45×** | **159×** |
| Occupied accuracy | +0.04 pp | +0.04 pp |

## 5.3 Reading the results
- **Pretrained YOLOs are perfect** (100% per-piece; mAP@50-95 0.995). The boxes are
  exactly grid-aligned, so localization is trivial and the only work is
  classification — which a COCO-pretrained backbone does flawlessly.
- **The custom CNN is within 0.04 pp** (99.96%) at **~1% of the cost** on every
  footprint axis.
- **The from-scratch pico collapses (20.7%)** and is even **slower than nano** —
  reducing width doesn't reduce the detection head's fixed layer/anchor overhead, and
  a sub-nano detector can't learn from 5k images in 12 epochs. A same-size *classifier*
  learns the task in *one epoch*. **Lesson: shrink the paradigm, not just the width.**

## 5.4 Trade-off & recommendation
The CNN's efficiency relies on the **8×8 grid assumption** (here free from the
synthetic geometry). YOLO needs **no board-cropping step** and would generalize to
real photographs far better. Therefore:
- **Grid-croppable / embedded:** ship the **custom CNN** (99.96% at 125 KB, 0.5 ms).
- **Robust real-world detection:** **yolov8n pretrained** (smallest *effective*
  end-to-end model; yolov8s adds cost without accuracy).
- **Avoid** training sub-nano YOLO from scratch on small data.

---

# 6. Reproducibility
```bash
./setup_env.sh                              # venv (system torch) + deps + hw check
# .env: WANDB_API_KEY, HF_TOKEN, KAGGLE_API_TOKEN
python scripts/01_build_data.py --limit-train 20000 --limit-test 4000
python scripts/02_train_cnn.py --epochs 8 --n-trials 20 --target 0.999 --budget-hours 1
python scripts/train_all_yolo.py            # pico / n / s
python scripts/full_compare.py              # comparison.json/md + W&B table
python scripts/push_artifacts.py            # push all models + dataset to HF
```
Determinism: fixed seeds for CNN trials; Optuna study persisted to SQLite (resumable);
all runs logged to W&B; exact dependency pins in `requirements.txt`.

# 7. Limitations & future work
- **Synthetic, grid-aligned** data — the CNN's per-cell framing assumes a pre-cropped
  board; real photos need a board-detection/perspective-correction front end.
- **Domain gap** — rendered pieces are clean; a real-photo test set (e.g. ChessReD2K)
  would stress generalization and likely favor YOLO.
- **YOLO at 25% data / 12 epochs** — already saturated, but full-data training and a
  proper YOLO HPO sweep are natural next steps.
- **Board → FEN reconstruction** and **legality checking** are out of scope here.

# 8. Engineering quality notes
- **TDD throughout:** 38 unit + smoke tests; per-task spec + code-quality review.
- **Bug caught pre-run #1 (Critical):** the system-RAM cap used `RLIMIT_AS`, which
  limits *virtual* address space — incompatible with CUDA's huge virtual reservations
  (illusory protection + latent hard crash). Replaced with a **monitored soft RAM
  guard** + the real 90% GPU-memory fraction cap.
- **Bug caught pre-run #2 (Important):** 64× redundant JPEG decode per cell →
  replaced with the **memmap crop cache** (§2.5).
- **Bug caught at eval (Minor):** `thop` FLOP profiling crashed on device mismatch
  and leaked hooks → profile a deep copy on the model's own device.
- **Tooling:** Kaggle `KGAT_` token support, wandb `wandb_v1_` key support (pin bump).
- **Commit hygiene:** zero AI attribution in any commit message or history.
```
```
