# Chess Piece Recognition: Custom CNN vs YOLO — Design Spec

**Date:** 2026-06-04
**Status:** Approved (design) — pending user review of this written spec
**Repo:** `cvproject` (branch `chess-cv-pipeline`)

## 1. Goal

Build a reproducible computer-vision pipeline that recognizes chess pieces from
board images, and **compare a custom CNN classifier against a YOLO object
detector** on the **same synthetic data**, measuring **accuracy, inference time,
and compute cost**.

The custom CNN is the primary deliverable: it is tuned with Optuna, tracked in
Weights & Biases, and the final model is pushed to the Hugging Face Hub.

## 2. Locked decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Shared dataset | Kaggle `koryakinp/chess-positions` (100k synthetic, CC0) |
| Comparison framing | Per-piece: CNN on crops vs YOLO end-to-end |
| Stopping criterion | Per-class top-1 ≥ 97% **OR** ~2–3 h A4000 tuning budget, whichever first |
| Use case | Research / personal (any license OK) |
| Augmentation | Kornia differentiable, GPU-accelerated |
| Self-attribution in git | **Never** — no AI attribution in commit messages or history |

## 3. Target hardware (detected)

- GPU: NVIDIA RTX A4000, 16 GB VRAM, CUDA 12.4, driver 550
- CPU: 112 cores · RAM: 503 GB
- Python 3.11.10, `torch 2.4.1+cu124`, CUDA available
- **Memory policy:** cap GPU and system RAM usage to **90%**.

## 4. Dataset details

`koryakinp/chess-positions`:
- 100k rendered boards, 400×400 px, 80k train / 20k test.
- Label = FEN encoded in filename (board ranks separated by `-` instead of `/`).
- Board fills the frame → clean 8×8 grid, each cell **50×50 px**.

### Two derived views from one source
- **CNN view** — slice each board into 64 cells, label each from FEN.
  **13 classes**: `empty`, `{w,b}×{P,N,B,R,Q,K}`. ~6.4M candidate cells;
  subsample to a class-balanced cap (configurable, e.g. ≤ N per class).
- **YOLO view** — for each *occupied* cell, derive a bounding box from grid
  geometry. Canonical box = the 50×50 cell. Optional tightening via flat-
  background foreground thresholding (off by default). YOLO uses the **12 piece
  classes** (empty is "no box"). Boxes are derived from the same cells the CNN
  crops → comparison is internally consistent.

## 5. Architecture & components

User-facing entrypoints map to the requested "files"; reusable logic lives in a
small package `src/cvchess/`.

```
cvproject/
  requirements.txt
  setup_env.sh                 # venv (--system-site-packages to keep cu124 torch) + deps
  .env.example                 # token names only; real .env is git-ignored
  scripts/
    detect_hardware.py         # -> hardware.summary()/caps
    01_build_data.py           # File 1: download + cache check + build cells/boxes + Kornia aug -> cache
    02_train_cnn.py            # File 2: Optuna + wandb + time prediction + train + save
    03_compare.py              # File 3: YOLO vs CNN comparison
  src/cvchess/
    __init__.py
    config.py                  # paths, cache dirs, class maps, constants
    logging_utils.py           # central structured logger
    hardware.py                # detect GPU/CPU/RAM, cap to 90%, summary
    data/
      fen.py                   # FEN parse, square<->label maps, filename decode
      download.py              # kagglehub download + disk/cache check
      build_dataset.py         # slice cells, derive YOLO boxes, write cache manifest
      augment.py               # Kornia differentiable GPU augmentation pipeline
      dataset.py               # torch Dataset/DataLoader for CNN cells
    models/
      cnn.py                   # custom CNN (Optuna-parameterized); pretrained toggle
    train/
      time_estimator.py        # calibration mini-run -> ETA
      train_cnn.py             # Optuna study (SQLite) + wandb per-trial + final train + save
    yolo/
      prepare_yolo.py          # write ultralytics YOLO dataset (images/labels/data.yaml)
      train_yolo.py            # train YOLO baseline
    eval/
      compare.py               # accuracy + latency + params/FLOPs/peak-VRAM
    hf_push.py                 # upload final CNN + model card to HF
    monitor.py                 # live performance monitoring (wandb)
  tests/
    test_fen.py test_hardware.py test_augment.py
    test_dataset.py test_cnn.py test_compare.py test_smoke.py
```

## 6. Data flow

```
download (cache hit?) -> build cells + derived boxes -> Kornia GPU aug
  -> torch DataLoader -> Optuna x wandb training -> best model
  -> save to disk + push to HF -> compare vs YOLO -> monitor
```

## 7. Key behaviors

### Hardware (`hardware.py`)
- Detect via `torch.cuda`, `nvidia-smi` parse, `psutil`.
- `torch.cuda.set_per_process_memory_fraction(0.9)`; soft `RLIMIT_AS ≈ 0.9×RAM`;
  bounded `num_workers`/prefetch. Log a full summary at startup.

### Download (`download.py`)
- Use `KAGGLE_API_TOKEN` (KGAT format). `kagglehub.dataset_download(...)`.
- **Cache check first:** if extracted dataset + manifest already on disk, skip
  re-download. Verify file count / checksum sentinel.

### Augmentation (`augment.py`)
- `kornia.augmentation.AugmentationSequential` on GPU: RandomAffine (small),
  ColorJitter, RandomPerspective, Gaussian blur/noise — differentiable, batched.
- Used both to optionally pre-generate a synthetic cache and live in-loop.

### CNN training (`train_cnn.py`)
- Custom CNN, from scratch by default (rendered pieces are simple).
- **Optuna** study with **SQLite RDB storage** (`sqlite:///optuna.db`) — this is
  the single DB mapping trials → loss/error. MedianPruner for early stop.
- Search space: lr, batch size, optimizer, conv depth/width, dropout, weight
  decay, augmentation strength.
- **Each trial logged to wandb** (params, per-epoch loss/acc, pruning).
- **Training-time prediction:** calibration mini-run measures steps/sec → ETA
  for full run, logged before training starts.
- Stop tuning when per-class top-1 ≥ 97% **or** budget exhausted.
- Save best model (weights + config JSON) to disk.

### YOLO baseline (`train_yolo.py`)
- `ultralytics` YOLO (e.g. `yolov8n`) on the derived-box dataset, wandb-logged.

### Comparison (`compare.py`)
- On the held-out synthetic test set: per-piece classification accuracy
  (CNN on GT cells vs YOLO matched detections), inference latency (ms/image,
  warm), compute (params, FLOPs via thop/ptflops, peak VRAM). Output a table +
  wandb artifact.

### HF push (`hf_push.py`)
- After final CNN training, upload weights + config + auto-generated model card
  to the user's HF account using `HF_TOKEN`.

### Monitoring (`monitor.py`)
- wandb run dashboards for all training; a script to poll/report current best
  metrics and flag regressions.

## 8. Tooling, secrets, env

- `setup_env.sh`: create venv with `--system-site-packages` (preserve working
  cu124 torch), `pip install -r requirements.txt`, then run `detect_hardware.py`.
- `requirements.txt` (pinned): torch (already present), torchvision, kornia,
  kagglehub, optuna, wandb, ultralytics, huggingface_hub, opencv-python, psutil,
  thop/ptflops, numpy, pillow, pytest, python-dotenv, GPUtil.
- Secrets in git-ignored `.env`: `WANDB_API_KEY`, `HF_TOKEN`, `KAGGLE_API_TOKEN`.
  `.env.example` documents names only. **No secrets committed.**

## 9. Testing strategy (pytest, TDD where it fits)

- `test_fen.py` — FEN decode/encode, filename→labels, square↔index maps.
- `test_hardware.py` — detector returns sane positive values; cap math correct.
- `test_augment.py` — Kornia output shape/dtype/value-range preserved.
- `test_dataset.py` — cell slicing geometry; label alignment; box derivation.
- `test_cnn.py` — forward pass shape, param count, deterministic seed.
- `test_compare.py` — accuracy/latency/FLOP metric math on fixtures.
- `test_smoke.py` — tiny (~50 image) end-to-end build→train(1 epoch)→compare.

## 10. Definition of done

1. `setup_env.sh` + `detect_hardware.py` run clean on the A4000 box.
2. `01_build_data.py` downloads (cache-aware), builds both views, caches them.
3. `02_train_cnn.py` runs Optuna×wandb, predicts time, trains, hits the stopping
   criterion, saves the model.
4. `03_compare.py` produces the CNN-vs-YOLO accuracy/time/compute table.
5. All tests pass; smoke test green.
6. Final CNN pushed to HF; wandb dashboards live; monitoring script works.
7. Iteration continues until the CNN meets the 97%-or-budget criterion.

## 11. Out of scope (YAGNI)

- Real-photo capture/board-corner detection at inference (synthetic only here).
- Full board→FEN reconstruction / legality checking.
- Deployment/serving, web UI, mobile capture.
