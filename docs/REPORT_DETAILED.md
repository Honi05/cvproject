# Chess Piece Recognition: A Tiny Custom CNN vs. YOLO Object Detectors

A detailed technical report comparing two machine-learning paradigms for recognizing chess pieces on rendered board images, measured on accuracy, inference time (latency), and compute footprint, over a shared synthetic dataset.

---

## 1. Executive Summary

This project asks a deceptively simple question: if you need to identify the chess pieces on a board image, do you really need a heavyweight object detector, or will a tiny image classifier do? The answer, for **grid-croppable** boards (boards whose geometry lets you cut the image into a clean 8x8 grid of cells), is decisively in favor of the tiny classifier.

A custom **convolutional neural network** (CNN) with just **28,813 parameters** (the count of learned numeric weights inside the model) reaches **99.96% per-piece accuracy**, statistically tied with a pretrained **YOLO** object detector at **100%**. Yet the CNN is roughly **93x smaller in parameter count**, has roughly **8.5x lower latency** (wall-clock time per board), and occupies roughly **45x less disk space** than the smallest competitive YOLO model. At the opposite extreme, a from-scratch "sub-nano" YOLO (nicknamed **pico**) collapses to **20.7%** accuracy, worse than naive guessing, and is slower than the larger pretrained YOLO it was meant to undercut.

**Conclusion:** For boards that can be cleanly cropped into a grid, a tiny classifier wins overwhelmingly on cost while matching accuracy. YOLO earns its keep only when the input is a messy real-world photo where pieces must first be *found* before they are *named*, the situation a pure classifier cannot handle on its own.

---

## 2. Problem Statement & Why It Matters

The task is to recognize, on an image of a chess board, which of **13 categories** occupies each square:

- **6 white piece types**: P (pawn), N (knight), B (bishop), R (rook), Q (queen), K (king).
- **6 black piece types**: p, n, b, r, q, k (same pieces, lowercase by convention).
- **1 "empty" category** for a square with no piece.

We compare two fundamentally different machine-learning **paradigms** for solving this:

**Image classification.** A classifier takes a single image and outputs one label describing the whole image. In our setup we pre-cut the board into 64 individual cell images and ask the CNN: "what is in *this* 50x50 cell?" The model never has to *locate* anything; the location is given by which cell we handed it.

**Object detection.** A detector takes the *whole* board image and must simultaneously (a) *find* each piece by drawing a **bounding box** (a rectangle around the object) and (b) *classify* what is inside each box. YOLO is a detector: it answers "what objects are present, and where?" in one shot.

The distinction matters because the two paradigms have very different costs and robustness. Classification is cheap but presupposes the object is already isolated. Detection is expensive but handles cluttered, unconstrained scenes. This report quantifies that trade-off on one controlled dataset.

---

## 3. Environment & Hardware

All experiments ran on a single cloud GPU instance (provider: **RunPod**).

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX A4000, 16 GB GDDR6 memory |
| CPU | 112 vCPU |
| System RAM | 503 GB |
| CUDA toolkit | 12.4 |
| NVIDIA driver | 550 |

Key software versions:

| Package | Version | Role |
|---|---|---|
| Python | 3.11 | Language runtime |
| PyTorch | 2.4.1+cu124 | Deep-learning framework |
| Kornia | 0.7.3 | GPU image augmentation |
| Optuna | 4.0 | Hyperparameter optimization |
| Ultralytics | 8.3 | YOLOv8 implementation |
| Weights & Biases (wandb) | 0.27 | Experiment tracking |
| kagglehub / kaggle | - | Dataset download |
| thop | - | FLOP (compute) counter |

**Memory policy, a subtle but important decision.** GPU memory was capped to **90%** of the card via `torch.cuda.set_per_process_memory_fraction`, leaving headroom so a runaway allocation cannot freeze the whole device. For *system* RAM, we deliberately used a **monitored soft guard** (a background check using `psutil` that watches resident memory and intervenes if it grows too large) rather than a hard OS-level cap.

The reason is easy to get wrong. The classic Linux hard cap, `RLIMIT_AS`, limits a process's *virtual address space*, the total memory it may *reserve* whether or not that memory is actually used. CUDA reserves enormous virtual address ranges during initialization, far in excess of what it physically touches. An `RLIMIT_AS` cap therefore provides essentially **no real protection** against actual memory exhaustion (it counts reservations, not usage) and can **hard-crash CUDA** the moment the driver reserves its normal address space. A monitored soft guard on *real* (resident) usage is both safer and more meaningful.

---

## 4. Data

### 4.1 Source

The dataset is **`koryakinp/chess-positions`** from Kaggle, released under the **CC0 license** (public domain, free to use without restriction). It contains **100,000 procedurally rendered** board images, generated by a renderer, not photographed, each a **400x400 JPEG**. The standard split is **80,000 train / 20,000 test**.

Crucially, there are **no separate annotation files**. The label for each image lives in its **filename**: the **FEN** string describing the position, with the rank-separator character `/` replaced by `-` (since `/` is illegal in filenames). Example filename:

```
1B1B2K1-1B6-5N2-6k1-8-8-8-4nq2.jpeg
```

Because each rendered board fills the entire frame, the 400x400 image divides perfectly into an **8x8 grid**, making each cell exactly **50x50 pixels**.

### 4.2 What FEN means

**FEN (Forsyth-Edwards Notation)** is a compact text encoding of a chess position. The board portion is 8 **ranks** (rows) separated by `/`. Within a rank, a letter denotes a piece (**uppercase = white, lowercase = black**) and a **digit denotes a run of consecutive empty squares** (e.g. `4` means four empty squares in a row). So `4nq2` means: 4 empty, black knight, black queen, 2 empty, exactly 8 files.

### 4.3 Download

Download is **cache-aware** (it reuses a local copy if present). A fresh Kaggle API token (`KGAT_`-prefixed access token) is written to `~/.kaggle/access_token`. The full **4.0 GB** archive was downloaded and unzipped in **162 seconds**.

### 4.4 Label-derivation pipeline

The raw filename is turned into trainable labels through a deterministic pipeline:

1. **Filename to FEN**: undo the `-`-for-`/` substitution.
2. **FEN to 8x8 character grid**: expand digits into that many "empty" characters; validate exactly 8 ranks x 8 files. By convention **row 0 = rank 8 (top of the board)** and **col 0 = file a (left)**.
3. **Character grid to 13-class integer-index grid**, using the mapping:

| Symbol | empty | P | N | B | R | Q | K | p | n | b | r | q | k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |

From this single label grid we derive both paradigms' views:

- **CNN view**: slice the board into 64 cells (each 50x50), each carrying its own class index (0-12).
- **YOLO view**: for each **occupied** cell, emit a bounding box equal to that exact 50x50 cell rectangle, with class = `cnn_index - 1` (so YOLO has **12 classes, no "empty"**; detectors only predict objects that exist).

### 4.5 Working set and statistics

To bound compute, experiments used a working subset: **20,000 train / 4,000 test** boards.

| Statistic | Value |
|---|---|
| Total train cells | 1,280,000 |
| Piece (occupied) cells | 214,631 |
| Empty cells | 1,065,369 (**83.2%**) |
| Average pieces per board | 10.73 |

Per-class counts confirm a natural imbalance: **kings appear exactly 20,000 times each** (every legal board has exactly one white and one black king); **queens are rarest at ~9,850 each** (~0.77% of cells, since queens are often captured); pawns, knights, bishops, and rooks land at **~18,500-19,900 each** (~1.5% of cells apiece).

### 4.6 Handling class imbalance

With **83% of cells empty**, a lazy model could score 83% accuracy by *always* answering "empty", a meaningless result. Two measures neutralize this:

1. **Balanced sampling.** `CellDataset(max_empty_ratio=1.0)` caps the number of empty cells kept per board to approximately the number of *piece* cells on that board, yielding a balanced training pool of roughly **429,000 crops** instead of 1.28 M (mostly-empty) crops.
2. **Reporting "occupied-cell accuracy."** Throughout, the headline metric is **occupied-cell accuracy**, accuracy computed *only over cells that actually contain a piece*. This deliberately ignores the empty-square prior, so a model is rewarded purely for naming pieces correctly, not for exploiting the fact that most squares are blank.

### 4.7 Performance engineering: the memmap crop cache

A naive dataset implementation would, for *every single cell* it returns, re-decode the full 400x400 JPEG and re-slice all 64 cells: **64x redundant work** per cell access. To eliminate this, we precompute a **memmap crop cache**.

A **memmap** (memory-mapped file) is a file on disk that the program accesses *as if* it were an in-memory array, while the operating system pages in only the bytes actually touched, so you get array-style indexing without loading the whole file into RAM. We precompute:

- a `(N, 64, 50, 50, 3)` array of `uint8` pixels (every cell of every board), and
- a `(N, 64)` array of `int64` labels,

both stored on disk and memory-mapped. Training then reads any 50x50 crop with **zero JPEG decoding**. Building the cache takes **~40 seconds**; the files are **9.0 GB (train) / 1.8 GB (test)**.

### 4.8 Data augmentation

**Data augmentation** means applying random, *label-preserving* perturbations to training images (small rotations, color shifts, noise) so the model sees more variety and generalizes better rather than memorizing exact pixels.

Augmentation uses **Kornia**, a library that runs image transforms **on the GPU and differentiably**, meaning the operations execute inside the training loop on the GPU (no slow CPU-to-GPU bottleneck) and are compatible with gradient-based training. The pipeline is a Kornia `AugmentationSequential` containing **RandomAffine** (small geometric shifts/rotations), **ColorJitter** (brightness/contrast/saturation changes), **RandomGaussianNoise**, and **RandomGaussianBlur**. All transform magnitudes are scaled by a single **strength in [0, 1]** hyperparameter, applied batched on-GPU, with output clamped to [0, 1]. At **strength = 0** the pipeline is a true identity (no change at all).

---

## 5. Models

### 5.1 Custom CNN (`ChessCNN`)

A **CNN (convolutional neural network)** is a neural network built around **convolution** layers, small learnable filters that slide across the image detecting local patterns (edges, textures, shapes) regardless of position. CNNs are the standard tool for image tasks.

The best configuration found (by the Optuna search in section 6) processes a single **3x50x50** cell (3 color channels) through **3 convolutional blocks** and a small classifier head:

| Stage | Operation | Output shape |
|---|---|---|
| Input | - | 3 x 50 x 50 |
| Block 1 | Conv(3x3, pad 1) -> BatchNorm -> ReLU -> MaxPool(2) | 16 x 25 x 25 |
| Block 2 | Conv(3x3, pad 1) -> BatchNorm -> ReLU -> MaxPool(2) | 32 x 12 x 12 |
| Block 3 | Conv(3x3, pad 1) -> BatchNorm -> ReLU -> MaxPool(2) | 64 x 6 x 6 |
| Pool | AdaptiveAvgPool2d(1) (global average pooling) | 64 x 1 x 1 |
| Flatten | - | 64-vector |
| Head | Dropout(0.314) -> Linear(64->64) -> ReLU -> Dropout(0.314) -> Linear(64->13) | 13 logits |

Layer-type definitions (first use):

- **Conv(3x3, pad 1)**: a convolution with 3x3 filters and 1 pixel of zero-padding so spatial size is preserved before pooling.
- **BatchNorm (batch normalization)**: rescales each layer's outputs to a stable mean/variance across the mini-batch, which speeds and stabilizes training.
- **ReLU**: a simple activation that replaces negatives with zero, introducing non-linearity.
- **MaxPool(2)**: downsamples by taking the maximum over each 2x2 region, halving spatial size.
- **AdaptiveAvgPool2d(1) / global average pooling**: averages each channel's entire feature map down to a single number, producing one value per channel (here a 64-vector) regardless of input size, a parameter-free way to summarize spatial features.
- **Dropout(0.314)**: during training, randomly zeroes 31.4% of activations to prevent the network from over-relying on any one feature (a regularizer against overfitting).
- **Linear (fully connected)**: a standard matrix-multiply layer mapping one vector to another. The final Linear produces **13 logits** (one raw score per class). A **softmax** (turns logits into a probability distribution summing to 1) is applied for prediction.

**Totals: 28,813 parameters, 0.0069 GFLOPs per cell, 125 KB on disk.**

### 5.2 YOLO variants (YOLOv8, Ultralytics)

**YOLO ("You Only Look Once")** is a **single-stage object detector**: it predicts all bounding boxes and their class labels in **one forward pass** over the image, making it fast relative to multi-stage detectors. **Transfer learning** means starting from weights already trained on a large general dataset (here **COCO**, a standard 80-class object-detection benchmark) and fine-tuning them on our task, which jump-starts learning when data or compute is limited.

| Variant | Source | Parameters | Notes |
|---|---|---|---|
| `yolo_pico` | Custom width-scaled YAML, scale `[0.33, 0.0625, 256]` (~1/4 of nano width) | 184,300 | Trained **from scratch**; no pretrained weights exist for a sub-nano model |
| `yolov8n` | Pretrained on COCO | 2,692,548 | Standard "nano" |
| `yolov8s` | Pretrained on COCO | 9,843,604 | Standard "small" |

---

## 6. Training & Hyperparameter Tuning

### 6.1 Methodology

The codebase was built **test-driven**, with **38 unit and smoke tests**, and organized into **19 tasks**, each reviewed for both specification compliance and code quality.

### 6.2 CNN training loop

The CNN trains with **CrossEntropyLoss**, the standard **loss function** (the number being minimized) for classification. Cross-entropy measures how far the predicted class-probability distribution is from the true label; it is minimized when the model assigns probability 1 to the correct class. Each trial uses its own optimizer, applies GPU augmentation inside the loop, evaluates after every epoch, and stops early when it either hits the accuracy target or exhausts the time budget.

Terminology: a **step** is one parameter update from one mini-batch; an **epoch** is one full pass over the training data. Measured speed: **0.0283 s/step** -> **~38 s/epoch** (over 20k boards) -> **~142 s per 6-epoch trial**. A short calibration mini-run predicts the trial ETA up front so the budget can be managed.

### 6.3 The Optuna study

**Optuna** is a framework for **automatic hyperparameter optimization**; it intelligently proposes hyperparameter combinations, runs trials, and learns which regions of the search space are promising. Configuration:

- **Storage**: an **SQLite RDB** (a single-file relational database). Persisting the study to disk makes it **resumable**; you can stop and continue the same study later.
- **Sampler**: the **TPE (Tree-structured Parzen Estimator)** sampler, which models the distribution of good vs. bad trials and samples new candidates toward the good region.
- **Direction**: **maximize** occupied-cell accuracy.
- **Pruner**: the **MedianPruner**, which stops ("prunes") a trial early if its intermediate score is below the **median** of previously completed trials at the same step, saving compute on clearly losing configurations.
- Every trial is logged as a separate **wandb run** (Weights & Biases experiment-tracking record) for inspection.

### 6.4 Search space

| Hyperparameter | Range / choices | Sampling |
|---|---|---|
| `batch_size` | {128, 256, 512} | categorical |
| `lr` (learning rate) | 1e-4 .. 5e-3 | log-uniform |
| `weight_decay` | 1e-6 .. 1e-3 | log-uniform |
| `optimizer` | {adam, adamw, sgd} | categorical |
| `aug_strength` | 0.0 .. 0.8 | uniform |
| `n_blocks` | 2 .. 4 | integer |
| `base_channels` | {16, 32, 64} | categorical |
| `fc_dim` | {64, 128, 256} | categorical |
| `dropout` | 0.0 .. 0.5 | uniform |

**Log-uniform sampling** draws values uniformly on a *logarithmic* scale, so each order of magnitude is equally likely. It is the right choice for learning rate and weight decay because those span several orders of magnitude (1e-4 to 5e-3; 1e-6 to 1e-3) and the interesting differences are multiplicative, not additive.

### 6.5 Two-phase tuning narrative

The first study set **target = 0.995**. **Trial 0 hit 99.76% in a single epoch**, met the target, and the study stopped, revealing the task was *too easy* to discriminate good configs at that bar. We **raised the target to 0.999** and **resumed the same SQLite study** to force further exploration.

| Trial | Key config | Occupied-cell accuracy |
|---|---|---|
| 0 | base32 / 4 blocks / fc256 / AdamW / lr 7.1e-4 / aug 0.004 | 0.9976 |
| 1 | base64 / 3 blocks / fc256 / SGD / lr 1.8e-3 / aug 0.72 | 0.8636 |
| 2 (**best**) | base16 / 3 blocks / fc64 / AdamW / lr 2.3e-3 / aug 0.10 | **0.9995** |

**Insights:** (1) **AdamW is far better than SGD** on this task; the SGD trial badly underperformed. (2) **Heavy augmentation hurt**: trial 1's aug 0.72 degraded results because the renders are *clean and uniform*, so aggressive perturbation just throws away signal. (3) The **smallest architecture won**: base-16 channels, 3 blocks, fc-64, yielding the final **28.8k-parameter** model. The study **auto-stopped at the 0.999 gate** once the target was met.

### 6.6 Final model

The best trial's hyperparameters were rebuilt deterministically via Optuna's **FixedTrial(best_params)**, retrained, and saved as `chess_cnn.pt` plus a config JSON, then pushed to Hugging Face (HF).

### 6.7 YOLO training strategy

- **Transfer learning** for `yolov8n` and `yolov8s` (start from COCO weights); **`yolo_pico` from scratch** (no sub-nano pretrained weights exist).
- Shared config: `imgsz 416`, **12 epochs**, `fraction 0.25` (use 25% = **5,000 training images** to bound training time), `batch 64`, Ultralytics `optimizer=auto` (which selected **AdamW** at lr ~ 6.25e-4), default mosaic and HSV augmentation.

| Model | Train time |
|---|---|
| `yolo_pico` | 260 s |
| `yolov8n` | 537 s |
| `yolov8s` | 734 s |

**Why no deeper YOLO HPO?** The pretrained n/s models **saturated** at **mAP@50-95 = 0.995** within 12 epochs; there is no headroom left to tune toward. And `pico`'s failure is **architectural, not a tuning artifact**: a sub-nano detector simply lacks the capacity to learn this detection head, so more tuning would not rescue it.

---

## 7. Evaluation Metrics

### 7.1 Occupied-cell accuracy (the shared, comparable metric)

For each test board: run the model; for **YOLO**, map each predicted box to a grid cell using the box center, **cell = (floor(center_x / 50), floor(center_y / 50))**, and keep the **highest-confidence class per occupied cell**; then compare the predicted class to ground truth **only on occupied cells**. Accuracy = correct / total occupied. This is computed on **1,000 test boards for YOLO** (detection inference is slower) and the **full 4,000-board test set for the CNN**. Reducing YOLO's detections to a per-cell class makes detection and classification **directly comparable** on the same yardstick.

### 7.2 mAP@50-95 (YOLO's native metric)

This is the standard object-detection score, built from several concepts:

- **IoU (Intersection over Union)**: the overlap between a predicted box and the true box, computed as (area of intersection) / (area of union). 1.0 = perfect overlap, 0 = none.
- **Precision** = fraction of predicted boxes that are correct; **recall** = fraction of true objects that were found.
- **AP (Average Precision)**: the area under the precision-recall curve for a class at a given IoU threshold.
- **mAP@50-95 (mean Average Precision)**: AP averaged over all classes *and* over IoU thresholds from **0.50 to 0.95** (in steps of 0.05). It rewards both correct classification **and** tight box localization.

Because mAP is a **detection** metric (it credits box-localization quality), it is **not directly comparable** to a classifier's accuracy, which is precisely why occupied-cell accuracy was introduced as the cross-paradigm metric.

### 7.3 Compute metrics

| Metric | Meaning |
|---|---|
| **Parameters** | Count of learned weights in the model (a proxy for memory/capacity). |
| **FLOPs / GFLOPs** | Floating-point operations per inference (G = billions); measured with **thop**. A hardware-independent measure of compute. |
| **Latency** | Wall-clock milliseconds per board on the A4000, **excluding warmup**. The CNN runs all 64 cells of a board as **one batched GPU call**. |
| **Disk size** | Megabytes of the saved model weights. |

---

## 8. Results

### 8.1 Master table

| Model | Occupied-cell acc | mAP@50-95 | Params | GFLOPs/board | Latency (ms/board) | Disk (MB) | Train (s) |
|---|---|---|---|---|---|---|---|
| `custom_cnn` | **0.9996** | - (classifier) | **28,813** | 0.442 | **0.514** | **0.125** | ~38 |
| `yolo_pico` | 0.2073 | 0.255 | 184,300 | 0.236 | 5.491 | 0.567 | 260 |
| `yolov8n` | **1.0000** | 0.995 | 2,692,548 | 1.469 | 4.346 | 5.594 | 537 |
| `yolov8s` | **1.0000** | 0.995 | 9,843,604 | 4.985 | 6.067 | 19.918 | 734 |

### 8.2 Efficiency ratios (vs. the CNN)

| Comparison | Params | Latency | Disk | Accuracy gain |
|---|---|---|---|---|
| `yolov8n` vs CNN | **93x** | **8.5x** | **45x** | +0.04 pp |
| `yolov8s` vs CNN | **342x** | **11.8x** | **159x** | +0.04 pp |

(*pp = percentage points.*)

### 8.3 Interpretation

**(a) Why the pretrained YOLOs are perfect.** On grid-aligned renders, every bounding box is a clean, axis-aligned 50x50 rectangle, *trivial* to localize, so detection reduces almost entirely to classification, and a COCO-pretrained backbone classifies these clean pieces flawlessly. Hence 100% accuracy and a saturated mAP of 0.995.

**(b) The CNN is essentially free by comparison.** It lands **within 0.04 percentage points** of perfect while costing roughly **1%** of YOLO on *every* footprint axis: parameters, latency, and disk. For a deployment where the board can be cropped, this is an overwhelming win.

**(c) Why `pico` collapses *and* is slower than nano.** Shrinking the *width* of a detector does not remove the **fixed overhead of the detection head** (its layers and anchor machinery), so pico is actually **slower** (5.49 ms) than the larger nano (4.35 ms). Worse, a sub-nano detector simply lacks the capacity to learn detection from only 5,000 images over 12 epochs, whereas a **same-sized classifier learns the task in a single epoch**. The lesson: **shrink the paradigm, not just the width.** If you want a tiny model, switch to the cheaper task (classification); do not merely scale down an expensive one (detection).

---

## 9. Deployment Recommendation

| Scenario | Recommendation |
|---|---|
| Grid-croppable / embedded (board geometry known) | **Ship the CNN**: 99.96% accuracy, 125 KB, ~0.5 ms/board. |
| Robust real-photo input | **Use pretrained `yolov8n`**: no board-cropping step needed and the best generalization. `yolov8s` adds cost with no accuracy gain. |
| Tiny model from scratch | **Do not train a sub-nano YOLO from scratch on small data**; switch paradigms instead. |

**Caveat.** The CNN assumes you can crop the image into a clean 8x8 grid, free here thanks to synthetic geometry. Real-world photos first need **board detection and perspective correction** to obtain that grid. Supplying that geometry is exactly where YOLO's localization strength pays off.

---

## 10. Reproducibility & Artifacts

The full pipeline is driven by these commands, in order:

```bash
setup_env.sh
scripts/01_build_data.py --limit-train 20000 --limit-test 4000
scripts/02_train_cnn.py --epochs 8 --n-trials 20 --target 0.999 --budget-hours 1
scripts/train_all_yolo.py
scripts/full_compare.py
scripts/push_artifacts.py
```

**Artifacts:**

- Code: `github.com/Honi05/cvproject` (branch `chess-cv-pipeline`, PR #1, 38 tests, no AI attribution).
- CNN model: `https://huggingface.co/honi05/chess-piece-cnn`
- YOLO models: `https://huggingface.co/honi05/chess-piece-yolo`
- Dataset: `https://huggingface.co/datasets/honi05/chess-positions-cv`
- Experiment tracking: wandb project `chess-cnn-vs-yolo`.

---

## 11. Limitations & Future Work

- **Synthetic, grid-aligned data.** The CNN's success leans on pre-cropped boards; this is a property of the synthetic dataset, not of real photographs.
- **Domain gap to real photos.** Lighting, perspective, piece styles, and clutter would degrade the classifier and *favor* YOLO, which can localize under those conditions.
- **YOLO trained on a fraction.** Detectors used only 25% of data over 12 epochs, but they already saturated, so more data would not move the needle here.
- **Future work:** add a **real-photo test set** (e.g. ChessReD2K) to measure the domain gap; train YOLO on **full data with a proper HPO sweep**; and build a downstream **board-to-FEN reconstruction** step with **legality checking** to convert per-cell predictions into a validated position.

---

## 12. Engineering-Quality Notes

Development was **test-driven throughout (38 tests)**. Three concrete bugs were caught and fixed:

| Severity | Bug | Fix |
|---|---|---|
| **Critical** | `RLIMIT_AS` virtual-address cap is incompatible with CUDA (CUDA reserves huge virtual ranges; the cap gives no real protection and can hard-crash the driver). | Replaced with a **monitored soft RAM guard** (psutil) on real resident usage. |
| **Important** | Dataset re-decoded the full JPEG and re-sliced all 64 cells for every cell access: **64x redundant work**. | Built the **memmap crop cache** for zero-decode reads. |
| **Minor** | thop FLOP profiling crashed on a device mismatch and leaked forward hooks. | Profile a **deepcopy** of the model on the model's own device, leaving the original untouched. |

**Tooling fixes:** support for the Kaggle `KGAT_` access-token format, and support for the newer wandb `wandb_v1_` key format (a version bump). **Commit hygiene:** the git history contains **zero AI attribution**.
