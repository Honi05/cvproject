"""Publish all trained models (CNN + 3 YOLO) and the derived dataset to the
Hugging Face Hub. Idempotent: re-running updates the repos.

Usage: python scripts/push_artifacts.py
Requires HF_TOKEN in the environment / .env.
"""
import os
import sys
import json
import tarfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dotenv import load_dotenv
load_dotenv()

from cvchess.config import OUTPUTS_DIR, CACHE_DIR, CNN_CLASSES, YOLO_CLASSES
from cvchess.logging_utils import get_logger

log = get_logger("push_artifacts")

HF_USER = "honi05"
CNN_REPO = f"{HF_USER}/chess-piece-cnn"
YOLO_REPO = f"{HF_USER}/chess-piece-yolo"
DATA_REPO = f"{HF_USER}/chess-positions-cv"

YOLO_RUNS = OUTPUTS_DIR / "yolo_runs"
YOLO_VARIANTS = ["yolo_pico", "yolov8n", "yolov8s"]


def _api():
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set")
    return HfApi(token=token), token


# ---------------------------------------------------------------- YOLO models
YOLO_CARD = """---
license: cc0-1.0
tags: [object-detection, chess, yolov8, ultralytics]
---

# Chess Piece Detectors (YOLOv8 — pico / n / s)

Three YOLOv8 detectors trained on cell-box annotations derived from the synthetic
[`koryakinp/chess-positions`](https://www.kaggle.com/datasets/koryakinp/chess-positions)
dataset (12 piece classes). Companion to the custom CNN classifier at
[`honi05/chess-piece-cnn`](https://huggingface.co/honi05/chess-piece-cnn).

| Variant | Origin | Params | mAP@50-95 | Occupied-cell acc | Latency ms/board | Disk MB |
|---|---|---|---|---|---|---|
| `yolo_pico/best.pt` | scaled YAML, from scratch | 184,300 | 0.255 | 0.2073 | 5.49 | 0.57 |
| `yolov8n/best.pt` | pretrained | 2,692,548 | 0.995 | 1.0000 | 4.35 | 5.59 |
| `yolov8s/best.pt` | pretrained | 9,843,604 | 0.995 | 1.0000 | 6.07 | 19.92 |

Classes (0-11): {classes}

## Usage
```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
w = hf_hub_download("{repo}", "yolov8n/best.pt")
model = YOLO(w)
results = model("board.jpg")
```

See the full report in the code repo `github.com/Honi05/cvproject`.
"""


def push_yolo_models():
    from huggingface_hub import create_repo
    api, token = _api()
    create_repo(YOLO_REPO, token=token, repo_type="model", exist_ok=True)
    for v in YOLO_VARIANTS:
        best = YOLO_RUNS / v / "weights" / "best.pt"
        if not best.exists():
            log.warning("missing %s — skipping", best)
            continue
        api.upload_file(path_or_fileobj=str(best), path_in_repo=f"{v}/best.pt",
                        repo_id=YOLO_REPO, repo_type="model")
        log.info("uploaded %s/best.pt", v)
    card = YOLO_CARD.format(classes=", ".join(YOLO_CLASSES), repo=YOLO_REPO)
    (OUTPUTS_DIR / "_yolo_README.md").write_text(card)
    api.upload_file(path_or_fileobj=str(OUTPUTS_DIR / "_yolo_README.md"),
                    path_in_repo="README.md", repo_id=YOLO_REPO, repo_type="model")
    log.info("pushed YOLO models -> https://huggingface.co/%s", YOLO_REPO)


# -------------------------------------------------------------------- dataset
DATA_CARD = """---
license: cc0-1.0
tags: [chess, computer-vision, object-detection, image-classification]
task_categories: [object-detection, image-classification]
---

# Chess Positions — CV-derived annotations (cells + YOLO boxes)

Annotations and a runnable sample **derived** from the synthetic, CC0
[`koryakinp/chess-positions`](https://www.kaggle.com/datasets/koryakinp/chess-positions)
dataset (400x400 rendered boards, FEN in filename, clean 8x8 grid → 50x50 cells).

This repo adds, on top of that source:
- **`train_manifest.json` / `test_manifest.json`** — per-board records:
  `{path, fen, cell_labels[64], boxes, n_pieces}`. `cell_labels` are 13-class CNN
  indices (0=empty, 1-6 white PNBRQK, 7-12 black pnbrqk); `boxes` are
  `(yolo_cls, x1, y1, x2, y2)` cell rectangles (12-class).
- **`yolo_labels.tar.gz`** — Ultralytics YOLO-format label files for all boards
  (`cls xc yc w h`, normalized), plus `data.yaml`.
- **`sample.zip`** — 500 example boards (images + YOLO labels) so the dataset is
  immediately runnable without re-downloading the full source.

CNN classes (0-12): {cnn_classes}
YOLO classes (0-11): {yolo_classes}

## Full images
The full 100k images are on Kaggle (CC0). Pull them and regenerate every artifact
with the project's `scripts/01_build_data.py` (see `github.com/Honi05/cvproject`).

## Trained models
- Custom CNN classifier: [`honi05/chess-piece-cnn`](https://huggingface.co/honi05/chess-piece-cnn)
- YOLO detectors (pico/n/s): [`honi05/chess-piece-yolo`](https://huggingface.co/honi05/chess-piece-yolo)
"""


def _build_dataset_archives(work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    yolo_ds = OUTPUTS_DIR / "yolo_dataset"
    # 1) tar.gz of all YOLO labels + data.yaml
    labels_tar = work / "yolo_labels.tar.gz"
    with tarfile.open(labels_tar, "w:gz") as tf:
        for split in ("train", "test"):
            ldir = yolo_ds / "labels" / split
            if ldir.is_dir():
                tf.add(ldir, arcname=f"labels/{split}")
        if (yolo_ds / "data.yaml").exists():
            tf.add(yolo_ds / "data.yaml", arcname="data.yaml")
    # 2) sample.zip — 500 test boards (image + label)
    sample_zip = work / "sample.zip"
    img_dir = yolo_ds / "images" / "test"
    lbl_dir = yolo_ds / "labels" / "test"
    imgs = sorted(img_dir.glob("*.jpeg"))[:500] + sorted(img_dir.glob("*.png"))[:500]
    imgs = imgs[:500]
    with zipfile.ZipFile(sample_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for im in imgs:
            zf.write(im, arcname=f"images/{im.name}")
            lbl = lbl_dir / f"{im.stem}.txt"
            if lbl.exists():
                zf.write(lbl, arcname=f"labels/{im.stem}.txt")
    return {"labels_tar": labels_tar, "sample_zip": sample_zip, "n_sample": len(imgs)}


def push_dataset():
    from huggingface_hub import create_repo
    api, token = _api()
    create_repo(DATA_REPO, token=token, repo_type="dataset", exist_ok=True)
    work = OUTPUTS_DIR / "_dataset_pkg"
    arch = _build_dataset_archives(work)
    # manifests + data.yaml
    for f in ("train_manifest.json", "test_manifest.json"):
        p = CACHE_DIR / f
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=f,
                            repo_id=DATA_REPO, repo_type="dataset")
            log.info("uploaded %s", f)
    dy = OUTPUTS_DIR / "yolo_dataset" / "data.yaml"
    if dy.exists():
        api.upload_file(path_or_fileobj=str(dy), path_in_repo="data.yaml",
                        repo_id=DATA_REPO, repo_type="dataset")
    for key in ("labels_tar", "sample_zip"):
        p = arch[key]
        api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                        repo_id=DATA_REPO, repo_type="dataset")
        log.info("uploaded %s", p.name)
    # DATA_CARD contains literal { } (JSON schema, YOLO label spec), so use
    # replace() rather than format() to fill the two class placeholders.
    card = (DATA_CARD
            .replace("{cnn_classes}", ", ".join(CNN_CLASSES))
            .replace("{yolo_classes}", ", ".join(YOLO_CLASSES)))
    (work / "README.md").write_text(card)
    api.upload_file(path_or_fileobj=str(work / "README.md"), path_in_repo="README.md",
                    repo_id=DATA_REPO, repo_type="dataset")
    log.info("pushed dataset (%d sample imgs) -> https://huggingface.co/datasets/%s",
             arch["n_sample"], DATA_REPO)


def main():
    log.info("=== Pushing YOLO models ===")
    push_yolo_models()
    log.info("=== Pushing derived dataset ===")
    push_dataset()
    log.info("Done. CNN already at https://huggingface.co/%s", CNN_REPO)


if __name__ == "__main__":
    main()
