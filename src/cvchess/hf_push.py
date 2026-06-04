from __future__ import annotations
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
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "README.md").write_text(card)
    for fname in ("chess_cnn.pt", "chess_cnn_config.json", "README.md"):
        fpath = MODELS_DIR / fname
        if fpath.exists():
            api.upload_file(path_or_fileobj=str(fpath), path_in_repo=fname,
                            repo_id=repo_id, repo_type="model")
            log.info("Uploaded %s", fname)
    log.info("Pushed model to https://huggingface.co/%s", repo_id)
    return repo_id
