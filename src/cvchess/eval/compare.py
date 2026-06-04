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
    md = ["| model | occ_accuracy | latency_ms | params | flops |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r.get('occ_accuracy','-')} | "
                  f"{r.get('latency_ms','-')} | {r.get('params','-')} | "
                  f"{r.get('flops','-')} |")
    (path.with_suffix(".md")).write_text("\n".join(md))
    log.info("Wrote comparison report to %s", path)
    return path
