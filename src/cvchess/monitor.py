from __future__ import annotations
import json
import time
from pathlib import Path

from cvchess.config import OUTPUTS_DIR
from cvchess.hardware import detect
from cvchess.logging_utils import get_logger

log = get_logger("monitor")


def gpu_utilization() -> dict:
    """Sample current GPU memory via torch if available."""
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
