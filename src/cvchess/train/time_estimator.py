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
