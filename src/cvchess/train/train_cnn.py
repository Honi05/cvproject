from __future__ import annotations
import json
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna

from cvchess.config import CNN_CLASSES, MODELS_DIR, OPTUNA_DB, OUTPUTS_DIR
from cvchess.data.dataset import CellDataset
from cvchess.data.augment import GpuAugmenter
from cvchess.models.cnn import build_from_trial, count_params
from cvchess.train.time_estimator import (
    calibrate_seconds_per_step, estimate_total_seconds,
)
from cvchess.hardware import recommended_num_workers, apply_caps
from cvchess.logging_utils import get_logger

log = get_logger("train_cnn")


@dataclass
class TrainConfig:
    train_manifest: str
    test_manifest: str
    epochs: int = 8
    target_accuracy: float = 0.97
    budget_seconds: float = 3 * 3600
    n_trials: int = 25
    max_empty_ratio: float = 1.0
    device: str = "cuda"
    wandb_project: str = "chess-cnn-vs-yolo"
    seed: int = 42


def _device(cfg: TrainConfig) -> str:
    return cfg.device if torch.cuda.is_available() else "cpu"


def _loaders(cfg: TrainConfig, batch_size: int):
    train_ds = CellDataset(cfg.train_manifest, max_empty_ratio=cfg.max_empty_ratio)
    test_ds = CellDataset(cfg.test_manifest, max_empty_ratio=cfg.max_empty_ratio)
    nw = recommended_num_workers()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=nw, pin_memory=True)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    correct = total = occ_correct = occ_total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        occ = y != 0
        occ_correct += (pred[occ] == y[occ]).sum().item()
        occ_total += occ.sum().item()
    return {
        "accuracy": correct / max(1, total),
        "occupied_accuracy": occ_correct / max(1, occ_total),
    }


def _train_one(model, cfg, batch_size, lr, weight_decay, optimizer_name,
               aug_strength, device, max_seconds, wandb_run=None, trial=None):
    train_loader, test_loader = _loaders(cfg, batch_size)
    opt_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW,
               "sgd": torch.optim.SGD}[optimizer_name]
    optimizer = opt_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    augment = GpuAugmenter(aug_strength, device) if device.startswith("cuda") else None

    sps = calibrate_seconds_per_step(model, train_loader, device, optimizer, loss_fn)
    eta = estimate_total_seconds(sps, len(train_loader), cfg.epochs)
    log.info("Predicted training time: %.1f s (%.2f min)", eta, eta / 60)
    if wandb_run:
        wandb_run.log({"predicted_seconds": eta, "seconds_per_step": sps})

    start = time.perf_counter()
    best = {"occupied_accuracy": 0.0, "accuracy": 0.0}
    for epoch in range(cfg.epochs):
        model.train()
        last_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if augment is not None:
                x = augment(x)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            last_loss = loss.item()
        metrics = evaluate(model, test_loader, device)
        log.info("epoch %d | loss %.4f | occ_acc %.4f", epoch, last_loss,
                 metrics["occupied_accuracy"])
        if wandb_run:
            wandb_run.log({"epoch": epoch, "loss": last_loss, **metrics})
        if trial is not None:
            trial.report(metrics["occupied_accuracy"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if metrics["occupied_accuracy"] > best["occupied_accuracy"]:
            best = metrics
        if metrics["occupied_accuracy"] >= cfg.target_accuracy:
            log.info("Hit target accuracy — stopping early")
            break
        if time.perf_counter() - start > max_seconds:
            log.info("Hit per-trial time budget — stopping")
            break
    return best


def make_objective(cfg: TrainConfig):
    import wandb

    def objective(trial: optuna.Trial) -> float:
        torch.manual_seed(cfg.seed)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
        aug_strength = trial.suggest_float("aug_strength", 0.0, 0.8)
        model = build_from_trial(trial, num_classes=len(CNN_CLASSES))
        device = _device(cfg)
        model.to(device)
        run = wandb.init(project=cfg.wandb_project, group="optuna",
                         name=f"trial-{trial.number}", reinit=True,
                         config={**trial.params, "params": count_params(model)})
        try:
            per_trial_budget = cfg.budget_seconds / max(1, cfg.n_trials)
            best = _train_one(model, cfg, batch_size, lr, weight_decay,
                              optimizer_name, aug_strength, device,
                              per_trial_budget, wandb_run=run, trial=trial)
            run.log({"best_occupied_accuracy": best["occupied_accuracy"]})
            return best["occupied_accuracy"]
        finally:
            run.finish()

    return objective


def run_study(cfg: TrainConfig) -> optuna.Study:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    apply_caps()
    study = optuna.create_study(
        study_name="chess_cnn", direction="maximize",
        storage=OPTUNA_DB, load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    deadline = time.perf_counter() + cfg.budget_seconds

    def stop_when_done(study, trial):
        if study.best_value is not None and study.best_value >= cfg.target_accuracy:
            study.stop()
        if time.perf_counter() > deadline:
            study.stop()

    study.optimize(make_objective(cfg), n_trials=cfg.n_trials,
                   callbacks=[stop_when_done])
    log.info("Best occupied accuracy: %.4f params: %s",
             study.best_value, study.best_params)
    return study


def train_final_and_save(cfg: TrainConfig, study: optuna.Study):
    import wandb
    best = study.best_params
    device = _device(cfg)
    fixed = optuna.trial.FixedTrial(best)
    model = build_from_trial(fixed, num_classes=len(CNN_CLASSES)).to(device)
    run = wandb.init(project=cfg.wandb_project, group="final", name="final-model",
                     reinit=True, config=best)
    try:
        _train_one(model, cfg, best["batch_size"], best["lr"],
                   best["weight_decay"], best["optimizer"], best["aug_strength"],
                   device, cfg.budget_seconds, wandb_run=run)
    finally:
        run.finish()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights = MODELS_DIR / "chess_cnn.pt"
    torch.save(model.state_dict(), weights)
    (MODELS_DIR / "chess_cnn_config.json").write_text(json.dumps({
        "classes": CNN_CLASSES, "best_params": best,
    }, indent=2))
    log.info("Saved model to %s", weights)
    return weights
