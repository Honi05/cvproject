"""File 2: Optuna-tuned CNN training with wandb + time prediction + save."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from cvchess.config import CACHE_DIR, ensure_dirs  # noqa: E402
from cvchess.hardware import log_summary  # noqa: E402
from cvchess.train.train_cnn import TrainConfig, run_study, train_final_and_save  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--target", type=float, default=0.97)
    ap.add_argument("--budget-hours", type=float, default=3.0)
    args = ap.parse_args()

    ensure_dirs()
    log_summary()
    cfg = TrainConfig(
        train_manifest=str(CACHE_DIR / "train_manifest.json"),
        test_manifest=str(CACHE_DIR / "test_manifest.json"),
        epochs=args.epochs, n_trials=args.n_trials,
        target_accuracy=args.target, budget_seconds=args.budget_hours * 3600,
    )
    study = run_study(cfg)
    train_final_and_save(cfg, study)


if __name__ == "__main__":
    main()
