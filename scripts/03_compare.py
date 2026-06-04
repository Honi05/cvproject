"""File 3: compare custom CNN vs YOLO on the synthetic test set."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from cvchess.config import CACHE_DIR, MODELS_DIR, CNN_CLASSES  # noqa: E402
from cvchess.data.dataset import CellDataset  # noqa: E402
from cvchess.models.cnn import build_from_trial  # noqa: E402
from cvchess.eval.compare import (  # noqa: E402
    evaluate_cnn, measure_latency_ms, compute_profile, write_report,
)
from cvchess.hardware import recommended_num_workers, log_summary  # noqa: E402
import optuna  # noqa: E402


def load_cnn(device):
    cfg = json.loads((MODELS_DIR / "chess_cnn_config.json").read_text())
    model = build_from_trial(optuna.trial.FixedTrial(cfg["best_params"]),
                             num_classes=len(CNN_CLASSES))
    model.load_state_dict(torch.load(MODELS_DIR / "chess_cnn.pt", map_location=device))
    return model.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-weights", type=str, default=None,
                    help="path to trained YOLO best.pt; if set, includes YOLO row")
    args = ap.parse_args()
    log_summary()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tc = CACHE_DIR / "test_cells.npy"; tl = CACHE_DIR / "test_labels.npy"
    test_ds = CellDataset(CACHE_DIR / "test_manifest.json", max_empty_ratio=1.0,
                          cells_path=tc if tc.exists() else None,
                          labels_path=tl if tl.exists() else None)
    loader = DataLoader(test_ds, batch_size=256, num_workers=recommended_num_workers())

    cnn = load_cnn(device)
    rows = [{
        "model": "custom_cnn",
        "occ_accuracy": round(evaluate_cnn(cnn, loader, device), 4),
        "latency_ms": round(measure_latency_ms(cnn, device=device), 4),
        **compute_profile(cnn),
    }]

    if args.yolo_weights:
        from ultralytics import YOLO
        ymodel = YOLO(args.yolo_weights)
        rows.append({
            "model": "yolov8n",
            "occ_accuracy": "see_val_mAP",
            "latency_ms": round(measure_latency_ms(
                ymodel.model, input_shape=(1, 3, 400, 400), device=device), 4),
            "params": sum(p.numel() for p in ymodel.model.parameters()),
            "flops": None,
        })

    write_report(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
