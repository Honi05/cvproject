"""Consolidated CNN-vs-YOLO comparison: accuracy (apples-to-apples per-piece),
deployment stats (params/FLOPs/latency/size), wandb summary."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dotenv import load_dotenv; load_dotenv()
import numpy as np, torch
from cvchess.config import CACHE_DIR, OUTPUTS_DIR, CELL, GRID
from cvchess.eval.compare import write_report
from cvchess.logging_utils import get_logger
log = get_logger("full_compare")

N_SAMPLE = 1000  # test boards for per-piece YOLO eval

def yolo_occupied_accuracy(weights, imgsz=416):
    """Map each YOLO detection to its grid cell; occupied-cell top-1 accuracy
    (directly comparable to the CNN's occupied accuracy)."""
    from ultralytics import YOLO
    model = YOLO(weights)
    n_params = sum(p.numel() for p in model.model.parameters())
    gflops = None
    try:
        info = model.info(verbose=False)  # (layers, params, grads, gflops)
        if info and len(info) >= 4 and info[3]:
            gflops = round(float(info[3]), 4)
    except Exception:
        pass
    recs = json.loads((CACHE_DIR / "test_manifest.json").read_text())[:N_SAMPLE]
    paths = [r["path"] for r in recs]
    t0 = time.time()
    results = model.predict(paths, imgsz=imgsz, verbose=False, batch=64)
    infer_s = time.time() - t0
    correct = total = 0
    for r, res in zip(recs, results):
        gt = r["cell_labels"]  # 64 CNN indices (0 empty, 1..12 pieces)
        best = {}  # cell_idx -> (conf, yolo_cls)
        if res.boxes is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            conf = res.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, conf):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                col, row = int(cx // CELL), int(cy // CELL)
                if not (0 <= col < GRID and 0 <= row < GRID):
                    continue
                ci = row * GRID + col
                if ci not in best or cf > best[ci][0]:
                    best[ci] = (cf, c)
        for ci in range(GRID * GRID):
            if gt[ci] == 0:
                continue
            total += 1
            pred_cnn = best[ci][1] + 1 if ci in best else 0  # yolo cls -> cnn idx
            if pred_cnn == gt[ci]:
                correct += 1
    return correct / max(1, total), infer_s / len(recs) * 1000.0, n_params, gflops

def main():
    cnn = json.loads(Path("/tmp/cnn_stats.json").read_text())
    yolo = json.loads(Path("/tmp/yolo_results.json").read_text())
    rows = [{
        "model": "custom_cnn",
        "occ_accuracy": cnn["cnn_occ_acc_test"],
        "params": cnn["params"],
        "gflops": cnn["gflops_per_cell"],
        "latency_ms_per_board": cnn["latency_ms_per_board_64cells"],
        "disk_mb": cnn["disk_mb"],
        "map50_95": "n/a (classifier)",
        "train_seconds": 38,  # ~1 epoch to target
    }]
    for y in yolo:
        log.info("Per-piece eval for %s ...", y["model"])
        acc, lat, n_params, gflops = yolo_occupied_accuracy(y["weights"])
        rows.append({
            "model": y["model"],
            "occ_accuracy": round(acc, 4),
            "params": n_params,
            "gflops": gflops,
            "latency_ms_per_board": round(lat, 4),
            "disk_mb": y["disk_mb"],
            "map50_95": y["map50_95"],
            "train_seconds": y["train_seconds"],
        })
    p = write_report(rows, OUTPUTS_DIR / "comparison.json")
    print(json.dumps(rows, indent=2))
    # wandb summary
    try:
        import wandb
        run = wandb.init(project="chess-cnn-vs-yolo", group="comparison",
                         name="cnn-vs-yolo-summary", reinit=True)
        cols = list(rows[0].keys())
        # map50_95 mixes numbers and the CNN's "n/a" string -> stringify for wandb
        tbl = wandb.Table(columns=cols, data=[
            [str(r.get(k)) if k == "map50_95" else r.get(k) for k in cols]
            for r in rows])
        run.log({"comparison": tbl})
        run.finish()
        print("logged comparison table to wandb")
    except Exception as e:
        log.warning("wandb summary failed: %s", e)
    print("report:", p)

if __name__ == "__main__":
    main()
