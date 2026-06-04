"""Train pico/n/s YOLO variants and record deployment + accuracy stats."""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dotenv import load_dotenv; load_dotenv()
from ultralytics import YOLO, settings
settings.update({"wandb": False})  # avoid ultralytics<->wandb integration breakage
from cvchess.config import OUTPUTS_DIR
from cvchess.logging_utils import get_logger
log = get_logger("yolo_all")

DATA = str(OUTPUTS_DIR / "yolo_dataset" / "data.yaml")
RUNS = str(OUTPUTS_DIR / "yolo_runs")
EPOCHS, IMGSZ, FRACTION, BATCH = 12, 416, 0.25, 64

MODELS = [
    ("yolo_pico", "/workspace/cvproject/configs/yolov8n.yaml", False),  # ~227k, scratch
    ("yolov8n",   "yolov8n.pt",  True),                                  # 3.16M, pretrained
    ("yolov8s",   "yolov8s.pt",  True),                                  # 11.2M, pretrained
]

results = []
for name, weights, pretrained in MODELS:
    log.info("=== Training %s (%s) ===", name, weights)
    t0 = time.time()
    model = YOLO(weights)
    model.train(data=DATA, epochs=EPOCHS, imgsz=IMGSZ, fraction=FRACTION,
                batch=BATCH, project=RUNS, name=name, exist_ok=True,
                verbose=False, plots=False, workers=16)
    train_s = time.time() - t0
    # validate for mAP + speed
    metrics = model.val(data=DATA, imgsz=IMGSZ, batch=BATCH, verbose=False,
                        project=RUNS, name=name + "_val", exist_ok=True)
    info = model.info(verbose=False)  # (layers, params, grads, gflops)
    best_pt = Path(RUNS) / name / "weights" / "best.pt"
    rec = {
        "model": name,
        "params": int(info[1]) if info else None,
        "gflops": round(float(info[3]), 4) if info and len(info) > 3 else None,
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "speed_ms_infer": round(float(metrics.speed.get("inference", 0)), 4),
        "speed_ms_total": round(sum(metrics.speed.values()), 4),
        "disk_mb": round(best_pt.stat().st_size / 1e6, 3) if best_pt.exists() else None,
        "train_seconds": round(train_s, 1),
        "weights": str(best_pt),
    }
    results.append(rec)
    log.info("DONE %s: %s", name, json.dumps(rec))
    json.dump(results, open("/tmp/yolo_results.json", "w"), indent=2)

print("ALL YOLO DONE")
print(json.dumps(results, indent=2))
