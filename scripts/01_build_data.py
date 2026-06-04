"""File 1: download dataset (cache-aware), build CNN+YOLO manifests."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from cvchess.config import CACHE_DIR, ensure_dirs  # noqa: E402
from cvchess.hardware import log_summary, apply_caps  # noqa: E402
from cvchess.data.download import ensure_dataset  # noqa: E402
from cvchess.data.build_dataset import build_manifest, build_crop_cache  # noqa: E402
from cvchess.logging_utils import get_logger  # noqa: E402

log = get_logger("build_data")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-test", type=int, default=None)
    args = ap.parse_args()

    ensure_dirs()
    log_summary()
    apply_caps()
    root = ensure_dataset()
    build_manifest(root / "train", CACHE_DIR / "train_manifest.json", args.limit_train)
    build_manifest(root / "test", CACHE_DIR / "test_manifest.json", args.limit_test)
    build_crop_cache(CACHE_DIR / "train_manifest.json",
                     CACHE_DIR / "train_cells.npy", CACHE_DIR / "train_labels.npy")
    build_crop_cache(CACHE_DIR / "test_manifest.json",
                     CACHE_DIR / "test_cells.npy", CACHE_DIR / "test_labels.npy")
    log.info("Data build complete.")


if __name__ == "__main__":
    main()
