"""Detect hardware, log a summary, and apply 90% memory caps."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cvchess.hardware import log_summary, apply_caps  # noqa: E402


def main() -> None:
    info = log_summary()
    apply_caps()
    if info.cuda_available:
        print(f"\nReady: {info.gpu_names[0]} with {info.gpu_total_gb[0]:.0f} GB")
    else:
        print("\nWARNING: no CUDA GPU detected — training will use CPU.")


if __name__ == "__main__":
    main()
