from __future__ import annotations
import resource
from dataclasses import dataclass, field

import psutil

from cvchess.config import MEMORY_FRACTION
from cvchess.logging_utils import get_logger

log = get_logger("hardware")


@dataclass
class HardwareInfo:
    cpu_count: int
    ram_total_gb: float
    gpu_count: int
    gpu_names: list[str] = field(default_factory=list)
    gpu_total_gb: list[float] = field(default_factory=list)
    cuda_available: bool = False
    memory_fraction: float = MEMORY_FRACTION


def detect() -> HardwareInfo:
    cpu_count = psutil.cpu_count(logical=True) or 1
    ram_total_gb = psutil.virtual_memory().total / (1024 ** 3)
    gpu_count, names, totals, cuda = 0, [], [], False
    try:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                p = torch.cuda.get_device_properties(i)
                names.append(p.name)
                totals.append(p.total_memory / (1024 ** 3))
    except Exception as e:  # torch missing or driver issue
        log.warning("GPU detection failed: %s", e)
    return HardwareInfo(
        cpu_count=cpu_count, ram_total_gb=ram_total_gb,
        gpu_count=gpu_count, gpu_names=names, gpu_total_gb=totals,
        cuda_available=cuda,
    )


def recommended_num_workers(cpu_count: int | None = None) -> int:
    if cpu_count is None:
        cpu_count = psutil.cpu_count(logical=True) or 1
    return max(1, min(32, cpu_count // 4))


def cap_gpu_memory(fraction: float = MEMORY_FRACTION) -> None:
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_per_process_memory_fraction(fraction, i)
            log.info("Capped GPU memory to %.0f%%", fraction * 100)
    except Exception as e:
        log.warning("Could not cap GPU memory: %s", e)


def cap_system_memory(fraction: float = MEMORY_FRACTION) -> None:
    total = psutil.virtual_memory().total
    soft_cap = int(total * fraction)
    try:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_hard = hard if hard != resource.RLIM_INFINITY else soft_cap
        resource.setrlimit(resource.RLIMIT_AS, (soft_cap, new_hard))
        log.info("Capped system RAM (RLIMIT_AS) to %.0f%% (%.1f GB)",
                 fraction * 100, soft_cap / (1024 ** 3))
    except (ValueError, OSError) as e:
        log.warning("Could not set RLIMIT_AS: %s", e)


def apply_caps(fraction: float = MEMORY_FRACTION) -> None:
    cap_gpu_memory(fraction)
    cap_system_memory(fraction)


def log_summary(info: HardwareInfo | None = None) -> HardwareInfo:
    if info is None:
        info = detect()
    log.info("CPU cores: %d", info.cpu_count)
    log.info("System RAM: %.1f GB", info.ram_total_gb)
    log.info("CUDA available: %s", info.cuda_available)
    for i, (n, g) in enumerate(zip(info.gpu_names, info.gpu_total_gb)):
        log.info("GPU %d: %s (%.1f GB)", i, n, g)
    log.info("Recommended num_workers: %d", recommended_num_workers(info.cpu_count))
    return info
