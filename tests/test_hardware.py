from cvchess.hardware import detect, recommended_num_workers, HardwareInfo


def test_detect_returns_sane_values():
    info = detect()
    assert isinstance(info, HardwareInfo)
    assert info.cpu_count >= 1
    assert info.ram_total_gb > 0
    assert info.gpu_count >= 0  # may be 0 on CI


def test_recommended_workers_bounded():
    n = recommended_num_workers(cpu_count=112)
    assert 1 <= n <= 32  # capped, not 112


def test_memory_fraction_constant():
    info = detect()
    assert 0 < info.memory_fraction <= 0.9
