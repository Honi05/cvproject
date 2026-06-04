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


def test_cap_system_memory_returns_threshold_and_does_not_crash():
    from cvchess.hardware import cap_system_memory
    import psutil
    t = cap_system_memory(0.9)
    total_gb = psutil.virtual_memory().total / (1024 ** 3)
    assert 0 < t <= total_gb


def test_system_memory_ok_returns_bool():
    from cvchess.hardware import system_memory_ok
    assert isinstance(system_memory_ok(0.9), bool)


def test_apply_caps_does_not_set_address_space_limit():
    import resource
    from cvchess.hardware import apply_caps
    before = resource.getrlimit(resource.RLIMIT_AS)
    apply_caps(0.9)
    after = resource.getrlimit(resource.RLIMIT_AS)
    assert before == after  # we must NOT modify RLIMIT_AS anymore
