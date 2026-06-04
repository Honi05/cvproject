from cvchess.train.time_estimator import estimate_total_seconds


def test_estimate_scales_with_steps_and_epochs():
    secs = estimate_total_seconds(seconds_per_step=0.01, steps_per_epoch=1000, epochs=5)
    assert abs(secs - 50.0) < 1e-6


def test_estimate_nonnegative():
    assert estimate_total_seconds(0.0, 100, 10) == 0.0
