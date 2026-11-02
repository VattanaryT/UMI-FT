"""Frame-gap warnings for live wrist and center cameras."""

from PyriteEnvSuites.env_runners.frame_rate_watch import FrameRateWatch


def test_steady_15hz_is_ok_and_silent():
    watch = FrameRateWatch("wrist D405", expected_hz=15.0)
    msg = None
    t = 0.0
    for _ in range(16):
        msg = watch.tick(t)
        t += 1.0 / 15.0
    assert msg is None
    summary = watch.summary()
    assert summary["ok"]
    assert summary["frames"] == 16
    assert abs(summary["hz"] - 15.0) < 0.2


def test_gap_warns_with_dropped_frame_count():
    watch = FrameRateWatch("center Orbbec", expected_hz=15.0)
    assert watch.tick(0.0) is None
    msg = watch.tick(0.0 + 4.0 / 15.0)
    assert msg is not None
    assert "center Orbbec" in msg
    assert "dropped" in msg
    # Throttled: a second gap inside 1 s does not print again.
    assert watch.tick(0.0 + 8.0 / 15.0) is None
    assert watch.summary()["gap_warns"] == 1


def test_slow_average_is_not_ok():
    watch = FrameRateWatch("wrist D405", expected_hz=15.0, warn_ratio=0.8)
    t = 0.0
    for _ in range(6):
        watch.tick(t)
        t += 0.2  # 5 Hz
    summary = watch.summary()
    assert not summary["ok"]
    assert summary["hz"] < 12.0
    assert "LOW" in watch.summary_line()
