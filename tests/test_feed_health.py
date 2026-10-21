from src.feed_health import generate_feed, health_score, score_feed, status_for


def test_health_score_endpoints():
    assert health_score(1.0, 0.0, 0.0) == 100.0
    assert health_score(0.5, 0.0, 0.0) == 70.0    # 50% of rows missing
    assert health_score(1.0, 0.05, 0.0) == 60.0   # nulls at saturation
    assert health_score(1.0, 0.0, 0.20) == 70.0   # late at saturation
    assert health_score(0.0, 1.0, 1.0) == 0.0     # everything broken


def test_status_thresholds():
    assert status_for(95) == "healthy"
    assert status_for(80) == "warning"
    assert status_for(50) == "break"


def test_generated_feed_flags_exactly_the_three_injected_breaks():
    feed = generate_feed(seed=42)
    assert len(feed) == 24
    breaks = feed.loc[feed["status"] == "break", "month"].tolist()
    assert breaks == ["2025-07", "2026-01", "2026-07"]  # row drop, null spike, late spike
    others = feed[feed["status"] != "break"]
    assert (others["status"] == "healthy").all()
    assert feed["health_score"].between(0, 100).all()


def test_score_feed_is_reproducible():
    a = generate_feed(seed=9)
    b = generate_feed(seed=9)
    assert a.equals(b)
