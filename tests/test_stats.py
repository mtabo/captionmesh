import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.stats import LatencySampler, StageLatencyStats


def test_empty_sampler_summary_is_safe_and_has_no_average():
    sampler = LatencySampler()

    summary = sampler.summary()

    assert summary.count == 0
    assert summary.first_ms is None
    assert summary.min_ms is None
    assert summary.avg_ms is None
    assert summary.max_ms is None


def test_sampler_tracks_first_min_avg_max():
    sampler = LatencySampler()
    for value in [300.0, 100.0, 500.0]:
        sampler.record(value)

    summary = sampler.summary()

    assert summary.count == 3
    assert summary.first_ms == 300.0
    assert summary.min_ms == 100.0
    assert summary.max_ms == 500.0
    assert summary.avg_ms == 300.0


def test_stage_latency_stats_keeps_interim_and_final_separate():
    stats = StageLatencyStats()
    stats.record_interim(100.0)
    stats.record_interim(200.0)
    stats.record_final(400.0)

    summary = stats.summary()

    assert summary["interim"]["count"] == 2
    assert summary["final"]["count"] == 1
    assert summary["final"]["first_ms"] == 400.0


def test_stage_latency_stats_ignores_none_latency():
    stats = StageLatencyStats()
    stats.record_interim(None)
    stats.record_final(None)

    summary = stats.summary()

    assert summary["interim"]["count"] == 0
    assert summary["final"]["count"] == 0
