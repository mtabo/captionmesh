import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.sources.ffmpeg import FileAudioSource


def test_deadline_target_is_absolute_stream_start_plus_chunk_index_times_chunk_ms():
    delay = FileAudioSource._pacing_delay_seconds(
        "deadline", stream_started_at=100.0, chunk_index=3, chunk_ms=100, now=100.25
    )
    # target = 100.0 + 3*0.1 = 100.3; delay = 100.3 - 100.25
    assert delay == pytest.approx(0.05)


def test_deadline_delay_is_clamped_to_zero_when_already_behind_schedule():
    delay = FileAudioSource._pacing_delay_seconds(
        "deadline", stream_started_at=0.0, chunk_index=5, chunk_ms=100, now=0.6
    )
    # target = 0.5s, now = 0.6s -> already late, must not return a negative sleep
    assert delay == 0.0


def test_naive_delay_is_always_the_fixed_chunk_duration_regardless_of_now():
    delay_on_schedule = FileAudioSource._pacing_delay_seconds(
        "naive", stream_started_at=0.0, chunk_index=5, chunk_ms=100, now=0.5
    )
    delay_when_behind = FileAudioSource._pacing_delay_seconds(
        "naive", stream_started_at=0.0, chunk_index=5, chunk_ms=100, now=2.0
    )
    assert delay_on_schedule == pytest.approx(0.1)
    assert delay_when_behind == pytest.approx(0.1)  # blind to how far behind we already are


def _simulate_total_drift_ms(pacing: str, n_chunks: int, chunk_ms: int, overhead_s: float) -> float:
    """Simulate n_chunks iterations where each incurs a fixed per-iteration
    processing overhead (standing in for downstream await time) before the
    pacing decision is made, then advance the fake clock by the computed
    sleep. Returns final (wall_elapsed - nominal_audio_duration) in ms.
    """
    stream_started_at = 0.0
    clock = stream_started_at
    for chunk_index in range(1, n_chunks + 1):
        clock += overhead_s  # time consumed elsewhere in the loop before we get to pace
        delay = FileAudioSource._pacing_delay_seconds(
            pacing, stream_started_at, chunk_index, chunk_ms, clock
        )
        clock += delay
    nominal_audio_ms = n_chunks * chunk_ms
    wall_elapsed_ms = (clock - stream_started_at) * 1000
    return wall_elapsed_ms - nominal_audio_ms


def test_naive_pacing_accumulates_drift_proportional_to_iteration_count():
    n_chunks = 50
    overhead_s = 0.02
    drift_ms = _simulate_total_drift_ms("naive", n_chunks, chunk_ms=100, overhead_s=overhead_s)

    expected_drift_ms = n_chunks * overhead_s * 1000
    assert drift_ms == pytest.approx(expected_drift_ms, rel=1e-6)


def test_deadline_pacing_does_not_accumulate_drift_from_the_same_overhead():
    n_chunks = 50
    overhead_s = 0.02
    drift_ms = _simulate_total_drift_ms("deadline", n_chunks, chunk_ms=100, overhead_s=overhead_s)

    # Bounded by a single chunk's worth of unavoidable overhead, not by n_chunks.
    assert abs(drift_ms) < overhead_s * 1000 * 2
