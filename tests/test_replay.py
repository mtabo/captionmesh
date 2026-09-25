import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.providers.replay import ReplayTranscriber


def write_fixture(tmp_path, entries) -> Path:
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(entries))
    return path


async def test_replay_yields_segments_in_fixture_order_with_correct_fields(tmp_path):
    path = write_fixture(
        tmp_path,
        [
            {"delay_ms": 0, "text": "Hel", "is_final": False},
            {"delay_ms": 0, "text": "Hello", "is_final": False},
            {"delay_ms": 0, "text": "Hello.", "is_final": True, "language": "en"},
        ],
    )
    transcriber = ReplayTranscriber(path, realtime=False)

    segments = [seg async for seg in transcriber.transcribe(iter(()))]

    assert [s.text for s in segments] == ["Hel", "Hello", "Hello."]
    assert [s.is_final for s in segments] == [False, False, True]
    assert segments[-1].language == "en"
    assert segments[0].language is None


async def test_replay_ignores_the_audio_chunks_argument():
    """A ReplayTranscriber never needs real audio; passing garbage as
    audio_chunks must not matter, proving it does not consume the stream."""
    path_holder = {}

    async def never_iterated():
        raise AssertionError("audio_chunks should never be iterated by ReplayTranscriber")
        yield b""  # pragma: no cover

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "fixture.json"
        path.write_text(json.dumps([{"delay_ms": 0, "text": "hi", "is_final": True}]))
        transcriber = ReplayTranscriber(path, realtime=False)
        segments = [seg async for seg in transcriber.transcribe(never_iterated())]

    assert [s.text for s in segments] == ["hi"]


async def test_replay_raises_if_fixture_file_is_missing(tmp_path):
    transcriber = ReplayTranscriber(tmp_path / "does-not-exist.json", realtime=False)

    try:
        async for _ in transcriber.transcribe(iter(())):
            pass
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


async def test_replay_realtime_false_does_not_sleep(tmp_path):
    path = write_fixture(
        tmp_path,
        [{"delay_ms": 5000, "text": "slow", "is_final": True}],
    )
    transcriber = ReplayTranscriber(path, realtime=False)

    t0 = time.monotonic()
    segments = [seg async for seg in transcriber.transcribe(iter(()))]
    elapsed = time.monotonic() - t0

    assert [s.text for s in segments] == ["slow"]
    assert elapsed < 1.0  # would be ~5s if delay_ms were honored


async def test_replay_realtime_true_honors_delay(tmp_path):
    path = write_fixture(
        tmp_path,
        [
            {"delay_ms": 0, "text": "a", "is_final": False},
            {"delay_ms": 50, "text": "b", "is_final": True},
        ],
    )
    transcriber = ReplayTranscriber(path, realtime=True)

    t0 = time.monotonic()
    segments = [seg async for seg in transcriber.transcribe(iter(()))]
    elapsed = time.monotonic() - t0

    assert [s.text for s in segments] == ["a", "b"]
    assert elapsed >= 0.05
