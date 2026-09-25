import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.spike_transcription import (
    CHANNELS,
    CHUNK_BYTES,
    SAMPLE_RATE,
    SAMPLE_WIDTH_BYTES,
    SpikeFailure,
    require_api_key,
    require_audio_file,
)


def test_chunk_size_matches_100ms_of_canonical_pcm():
    assert SAMPLE_RATE == 16000
    assert CHANNELS == 1
    assert SAMPLE_WIDTH_BYTES == 2
    assert CHUNK_BYTES == 3200


def test_require_api_key_raises_when_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    try:
        require_api_key()
        assert False, "expected SpikeFailure"
    except SpikeFailure:
        pass


def test_require_api_key_returns_value_when_present(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert require_api_key() == "test-key"


def test_require_audio_file_raises_when_missing(tmp_path):
    missing = tmp_path / "does-not-exist.wav"
    try:
        require_audio_file(missing)
        assert False, "expected SpikeFailure"
    except SpikeFailure:
        pass


def test_require_audio_file_returns_path_when_present(tmp_path):
    existing = tmp_path / "test.wav"
    existing.write_bytes(b"\x00")
    assert require_audio_file(existing) == existing
