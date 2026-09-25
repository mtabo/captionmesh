import json
import sys
from pathlib import Path
from typing import AsyncIterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import FileSourceConfig, StageConfig
from app.providers import TranscriptSegment
from app.stage import StagePipeline
from app.store import JsonlEventStore


class FakeTranscriber:
    """A TranscriptionProvider stub that ignores the audio and yields fixed segments."""

    def __init__(self, segments):
        self._segments = segments

    async def transcribe(self, audio_chunks) -> AsyncIterator[TranscriptSegment]:
        for segment in self._segments:
            yield segment


class FailingTranscriber:
    async def transcribe(self, audio_chunks) -> AsyncIterator[TranscriptSegment]:
        if False:
            yield  # pragma: no cover - makes this an async generator
        raise RuntimeError("session dropped")


def make_stage_config(language: str = "auto") -> StageConfig:
    return StageConfig(
        id="main",
        name="Main Stage",
        language=language,
        targets=["es"],
        source=FileSourceConfig(type="file", path="unused.wav"),
    )


async def test_final_segments_get_stable_incrementing_seg_ids(tmp_path):
    segments = [
        TranscriptSegment(text="Hello", is_final=False),
        TranscriptSegment(text="Hello.", is_final=True, language="en"),
        TranscriptSegment(text="World", is_final=False),
        TranscriptSegment(text="World.", is_final=True, language="en"),
    ]
    store = JsonlEventStore(base_dir=tmp_path)
    pipeline = StagePipeline(make_stage_config(), FakeTranscriber(segments), store)

    await pipeline.run()

    assert pipeline.status == "stopped"
    lines = (tmp_path / "main.jsonl").read_text().splitlines()
    events = [json.loads(line) for line in lines]

    assert [e["seg_id"] for e in events] == ["main-000001", "main-000002"]
    assert [e["text"] for e in events] == ["Hello.", "World."]
    assert all(e["stage_id"] == "main" for e in events)


async def test_interim_segments_are_not_persisted(tmp_path):
    segments = [
        TranscriptSegment(text="partial", is_final=False),
        TranscriptSegment(text="final.", is_final=True, language="en"),
    ]
    store = JsonlEventStore(base_dir=tmp_path)
    pipeline = StagePipeline(make_stage_config(), FakeTranscriber(segments), store)

    await pipeline.run()

    lines = (tmp_path / "main.jsonl").read_text().splitlines()
    assert len(lines) == 1


async def test_language_falls_back_to_configured_language_when_provider_omits_it(tmp_path):
    segments = [TranscriptSegment(text="Hola.", is_final=True, language=None)]
    store = JsonlEventStore(base_dir=tmp_path)
    pipeline = StagePipeline(make_stage_config(language="es"), FakeTranscriber(segments), store)

    await pipeline.run()

    event = json.loads((tmp_path / "main.jsonl").read_text().splitlines()[0])
    assert event["language"] == "es"


async def test_language_falls_back_to_und_when_auto_and_provider_omits_it(tmp_path):
    segments = [TranscriptSegment(text="Hello.", is_final=True, language=None)]
    store = JsonlEventStore(base_dir=tmp_path)
    pipeline = StagePipeline(make_stage_config(language="auto"), FakeTranscriber(segments), store)

    await pipeline.run()

    event = json.loads((tmp_path / "main.jsonl").read_text().splitlines()[0])
    assert event["language"] == "und"


async def test_a_failing_transcriber_is_isolated_as_error_status(tmp_path):
    store = JsonlEventStore(base_dir=tmp_path)
    pipeline = StagePipeline(make_stage_config(), FailingTranscriber(), store)

    await pipeline.run()  # must not raise

    assert pipeline.status == "error"
