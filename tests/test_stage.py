import json
import sys
from pathlib import Path
from typing import AsyncIterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.broadcast import Broadcaster
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


class RecordingBroadcaster(Broadcaster):
    def __init__(self):
        super().__init__()
        self.published = []

    def publish(self, event):
        self.published.append(event)
        super().publish(event)


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
    pipeline = StagePipeline(make_stage_config(), FakeTranscriber(segments), store, Broadcaster())

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
    pipeline = StagePipeline(make_stage_config(), FakeTranscriber(segments), store, Broadcaster())

    await pipeline.run()

    lines = (tmp_path / "main.jsonl").read_text().splitlines()
    assert len(lines) == 1


async def test_interim_and_final_of_same_segment_share_seg_id(tmp_path):
    segments = [
        TranscriptSegment(text="Hel", is_final=False),
        TranscriptSegment(text="Hello", is_final=False),
        TranscriptSegment(text="Hello.", is_final=True, language="en"),
    ]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    pipeline = StagePipeline(make_stage_config(), FakeTranscriber(segments), store, broadcaster)

    await pipeline.run()

    seg_ids = {e.seg_id for e in broadcaster.published}
    assert seg_ids == {"main-000001"}
    types = [e.type for e in broadcaster.published]
    assert types == ["caption.interim", "caption.interim", "caption.final"]


async def test_a_new_segment_after_final_gets_a_new_seg_id(tmp_path):
    segments = [
        TranscriptSegment(text="Hi", is_final=False),
        TranscriptSegment(text="Hi.", is_final=True, language="en"),
        TranscriptSegment(text="Bye", is_final=False),
        TranscriptSegment(text="Bye.", is_final=True, language="en"),
    ]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    pipeline = StagePipeline(make_stage_config(), FakeTranscriber(segments), store, broadcaster)

    await pipeline.run()

    seg_ids = [e.seg_id for e in broadcaster.published]
    assert seg_ids == ["main-000001", "main-000001", "main-000002", "main-000002"]


async def test_interim_events_are_broadcast_but_final_events_are_broadcast_and_persisted(tmp_path):
    segments = [
        TranscriptSegment(text="partial", is_final=False),
        TranscriptSegment(text="final.", is_final=True, language="en"),
    ]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    pipeline = StagePipeline(make_stage_config(), FakeTranscriber(segments), store, broadcaster)

    await pipeline.run()

    assert [e.type for e in broadcaster.published] == ["caption.interim", "caption.final"]
    persisted = (tmp_path / "main.jsonl").read_text().splitlines()
    assert len(persisted) == 1
    assert json.loads(persisted[0])["type"] == "caption.final"


async def test_language_falls_back_to_configured_language_when_provider_omits_it(tmp_path):
    segments = [TranscriptSegment(text="Hola.", is_final=True, language=None)]
    store = JsonlEventStore(base_dir=tmp_path)
    pipeline = StagePipeline(
        make_stage_config(language="es"), FakeTranscriber(segments), store, Broadcaster()
    )

    await pipeline.run()

    event = json.loads((tmp_path / "main.jsonl").read_text().splitlines()[0])
    assert event["language"] == "es"


async def test_language_falls_back_to_und_when_auto_and_provider_omits_it(tmp_path):
    segments = [TranscriptSegment(text="Hello.", is_final=True, language=None)]
    store = JsonlEventStore(base_dir=tmp_path)
    pipeline = StagePipeline(
        make_stage_config(language="auto"), FakeTranscriber(segments), store, Broadcaster()
    )

    await pipeline.run()

    event = json.loads((tmp_path / "main.jsonl").read_text().splitlines()[0])
    assert event["language"] == "und"


async def test_a_failing_transcriber_is_isolated_as_error_status(tmp_path):
    store = JsonlEventStore(base_dir=tmp_path)
    pipeline = StagePipeline(make_stage_config(), FailingTranscriber(), store, Broadcaster())

    await pipeline.run()  # must not raise

    assert pipeline.status == "error"


async def test_timing_is_propagated_from_segment_to_event_and_persisted(tmp_path):
    segments = [
        TranscriptSegment(text="final.", is_final=True, language="en", audio_elapsed_ms=1000.0, asr_latency_ms=250.0)
    ]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    pipeline = StagePipeline(make_stage_config(), FakeTranscriber(segments), store, broadcaster)

    await pipeline.run()

    published = broadcaster.published[0]
    assert published.timing.audio_elapsed_ms == 1000.0
    assert published.timing.asr_latency_ms == 250.0

    persisted = json.loads((tmp_path / "main.jsonl").read_text().splitlines()[0])
    assert persisted["timing"] == {"audio_elapsed_ms": 1000.0, "asr_latency_ms": 250.0}


async def test_missing_timing_from_provider_yields_no_timing_and_is_not_recorded(tmp_path):
    segments = [TranscriptSegment(text="final.", is_final=True, language="en")]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    pipeline = StagePipeline(make_stage_config(), FakeTranscriber(segments), store, broadcaster)

    await pipeline.run()  # must not raise

    assert broadcaster.published[0].timing is None
    persisted = json.loads((tmp_path / "main.jsonl").read_text().splitlines()[0])
    assert persisted["timing"] is None
    assert pipeline.stats.final.summary().count == 0


async def test_latency_stats_aggregate_across_interim_and_final_events(tmp_path):
    segments = [
        TranscriptSegment(text="a", is_final=False, audio_elapsed_ms=0.0, asr_latency_ms=100.0),
        TranscriptSegment(text="ab", is_final=False, audio_elapsed_ms=100.0, asr_latency_ms=300.0),
        TranscriptSegment(text="ab.", is_final=True, language="en", audio_elapsed_ms=200.0, asr_latency_ms=500.0),
    ]
    store = JsonlEventStore(base_dir=tmp_path)
    pipeline = StagePipeline(make_stage_config(), FakeTranscriber(segments), store, Broadcaster())

    await pipeline.run()

    interim_summary = pipeline.stats.interim.summary()
    assert interim_summary.count == 2
    assert interim_summary.first_ms == 100.0
    assert interim_summary.min_ms == 100.0
    assert interim_summary.max_ms == 300.0
    assert interim_summary.avg_ms == 200.0

    final_summary = pipeline.stats.final.summary()
    assert final_summary.count == 1
    assert final_summary.first_ms == final_summary.min_ms == final_summary.max_ms == 500.0
