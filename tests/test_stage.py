import asyncio
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


class FakeTranslator:
    """A TranslationProvider stub recording calls instead of hitting Gemini."""

    def __init__(self, fail_for=frozenset()):
        self.calls = []  # (text, source_language, target_language)
        self._fail_for = fail_for

    async def translate(self, text: str, source_language: str, target_language: str) -> str:
        self.calls.append((text, source_language, target_language))
        if target_language in self._fail_for:
            raise RuntimeError("translation failed")
        return f"[{target_language}] {text}"


class SlowTranslator:
    """A TranslationProvider stub whose translate() blocks until released,
    used to prove that scheduling a translation never blocks the pipeline."""

    def __init__(self):
        self.calls = []  # (text, source_language, target_language)
        self._release = asyncio.Event()

    async def translate(self, text: str, source_language: str, target_language: str) -> str:
        self.calls.append((text, source_language, target_language))
        await self._release.wait()
        return f"[{target_language}] {text}"

    def release(self) -> None:
        self._release.set()


class RecordingBroadcaster(Broadcaster):
    def __init__(self):
        super().__init__()
        self.published = []

    def publish(self, event):
        self.published.append(event)
        super().publish(event)


def make_stage_config(language: str = "auto", targets=("es",)) -> StageConfig:
    return StageConfig(
        id="main",
        name="Main Stage",
        language=language,
        targets=list(targets),
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


async def test_interim_events_never_trigger_translation(tmp_path):
    segments = [TranscriptSegment(text="partial", is_final=False)]
    store = JsonlEventStore(base_dir=tmp_path)
    translator = FakeTranslator()
    pipeline = StagePipeline(
        make_stage_config(targets=["es"]), FakeTranscriber(segments), store, Broadcaster(), translator
    )

    await pipeline.run()

    assert translator.calls == []


async def test_no_translator_configured_skips_translation_gracefully(tmp_path):
    segments = [TranscriptSegment(text="final.", is_final=True, language="en")]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    pipeline = StagePipeline(
        make_stage_config(targets=["es"]), FakeTranscriber(segments), store, broadcaster
    )  # translator omitted (defaults to None)

    await pipeline.run()  # must not raise

    assert [e.type for e in broadcaster.published] == ["caption.final"]


async def test_final_event_triggers_translation_for_each_configured_target(tmp_path):
    segments = [TranscriptSegment(text="Hello.", is_final=True, language="en")]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    translator = FakeTranslator()
    pipeline = StagePipeline(
        make_stage_config(targets=["es", "fr"]),
        FakeTranscriber(segments),
        store,
        broadcaster,
        translator,
    )

    await pipeline.run()

    assert translator.calls == [
        ("Hello.", "en", "es"),
        ("Hello.", "en", "fr"),
    ]
    translation_events = [e for e in broadcaster.published if e.type == "caption.translation"]
    assert {e.language for e in translation_events} == {"es", "fr"}
    assert all(e.seg_id == "main-000001" for e in translation_events)
    assert all(e.source_language == "en" for e in translation_events)
    assert {e.text for e in translation_events} == {"[es] Hello.", "[fr] Hello."}


async def test_translation_seg_id_matches_the_source_final_seg_id(tmp_path):
    segments = [
        TranscriptSegment(text="One.", is_final=True, language="en"),
        TranscriptSegment(text="Two.", is_final=True, language="en"),
    ]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    translator = FakeTranslator()
    pipeline = StagePipeline(
        make_stage_config(targets=["es"]), FakeTranscriber(segments), store, broadcaster, translator
    )

    await pipeline.run()

    finals = [e for e in broadcaster.published if e.type == "caption.final"]
    translations = [e for e in broadcaster.published if e.type == "caption.translation"]
    assert [f.seg_id for f in finals] == ["main-000001", "main-000002"]
    assert [t.seg_id for t in translations] == ["main-000001", "main-000002"]


async def test_source_language_equal_to_target_does_not_trigger_translation(tmp_path):
    segments = [TranscriptSegment(text="Hola.", is_final=True, language="es")]
    store = JsonlEventStore(base_dir=tmp_path)
    translator = FakeTranslator()
    pipeline = StagePipeline(
        make_stage_config(targets=["es"]), FakeTranscriber(segments), store, Broadcaster(), translator
    )

    await pipeline.run()

    assert translator.calls == []


async def test_translation_events_are_persisted_alongside_finals(tmp_path):
    segments = [TranscriptSegment(text="Hello.", is_final=True, language="en")]
    store = JsonlEventStore(base_dir=tmp_path)
    translator = FakeTranslator()
    pipeline = StagePipeline(
        make_stage_config(targets=["es"]), FakeTranscriber(segments), store, Broadcaster(), translator
    )

    await pipeline.run()

    persisted = [json.loads(line) for line in (tmp_path / "main.jsonl").read_text().splitlines()]
    assert [e["type"] for e in persisted] == ["caption.final", "caption.translation"]
    assert persisted[1]["seg_id"] == persisted[0]["seg_id"]
    assert persisted[1]["source_language"] == "en"
    assert persisted[1]["language"] == "es"


async def test_translation_failure_for_one_target_does_not_block_other_targets_or_crash_stage(tmp_path):
    segments = [TranscriptSegment(text="Hello.", is_final=True, language="en")]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    translator = FakeTranslator(fail_for={"es"})
    pipeline = StagePipeline(
        make_stage_config(targets=["es", "fr"]),
        FakeTranscriber(segments),
        store,
        broadcaster,
        translator,
    )

    await pipeline.run()  # must not raise

    assert pipeline.status == "stopped"
    translation_events = [e for e in broadcaster.published if e.type == "caption.translation"]
    assert {e.language for e in translation_events} == {"fr"}


async def test_translation_latency_is_recorded_in_stats(tmp_path):
    segments = [TranscriptSegment(text="Hello.", is_final=True, language="en")]
    store = JsonlEventStore(base_dir=tmp_path)
    translator = FakeTranslator()
    pipeline = StagePipeline(
        make_stage_config(targets=["es"]), FakeTranscriber(segments), store, Broadcaster(), translator
    )

    await pipeline.run()

    summary = pipeline.stats.translation.summary()
    assert summary.count == 1
    assert summary.first_ms is not None and summary.first_ms >= 0


async def test_handle_final_does_not_block_on_a_translation_that_never_completes(tmp_path):
    """A translator that never resolves must not prevent run() from finishing:
    proves _handle_final does not await translation, and that shutdown
    cancels a stuck task instead of hanging forever."""
    segments = [
        TranscriptSegment(text="One.", is_final=True, language="en"),
        TranscriptSegment(text="Two.", is_final=True, language="en"),
    ]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    translator = SlowTranslator()  # never released in this test
    pipeline = StagePipeline(
        make_stage_config(targets=["es"]),
        FakeTranscriber(segments),
        store,
        broadcaster,
        translator,
        translation_shutdown_grace_seconds=0.05,
    )

    await asyncio.wait_for(pipeline.run(), timeout=2.0)

    assert pipeline.status == "stopped"
    finals = [e for e in broadcaster.published if e.type == "caption.final"]
    assert [f.text for f in finals] == ["One.", "Two."]


async def test_subsequent_finals_are_processed_while_an_earlier_translation_is_still_pending(tmp_path):
    segments = [
        TranscriptSegment(text="One.", is_final=True, language="en"),
        TranscriptSegment(text="Two.", is_final=True, language="en"),
    ]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    translator = SlowTranslator()
    pipeline = StagePipeline(
        make_stage_config(targets=["es"]),
        FakeTranscriber(segments),
        store,
        broadcaster,
        translator,
        translation_shutdown_grace_seconds=0.05,
    )

    run_task = asyncio.create_task(pipeline.run())
    await asyncio.sleep(0)  # let run() advance to its first real suspension point

    # Both finals already processed even though neither translation has resolved.
    finals = [e for e in broadcaster.published if e.type == "caption.final"]
    assert [f.text for f in finals] == ["One.", "Two."]
    assert pipeline.pending_translation_count == 2

    translator.release()
    await asyncio.wait_for(run_task, timeout=2.0)

    translations = [e for e in broadcaster.published if e.type == "caption.translation"]
    assert {t.text for t in translations} == {"[es] One.", "[es] Two."}
    assert pipeline.pending_translation_count == 0


async def test_multiple_translation_tasks_for_one_final_are_pending_independently(tmp_path):
    segments = [TranscriptSegment(text="Hello.", is_final=True, language="en")]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    translator = SlowTranslator()
    pipeline = StagePipeline(
        make_stage_config(targets=["es", "fr"]),
        FakeTranscriber(segments),
        store,
        broadcaster,
        translator,
        translation_shutdown_grace_seconds=0.05,
    )

    run_task = asyncio.create_task(pipeline.run())
    await asyncio.sleep(0)

    assert pipeline.pending_translation_count == 2

    translator.release()
    await asyncio.wait_for(run_task, timeout=2.0)

    translations = [e for e in broadcaster.published if e.type == "caption.translation"]
    assert {t.language for t in translations} == {"es", "fr"}
    assert pipeline.pending_translation_count == 0


async def test_shutdown_cancels_stuck_translation_tasks_without_leaving_orphans(tmp_path):
    segments = [TranscriptSegment(text="Hello.", is_final=True, language="en")]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    translator = SlowTranslator()  # never released
    pipeline = StagePipeline(
        make_stage_config(targets=["es"]),
        FakeTranscriber(segments),
        store,
        broadcaster,
        translator,
        translation_shutdown_grace_seconds=0.05,
    )

    baseline_tasks = set(asyncio.all_tasks())

    await asyncio.wait_for(pipeline.run(), timeout=2.0)

    assert pipeline.pending_translation_count == 0
    leftover = set(asyncio.all_tasks()) - baseline_tasks - {asyncio.current_task()}
    assert all(t.done() for t in leftover), "no task created by this stage should remain unfinished"
    # No translation event was ever emitted, since the task was cancelled before completing.
    translations = [e for e in broadcaster.published if e.type == "caption.translation"]
    assert translations == []


async def test_a_translation_that_completes_within_the_grace_window_is_not_cancelled(tmp_path):
    segments = [TranscriptSegment(text="Hello.", is_final=True, language="en")]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    translator = SlowTranslator()
    pipeline = StagePipeline(
        make_stage_config(targets=["es"]),
        FakeTranscriber(segments),
        store,
        broadcaster,
        translator,
        translation_shutdown_grace_seconds=2.0,
    )

    run_task = asyncio.create_task(pipeline.run())
    await asyncio.sleep(0)
    translator.release()  # resolves well within the 2s grace window

    await asyncio.wait_for(run_task, timeout=2.0)

    translations = [e for e in broadcaster.published if e.type == "caption.translation"]
    assert [t.text for t in translations] == ["[es] Hello."]
