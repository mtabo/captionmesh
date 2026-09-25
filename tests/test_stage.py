import asyncio
import json
import sys
import time
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


class FakeSegmenter:
    """A SegmentationProvider stub.

    `results` maps an exact input text to the list[str] it should return;
    any text not in the map is returned unchanged as a single segment
    (mirrors the "no split needed" case). `fail_for` raises for those exact
    texts. `hang_for` blocks until `release()` is called — used to prove
    the segmentation timeout/fallback actually bounds a stuck call.
    """

    def __init__(self, results=None, fail_for=frozenset(), hang_for=frozenset()):
        self.calls = []
        self._results = results or {}
        self._fail_for = fail_for
        self._hang_for = hang_for
        self._release = asyncio.Event()

    async def segment(self, text: str) -> list[str]:
        self.calls.append(text)
        if text in self._hang_for:
            await self._release.wait()
        if text in self._fail_for:
            raise RuntimeError("segmentation failed")
        return self._results.get(text, [text])

    def release(self) -> None:
        self._release.set()


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

    # Interims are published immediately as they arrive; finals are handed
    # off to the segmentation worker and only published once it processes
    # them (see StagePipeline._run_segmentation_worker), so with a
    # zero-await FakeTranscriber all interims land before any final —
    # each stream is still independently in the right order, which is
    # what this test is actually about.
    interim_seg_ids = [e.seg_id for e in broadcaster.published if e.type == "caption.interim"]
    final_seg_ids = [e.seg_id for e in broadcaster.published if e.type == "caption.final"]
    assert interim_seg_ids == ["main-000001", "main-000002"]
    assert final_seg_ids == ["main-000001", "main-000002"]


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
    # Finals are now handed off to a separate segmentation-worker task (see
    # StagePipeline._run_segmentation_worker) rather than published inline,
    # so give it a couple of real event-loop turns to actually drain the
    # queue and schedule translations — a single sleep(0) is not enough to
    # guarantee that worker has run yet.
    await asyncio.sleep(0.01)

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
    # See comment in the previous test — the segmentation worker needs a
    # couple of real turns to drain its queue before translations exist.
    await asyncio.sleep(0.01)

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


async def test_a_split_final_gets_derived_seg_ids_for_each_sub_segment(tmp_path):
    segments = [TranscriptSegment(text="Hello. World.", is_final=True, language="en")]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    segmenter = FakeSegmenter(results={"Hello. World.": ["Hello.", "World."]})
    pipeline = StagePipeline(
        make_stage_config(targets=[]), FakeTranscriber(segments), store, broadcaster, segmenter=segmenter
    )

    await pipeline.run()

    finals = [e for e in broadcaster.published if e.type == "caption.final"]
    assert [f.text for f in finals] == ["Hello.", "World."]
    assert [f.seg_id for f in finals] == ["main-000001-1", "main-000001-2"]


async def test_a_single_returned_segment_keeps_the_plain_seg_id(tmp_path):
    """Even with a real segmenter configured, if it decides not to split
    (or there was nothing to split), the derived "-N" suffix is not
    introduced — the plain seg_id format stays unchanged."""
    segments = [TranscriptSegment(text="Hi.", is_final=True, language="en")]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    segmenter = FakeSegmenter()  # returns [text] unchanged for anything not in `results`
    pipeline = StagePipeline(
        make_stage_config(targets=[]), FakeTranscriber(segments), store, broadcaster, segmenter=segmenter
    )

    await pipeline.run()

    finals = [e for e in broadcaster.published if e.type == "caption.final"]
    assert [f.seg_id for f in finals] == ["main-000001"]


async def test_each_sub_segment_gets_its_own_independent_translation(tmp_path):
    segments = [TranscriptSegment(text="Hello. World.", is_final=True, language="en")]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    segmenter = FakeSegmenter(results={"Hello. World.": ["Hello.", "World."]})
    translator = FakeTranslator()
    pipeline = StagePipeline(
        make_stage_config(targets=["es"]),
        FakeTranscriber(segments),
        store,
        broadcaster,
        translator,
        segmenter=segmenter,
    )

    await pipeline.run()

    assert translator.calls == [("Hello.", "en", "es"), ("World.", "en", "es")]
    translations = [e for e in broadcaster.published if e.type == "caption.translation"]
    assert {(t.seg_id, t.text) for t in translations} == {
        ("main-000001-1", "[es] Hello."),
        ("main-000001-2", "[es] World."),
    }


async def test_segmentation_failure_falls_back_to_the_original_text_as_one_segment(tmp_path):
    segments = [TranscriptSegment(text="Hello there.", is_final=True, language="en")]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    segmenter = FakeSegmenter(fail_for={"Hello there."})
    pipeline = StagePipeline(
        make_stage_config(targets=[]), FakeTranscriber(segments), store, broadcaster, segmenter=segmenter
    )

    await pipeline.run()  # must not raise — the caption is never lost

    finals = [e for e in broadcaster.published if e.type == "caption.final"]
    assert [f.text for f in finals] == ["Hello there."]
    assert [f.seg_id for f in finals] == ["main-000001"]  # no derived suffix — fallback is one segment


async def test_segmentation_returning_no_segments_falls_back_to_the_original_text(tmp_path):
    segments = [TranscriptSegment(text="Hello there.", is_final=True, language="en")]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    segmenter = FakeSegmenter(results={"Hello there.": []})
    pipeline = StagePipeline(
        make_stage_config(targets=[]), FakeTranscriber(segments), store, broadcaster, segmenter=segmenter
    )

    await pipeline.run()  # must not raise

    finals = [e for e in broadcaster.published if e.type == "caption.final"]
    assert [f.text for f in finals] == ["Hello there."]


async def test_segmentation_timeout_falls_back_to_the_original_text(tmp_path):
    """A segmenter that never resolves must not prevent run() from
    finishing, and must not lose the caption — proves the timeout bound
    around the segmentation call actually works, not just failure handling."""
    segments = [TranscriptSegment(text="Hello there.", is_final=True, language="en")]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    segmenter = FakeSegmenter(hang_for={"Hello there."})  # never released in this test
    pipeline = StagePipeline(
        make_stage_config(targets=[]),
        FakeTranscriber(segments),
        store,
        broadcaster,
        segmenter=segmenter,
        segmentation_timeout_seconds=0.05,
    )

    await asyncio.wait_for(pipeline.run(), timeout=2.0)

    assert pipeline.status == "stopped"
    finals = [e for e in broadcaster.published if e.type == "caption.final"]
    assert [f.text for f in finals] == ["Hello there."]


async def test_sub_segments_are_emitted_in_the_order_the_segmenter_returned_them(tmp_path):
    segments = [TranscriptSegment(text="A. B. C.", is_final=True, language="en")]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    segmenter = FakeSegmenter(results={"A. B. C.": ["A.", "B.", "C."]})
    pipeline = StagePipeline(
        make_stage_config(targets=[]), FakeTranscriber(segments), store, broadcaster, segmenter=segmenter
    )

    await pipeline.run()

    persisted = [json.loads(line) for line in (tmp_path / "main.jsonl").read_text().splitlines()]
    assert [e["text"] for e in persisted] == ["A.", "B.", "C."]
    assert [e["seg_id"] for e in persisted] == ["main-000001-1", "main-000001-2", "main-000001-3"]


async def test_current_audio_position_ms_is_none_before_any_timed_segment(tmp_path):
    store = JsonlEventStore(base_dir=tmp_path)
    pipeline = StagePipeline(make_stage_config(), FakeTranscriber([]), store, Broadcaster())

    assert pipeline.current_audio_position_ms is None


async def test_current_audio_position_ms_interpolates_forward_while_running(tmp_path):
    segments = [
        TranscriptSegment(text="partial", is_final=False, audio_elapsed_ms=1000.0, asr_latency_ms=50.0),
    ]
    store = JsonlEventStore(base_dir=tmp_path)
    pipeline = StagePipeline(make_stage_config(), FakeTranscriber(segments), store, Broadcaster())

    pipeline.status = "running"
    pipeline._handle_interim(segments[0])
    position_immediately_after = pipeline.current_audio_position_ms
    assert position_immediately_after is not None
    assert position_immediately_after >= 1000.0

    await asyncio.sleep(0.05)
    position_later = pipeline.current_audio_position_ms
    assert position_later > position_immediately_after  # interpolated forward


async def test_current_audio_position_ms_freezes_once_not_running(tmp_path):
    segments = [
        TranscriptSegment(text="final.", is_final=True, language="en", audio_elapsed_ms=1000.0, asr_latency_ms=50.0),
    ]
    store = JsonlEventStore(base_dir=tmp_path)
    pipeline = StagePipeline(make_stage_config(), FakeTranscriber(segments), store, Broadcaster())

    await pipeline.run()  # ends with status == "stopped"

    assert pipeline.status == "stopped"
    frozen = pipeline.current_audio_position_ms
    assert frozen == 1000.0
    await asyncio.sleep(0.05)
    assert pipeline.current_audio_position_ms == frozen  # no further extrapolation


async def test_current_audio_position_ms_caps_extrapolation_when_stale(tmp_path):
    from app.stage import MAX_AUDIO_POSITION_EXTRAPOLATION_MS

    store = JsonlEventStore(base_dir=tmp_path)
    pipeline = StagePipeline(make_stage_config(), FakeTranscriber([]), store, Broadcaster())
    pipeline.status = "running"
    pipeline._latest_audio_elapsed_ms = 1000.0
    pipeline._latest_audio_position_updated_at = time.monotonic() - 3600  # an hour stale

    position = pipeline.current_audio_position_ms

    assert position == 1000.0 + MAX_AUDIO_POSITION_EXTRAPOLATION_MS  # capped, not 3.6M ms later


async def test_interim_sentences_are_committed_progressively_and_final_only_adds_the_tail(tmp_path):
    texts = [
        "I", "I am", "I am testing", "I am testing CaptionMesh.",
        "I am testing CaptionMesh. This", "I am testing CaptionMesh. This is",
        "I am testing CaptionMesh. This is a demo.",
    ]
    segments = [TranscriptSegment(text=t, is_final=False, language="en") for t in texts]
    segments.append(TranscriptSegment(text="I am testing CaptionMesh. This is a demo.", is_final=True, language="en"))
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    translator = FakeTranslator()
    pipeline = StagePipeline(
        make_stage_config(targets=["es"]), FakeTranscriber(segments), store, broadcaster, translator
    )

    await pipeline.run()

    finals = [e for e in broadcaster.published if e.type == "caption.final"]
    assert [(f.seg_id, f.text) for f in finals] == [
        ("main-000001-1", "I am testing CaptionMesh."),
        ("main-000001-2", "This is a demo."),
    ]
    # Each sentence translated exactly once — never the whole paragraph, never twice.
    assert translator.calls == [
        ("I am testing CaptionMesh.", "en", "es"),
        ("This is a demo.", "en", "es"),
    ]
    # The live interim only ever shows the sentence still under construction.
    interims = [e.text for e in broadcaster.published if e.type == "caption.interim"]
    assert "This is" in interims
    assert not any(t.startswith("I am testing CaptionMesh. This") for t in interims)


async def test_committed_sentence_is_emitted_before_the_final_arrives(tmp_path):
    segments = [
        TranscriptSegment(text="First one. Second", is_final=False, language="en"),
    ]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    pipeline = StagePipeline(make_stage_config(targets=[]), FakeTranscriber(segments), store, broadcaster)

    await pipeline.run()  # no final ever arrives

    finals = [e.text for e in broadcaster.published if e.type == "caption.final"]
    assert finals == ["First one."]


async def test_a_multi_sentence_final_without_prior_commits_is_split_per_sentence(tmp_path):
    segments = [TranscriptSegment(text="Connection to the audience. It runs on FastAPI.", is_final=True, language="en")]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    translator = FakeTranslator()
    pipeline = StagePipeline(
        make_stage_config(targets=["es"]), FakeTranscriber(segments), store, broadcaster, translator
    )

    await pipeline.run()

    finals = [(e.seg_id, e.text) for e in broadcaster.published if e.type == "caption.final"]
    assert finals == [
        ("main-000001-1", "Connection to the audience."),
        ("main-000001-2", "It runs on FastAPI."),
    ]
    assert [c[0] for c in translator.calls] == ["Connection to the audience.", "It runs on FastAPI."]


async def test_gemini_resent_and_restarted_finals_are_not_duplicated(tmp_path):
    """Exact sequence observed from Gemini with short end-of-speech detection:
    a re-sent previous final, and a sentence restarted after a mid-way final."""
    segments = [
        TranscriptSegment(text="Contributors do not need to install anything.", is_final=True, language="en"),
        TranscriptSegment(text="The system is designed to scale from a single stage running on", is_final=True, language="en"),
        TranscriptSegment(text="Contributors do not need to install anything.", is_final=True, language="en"),
        TranscriptSegment(
            text="The system is designed to scale from a single stage running on one laptop.",
            is_final=True, language="en",
        ),
    ]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    translator = FakeTranslator()
    pipeline = StagePipeline(
        make_stage_config(targets=["es"]), FakeTranscriber(segments), store, broadcaster, translator
    )

    await pipeline.run()

    finals = [e.text for e in broadcaster.published if e.type == "caption.final"]
    assert finals == [
        "Contributors do not need to install anything.",
        "The system is designed to scale from a single stage running on",
        "one laptop.",
    ]
    assert len(translator.calls) == 3  # nothing translated twice


async def test_a_resent_variant_of_recent_captions_only_adds_its_new_part(tmp_path):
    """Observed: Gemini re-sent two already-published sentences as one
    variant with different casing/punctuation, followed by new speech."""
    segments = [
        TranscriptSegment(text="WebSocket connection to the audience.", is_final=True, language="en"),
        TranscriptSegment(text="CaptionMesh runs as a FastAPI application.", is_final=True, language="en"),
        TranscriptSegment(
            text="WebSocket connection to the audience caption mesh runs as a fast API application and uses Docker",
            is_final=True, language="en",
        ),
    ]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    pipeline = StagePipeline(make_stage_config(targets=[]), FakeTranscriber(segments), store, broadcaster)

    await pipeline.run()

    finals = [e.text for e in broadcaster.published if e.type == "caption.final"]
    assert finals == [
        "WebSocket connection to the audience.",
        "CaptionMesh runs as a FastAPI application.",
        "and uses Docker",
    ]


async def test_a_turn_that_restarts_without_a_final_publishes_its_unfinished_tail(tmp_path):
    """Observed: Gemini began a new turn without ever finalizing the previous
    one — its text must not silently vanish from the live line."""
    segments = [
        TranscriptSegment(text="Every audio source", is_final=False, language="en"),
        TranscriptSegment(text="Every audio source is normalized with FFmpeg", is_final=False, language="en"),
        TranscriptSegment(text="transcription pipeline", is_final=False, language="en"),
        TranscriptSegment(text="Transcription pipeline and live subtitles.", is_final=True, language="en"),
    ]
    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = RecordingBroadcaster()
    translator = FakeTranslator()
    pipeline = StagePipeline(
        make_stage_config(targets=["es"]), FakeTranscriber(segments), store, broadcaster, translator
    )

    await pipeline.run()

    finals = [(e.seg_id, e.text) for e in broadcaster.published if e.type == "caption.final"]
    assert finals == [
        ("main-000001", "Every audio source is normalized with FFmpeg"),
        ("main-000002", "Transcription pipeline and live subtitles."),
    ]
    assert len(translator.calls) == 2
