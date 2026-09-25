import asyncio
import logging
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from app.broadcast import Broadcaster
from app.config import StageConfig
from app.events import CaptionFinalEvent, CaptionInterimEvent, CaptionTranslationEvent, EventTiming
from app.providers import SegmentationProvider, TranscriptionProvider, TranscriptSegment, TranslationProvider
from app.sources import AudioSource
from app.stats import StageLatencyStats
from app.store import JsonlEventStore
from app.vtt import build_vtt, write_timestamped_vtt_file

logger = logging.getLogger(__name__)


# Bounded grace window, mirroring the existing RECEIVE_GRACE_SECONDS idiom in
# GeminiTranscriber: once the main transcription loop ends, give pending
# translation tasks a short window to finish rather than dropping them
# instantly, but never block shutdown indefinitely for a slow/hung call.
TRANSLATION_SHUTDOWN_GRACE_SECONDS = 5

# Same idiom, for the segmentation worker (see _shutdown_segmentation_worker).
SEGMENTATION_SHUTDOWN_GRACE_SECONDS = 5

# Bounded wait on a single segmentation call. On timeout (or any other
# failure), _segment_text falls back to the original, unsegmented text —
# the caption is never lost because segmentation is slow or broken.
SEGMENTATION_TIMEOUT_SECONDS = 10

# current_audio_position_ms interpolates forward from the last known
# position using elapsed wall-clock time. If nothing has updated that
# position in a while (e.g. a stalled ASR session — observed for real
# during RC validation), stop trusting the interpolation past this many ms
# of staleness rather than reporting an ever-growing, meaningless position.
MAX_AUDIO_POSITION_EXTRAPOLATION_MS = 5000

# Splits on the sentence punctuation the ASR itself emits (followed by
# whitespace, so "3.50" is never split). A piece only counts as a complete
# sentence once another piece follows it in the same turn's text.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_BREAK.split(text.strip()) if s]


# How many leading letters/digits of the previous interim a new interim must
# share to count as the same turn (Gemini revises the end of an interim, not
# its beginning).
TURN_RESTART_ANCHOR_CHARS = 12


def _alnum(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _drop_alnum_prefix(text: str, count: int) -> str:
    """`text` without its first `count` letters/digits (and the
    punctuation/whitespace right after them)."""
    seen = 0
    for index, ch in enumerate(text):
        if seen == count:
            return text[index:].lstrip(" .,;:!?-")
        if ch.isalnum():
            seen += 1
    return ""


async def _empty_audio_stream():
    """No real audio (e.g. a replay-driven stage, which generates its own
    timeline). An async generator that yields nothing."""
    return
    yield  # pragma: no cover - unreachable; makes this a generator function


class StagePipeline:
    """Coordinates Source -> Transcription -> Events -> Broadcast/Store for one stage.

    Interim and finalized transcripts of the same logical segment share a
    stable `seg_id` so audience clients can correlate them. Only finalized
    events are persisted; interim events are broadcast only.

    Finalized segments are first split into subtitle-sized sub-segments by
    a `SegmentationProvider` (see `_run_segmentation_worker`), each getting
    its own derived `seg_id` (`main-000005-1`, `main-000005-2`, ...). Each
    sub-segment is then translated into each configured target language
    (skipping the source language itself) as independent, tracked background
    tasks — neither segmentation nor translation is ever awaited from the
    main transcription loop, so a slow or failing call on either side cannot
    stall reading further transcription messages. Translation events share
    their sub-segment's `seg_id` and are both persisted and broadcast, same
    as finals.

    A stage failure is caught and reflected in `status` rather than raised,
    so it can run as an isolated task without taking down other stages.
    """

    def __init__(
        self,
        config: StageConfig,
        transcriber: TranscriptionProvider,
        store: JsonlEventStore,
        broadcaster: Broadcaster,
        translator: Optional[TranslationProvider] = None,
        segmenter: Optional[SegmentationProvider] = None,
        translation_shutdown_grace_seconds: float = TRANSLATION_SHUTDOWN_GRACE_SECONDS,
        segmentation_shutdown_grace_seconds: float = SEGMENTATION_SHUTDOWN_GRACE_SECONDS,
        segmentation_timeout_seconds: float = SEGMENTATION_TIMEOUT_SECONDS,
        audio_source: Optional[AudioSource] = None,
        vtt_output_dir: Optional[Path] = None,
        vtt_timestamp_fn: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._config = config
        self._transcriber = transcriber
        self._store = store
        self._broadcaster = broadcaster
        self._translator = translator
        self._segmenter = segmenter
        self._translation_shutdown_grace_seconds = translation_shutdown_grace_seconds
        self._segmentation_shutdown_grace_seconds = segmentation_shutdown_grace_seconds
        self._segmentation_timeout_seconds = segmentation_timeout_seconds
        self._audio_source = audio_source
        # Where/when _write_session_vtt (see run()) writes its timestamped
        # snapshot. None (the default) disables it entirely — an explicit
        # opt-in, not "write to data/vtt/ unless told otherwise": dozens of
        # existing tests construct a StagePipeline directly and run it to
        # completion without caring about VTT at all, and defaulting to the
        # real VTT_OUTPUT_DIR meant every one of them silently wrote files
        # into the real data/vtt/ on every test run. StageSupervisor (the
        # only production call site) passes VTT_OUTPUT_DIR explicitly.
        self._vtt_output_dir = vtt_output_dir
        self._vtt_timestamp_fn = vtt_timestamp_fn
        self._seg_counter = 0
        self._current_seg_id: str | None = None
        self.status = "created"
        self.stats = StageLatencyStats()
        self._pending_translation_tasks: set[asyncio.Task] = set()
        # FIFO queue of raw finals awaiting segmentation, processed strictly
        # in order by _run_segmentation_worker — see that method for why
        # this is a separate task rather than an inline await.
        self._segmentation_queue: asyncio.Queue = asyncio.Queue()
        self._segmentation_task: Optional[asyncio.Task] = None
        # Best-effort "where is the source audio right now" — updated from
        # every interim/final (see _record_audio_position), read by
        # current_audio_position_ms for audience-player sync (/health).
        self._latest_audio_elapsed_ms: Optional[float] = None
        self._latest_audio_position_updated_at: Optional[float] = None
        # Sentences of the current turn (seg_id) already published as
        # finals from interims — see _handle_interim.
        self._committed_count = 0
        # Last few published captions (normalized), for _emit_caption's
        # duplicate guard.
        self._recent_captions: deque[str] = deque(maxlen=5)
        # Cumulative text of the latest interim in the current turn, to
        # detect a turn restart that never got a final.
        self._last_interim_text: Optional[str] = None

    @property
    def pending_translation_count(self) -> int:
        return len(self._pending_translation_tasks)

    @property
    def current_audio_position_ms(self) -> Optional[float]:
        """Best-effort estimate of the source audio position currently being
        processed, for audience-player sync — not a precise measurement.

        While `running`, interpolates forward from the last interim/final's
        `audio_elapsed_ms` using elapsed wall-clock time since it arrived
        (capped at MAX_AUDIO_POSITION_EXTRAPOLATION_MS, so a stalled session
        doesn't produce an ever-growing, meaningless position). Once the
        stage is no longer running, freezes at the last known value instead
        of extrapolating past when audio was actually still flowing.
        """
        if self._latest_audio_elapsed_ms is None:
            return None
        if self.status != "running" or self._latest_audio_position_updated_at is None:
            return self._latest_audio_elapsed_ms
        elapsed_since_update_ms = (time.monotonic() - self._latest_audio_position_updated_at) * 1000
        elapsed_since_update_ms = min(elapsed_since_update_ms, MAX_AUDIO_POSITION_EXTRAPOLATION_MS)
        return self._latest_audio_elapsed_ms + elapsed_since_update_ms

    def _record_audio_position(self, segment: TranscriptSegment) -> None:
        if segment.audio_elapsed_ms is not None:
            self._latest_audio_elapsed_ms = segment.audio_elapsed_ms
            self._latest_audio_position_updated_at = time.monotonic()

    async def run(self) -> None:
        self.status = "running"
        self._segmentation_task = asyncio.create_task(self._run_segmentation_worker())
        try:
            audio_chunks = (
                self._audio_source.stream() if self._audio_source is not None else _empty_audio_stream()
            )
            async for segment in self._transcriber.transcribe(audio_chunks):
                if segment.is_final:
                    self._handle_final(segment)
                else:
                    self._handle_interim(segment)
            self.status = "stopped"
        except Exception:
            self.status = "error"
            logger.exception("Stage %s failed", self._config.id)
        finally:
            # Segmentation feeds translation scheduling, so drain it first.
            await self._shutdown_segmentation_worker()
            await self._shutdown_pending_translations()
            if self.status == "stopped":
                # Only a session that finished cleanly gets an archived VTT
                # snapshot — not one still running, and not one that ended
                # in "error" (its event stream may be incomplete/mid-final).
                self._write_session_vtt()

    def _write_session_vtt(self) -> None:
        """Auto-generates a timestamped WebVTT archive of this completed
        session, derived from the same persisted event store the on-demand
        `captions.vtt` endpoint reads (`build_vtt` — no second source of
        truth). A write failure (e.g. disk full) is logged, not raised —
        it must never make an otherwise-successful session look failed."""
        if self._vtt_output_dir is None:
            return
        try:
            events = self._store.read_events(self._config.id)
            vtt_content = build_vtt(events)
            path = write_timestamped_vtt_file(
                self._config.id,
                vtt_content,
                when=self._vtt_timestamp_fn(),
                base_dir=self._vtt_output_dir,
            )
            logger.info("[%s] wrote session VTT snapshot: %s", self._config.id, path)
        except Exception:
            logger.exception("[%s] failed to write session VTT snapshot", self._config.id)

    def _resolve_language(self, segment: TranscriptSegment) -> str:
        return segment.language or (
            self._config.language if self._config.language != "auto" else "und"
        )

    def _open_seg_id(self) -> str:
        if self._current_seg_id is None:
            self._seg_counter += 1
            self._current_seg_id = f"{self._config.id}-{self._seg_counter:06d}"
            self._committed_count = 0
        return self._current_seg_id

    @staticmethod
    def _build_timing(segment: TranscriptSegment) -> Optional[EventTiming]:
        if segment.audio_elapsed_ms is None or segment.asr_latency_ms is None:
            return None
        return EventTiming(
            audio_elapsed_ms=segment.audio_elapsed_ms,
            asr_latency_ms=segment.asr_latency_ms,
        )

    def _publish(self, event) -> None:
        t0 = time.monotonic()
        self._broadcaster.publish(event)
        broadcast_latency_ms = (time.monotonic() - t0) * 1000
        logger.debug("[%s] broadcast latency: %.3fms", self._config.id, broadcast_latency_ms)

    def _handle_interim(self, segment: TranscriptSegment) -> None:
        self._record_audio_position(segment)
        if segment.session_boundary:
            # A new provider session started (e.g. ASR rotation/reconnect).
            # Discard any still-open seg_id from the previous session rather
            # than letting unrelated new content inherit it — the old
            # interim simply disappears, per spec; never fabricate a final.
            self._current_seg_id = None
            self._last_interim_text = None
        timing = self._build_timing(segment)
        language = self._resolve_language(segment)
        if timing is not None:
            self.stats.record_interim(timing.asr_latency_ms)

        # Gemini was observed starting a new turn without ever sending a
        # final for the previous one: the cumulative interim text restarts
        # instead of growing. Publish the previous turn's unfinished tail
        # rather than letting it vanish from the live line.
        previous = self._last_interim_text
        if (
            previous
            and self._current_seg_id is not None
            and not _alnum(segment.text).startswith(_alnum(previous)[:TURN_RESTART_ANCHOR_CHARS])
        ):
            leftover = _split_sentences(previous)[self._committed_count:]
            if leftover:
                single = self._committed_count == 0
                self._committed_count += 1
                sub_seg_id = self._current_seg_id if single else f"{self._current_seg_id}-{self._committed_count}"
                self._emit_caption(sub_seg_id, " ".join(leftover), language, timing)
            self._current_seg_id = None
        self._last_interim_text = segment.text

        seg_id = self._open_seg_id()

        # The interim text is cumulative for the whole turn. Publish each
        # sentence as soon as the next one has started, and keep only the
        # still-in-progress tail as the live interim.
        sentences = _split_sentences(segment.text)
        for index in range(self._committed_count, len(sentences) - 1):
            self._committed_count += 1
            self._emit_caption(
                f"{seg_id}-{self._committed_count}", sentences[index], language, timing
            )
        tail = " ".join(sentences[self._committed_count:])
        if not tail:
            return
        event = CaptionInterimEvent(
            stage_id=self._config.id,
            seg_id=seg_id,
            text=tail,
            language=language,
            timing=timing,
        )
        self._publish(event)
        logger.info("[%s][INTERIM] %s", self._config.id, tail)

    def _emit_caption(
        self, seg_id: str, text: str, language: str, timing: Optional[EventTiming]
    ) -> None:
        """Publish + persist one caption and schedule its translations.
        Synchronous and non-blocking (translation runs as its own task).

        Gemini (with short end-of-speech detection) was observed re-sending a
        previous final verbatim, and restarting a sentence it had already
        finalized mid-way. Skip exact repeats of a recent caption, and if a
        caption extends the last one, publish only the new remainder — so a
        sentence is never shown or translated twice."""
        # Compared on letters/digits only: re-sends vary in casing, spacing
        # and punctuation ("CaptionMesh runs" vs "caption mesh runs").
        key = _alnum(text)
        if not key or key in self._recent_captions:
            return
        recent = list(self._recent_captions)
        overlap = 0
        for k in range(1, len(recent) + 1):
            joined = "".join(recent[-k:])
            if key.startswith(joined):
                overlap = len(joined)
        if overlap:
            text = _drop_alnum_prefix(text, overlap)
            key = _alnum(text)
            if not key:
                return
        self._recent_captions.append(key)
        event = CaptionFinalEvent(
            stage_id=self._config.id, seg_id=seg_id, text=text, language=language, timing=timing
        )
        self._store.append(self._config.id, event)
        self._publish(event)
        logger.info("[%s][FINAL] %s", self._config.id, text)
        self._schedule_translations(seg_id, text, language)

    def _handle_final(self, segment: TranscriptSegment) -> None:
        self._record_audio_position(segment)
        if segment.session_boundary:
            self._current_seg_id = None
        timing = self._build_timing(segment)
        seg_id = self._open_seg_id()
        source_language = self._resolve_language(segment)
        self._current_seg_id = None
        self._last_interim_text = None

        if self._committed_count > 0 or self._segmenter is None:
            # Publish only the sentences of this turn not yet committed from
            # interims (never re-emit or re-translate a committed one). If
            # Gemini revised earlier text, the committed versions stand —
            # accepted for live captions. A single-sentence turn keeps the
            # plain seg_id.
            sentences = _split_sentences(segment.text) or [segment.text]
            single = self._committed_count == 0 and len(sentences) == 1
            for sentence in sentences[self._committed_count:]:
                self._committed_count += 1
                sub_seg_id = seg_id if single else f"{seg_id}-{self._committed_count}"
                self._emit_caption(sub_seg_id, sentence, source_language, timing)
            if timing is not None:
                self.stats.record_final(timing.asr_latency_ms)
            return

        # Segmentation (a Gemini call) happens off the main loop, in
        # _run_segmentation_worker — this method must stay non-blocking so a
        # slow/failing segmentation call can never stall reading further
        # transcription messages, same reasoning as translation below.
        # CaptionFinalEvent emission is therefore deferred to the worker too.
        self._segmentation_queue.put_nowait((seg_id, segment.text, source_language, timing))

    async def _run_segmentation_worker(self) -> None:
        """Processes queued finals strictly in order, one at a time, so
        sub-segments and finals across different original finals never
        appear out of order — unlike translation, which is safe to run
        fully concurrently per (segment, target) pair."""
        while True:
            item = await self._segmentation_queue.get()
            if item is None:  # shutdown sentinel — see _shutdown_segmentation_worker
                return
            seg_id, text, source_language, timing = item
            await self._segment_and_emit_final(seg_id, text, source_language, timing)

    async def _segment_and_emit_final(
        self, seg_id: str, text: str, source_language: str, timing: Optional[EventTiming]
    ) -> None:
        sub_texts = await self._segment_text(text)
        # Only introduce the derived "-N" seg_id scheme when a final was
        # actually split — the common single-segment case (no segmenter
        # configured, or Gemini decided not to split) keeps the plain
        # seg_id unchanged, preserving the existing format everywhere else
        # (interim events, JSONL consumers, VTT) relies on.
        split = len(sub_texts) > 1

        # The original final's timing describes one point in time (when
        # Gemini finalized the *whole* block) — there is no measured
        # boundary between sub-segments. Rather than fabricate per-segment
        # timestamps we don't have, every sub-segment gets the same, real
        # timing: honest about the limitation, and (unlike giving only the
        # last sub-segment real timing) never drops earlier sub-segments
        # from the VTT export, which only includes finals with timing.
        for index, sub_text in enumerate(sub_texts, start=1):
            sub_seg_id = f"{seg_id}-{index}" if split else seg_id
            event = CaptionFinalEvent(
                stage_id=self._config.id,
                seg_id=sub_seg_id,
                text=sub_text,
                language=source_language,
                timing=timing,
            )
            self._store.append(self._config.id, event)
            self._publish(event)
            logger.info("[%s][FINAL] %s", self._config.id, sub_text)
            self._schedule_translations(sub_seg_id, sub_text, source_language)

        # Recorded once per original ASR final, not per sub-segment — this
        # stat characterizes ASR behavior, not caption-emission granularity.
        if timing is not None:
            self.stats.record_final(timing.asr_latency_ms)

    async def _segment_text(self, text: str) -> list[str]:
        if self._segmenter is None:
            return [text]
        try:
            sub_texts = await asyncio.wait_for(
                self._segmenter.segment(text), timeout=self._segmentation_timeout_seconds
            )
        except Exception:
            logger.exception(
                "[%s] segmentation failed for final; falling back to a single segment",
                self._config.id,
            )
            return [text]
        if not sub_texts:
            logger.warning(
                "[%s] segmentation returned no segments; falling back to a single segment",
                self._config.id,
            )
            return [text]
        return sub_texts

    async def _shutdown_segmentation_worker(self) -> None:
        if self._segmentation_task is None:
            return
        self._segmentation_queue.put_nowait(None)  # sentinel: no more items coming
        try:
            await asyncio.wait_for(
                self._segmentation_task, timeout=self._segmentation_shutdown_grace_seconds
            )
        except asyncio.TimeoutError:
            logger.info(
                "[%s] cancelling segmentation worker still running after %ss shutdown grace",
                self._config.id,
                self._segmentation_shutdown_grace_seconds,
            )
            self._segmentation_task.cancel()
            try:
                await self._segmentation_task
            except asyncio.CancelledError:
                pass

    def _schedule_translations(self, seg_id: str, text: str, source_language: str) -> None:
        """Fire off one background task per eligible target. Never awaited here —
        this is what keeps a slow/failing translation from blocking transcription."""
        if self._translator is None:
            return
        for target_language in self._config.targets:
            if target_language == source_language:
                continue
            task = asyncio.create_task(
                self._translate_one(seg_id, text, source_language, target_language)
            )
            self._pending_translation_tasks.add(task)
            task.add_done_callback(self._pending_translation_tasks.discard)

    async def _translate_one(
        self, seg_id: str, text: str, source_language: str, target_language: str
    ) -> None:
        try:
            t0 = time.monotonic()
            translated_text = await self._translator.translate(text, source_language, target_language)
            translation_latency_ms = (time.monotonic() - t0) * 1000
        except Exception:
            logger.exception(
                "[%s] translation to %s failed for %s", self._config.id, target_language, seg_id
            )
            return

        event = CaptionTranslationEvent(
            stage_id=self._config.id,
            seg_id=seg_id,
            text=translated_text,
            language=target_language,
            source_language=source_language,
            translation_latency_ms=translation_latency_ms,
        )
        self._store.append(self._config.id, event)
        self._publish(event)
        self.stats.record_translation(translation_latency_ms)
        logger.info("[%s][TRANSLATION:%s] %s", self._config.id, target_language, translated_text)

    async def _shutdown_pending_translations(self) -> None:
        if not self._pending_translation_tasks:
            return
        pending = set(self._pending_translation_tasks)
        _done, still_pending = await asyncio.wait(
            pending, timeout=self._translation_shutdown_grace_seconds
        )
        if still_pending:
            logger.info(
                "[%s] cancelling %d translation task(s) still running after %ss shutdown grace",
                self._config.id,
                len(still_pending),
                self._translation_shutdown_grace_seconds,
            )
            for task in still_pending:
                task.cancel()
            await asyncio.gather(*still_pending, return_exceptions=True)
