import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from app.broadcast import Broadcaster
from app.config import StageConfig
from app.events import CaptionFinalEvent, CaptionInterimEvent, CaptionTranslationEvent, EventTiming
from app.providers import TranscriptionProvider, TranscriptSegment, TranslationProvider
from app.sources.ffmpeg import FileAudioSource
from app.stats import StageLatencyStats
from app.store import JsonlEventStore

logger = logging.getLogger(__name__)

# Bounded grace window, mirroring the existing RECEIVE_GRACE_SECONDS idiom in
# GeminiTranscriber: once the main transcription loop ends, give pending
# translation tasks a short window to finish rather than dropping them
# instantly, but never block shutdown indefinitely for a slow/hung call.
TRANSLATION_SHUTDOWN_GRACE_SECONDS = 5


class StagePipeline:
    """Coordinates Source -> Transcription -> Events -> Broadcast/Store for one stage.

    Interim and finalized transcripts of the same logical segment share a
    stable `seg_id` so audience clients can correlate them. Only finalized
    events are persisted; interim events are broadcast only.

    Finalized segments are translated into each configured target language
    (skipping the source language itself) as independent, tracked background
    tasks — `_handle_final` never awaits a translation call, so a slow or
    failing translation cannot stall reading further transcription messages.
    Translation events share the source final's `seg_id` and are both
    persisted and broadcast, same as finals.

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
        translation_shutdown_grace_seconds: float = TRANSLATION_SHUTDOWN_GRACE_SECONDS,
    ) -> None:
        self._config = config
        self._transcriber = transcriber
        self._store = store
        self._broadcaster = broadcaster
        self._translator = translator
        self._translation_shutdown_grace_seconds = translation_shutdown_grace_seconds
        self._seg_counter = 0
        self._current_seg_id: str | None = None
        self.status = "created"
        self.stats = StageLatencyStats()
        self._pending_translation_tasks: set[asyncio.Task] = set()

    @property
    def pending_translation_count(self) -> int:
        return len(self._pending_translation_tasks)

    async def run(self) -> None:
        self.status = "running"
        try:
            source = FileAudioSource(Path(self._config.source.path))
            async for segment in self._transcriber.transcribe(source.stream()):
                if segment.is_final:
                    self._handle_final(segment)
                else:
                    self._handle_interim(segment)
            self.status = "stopped"
        except Exception:
            self.status = "error"
            logger.exception("Stage %s failed", self._config.id)
        finally:
            await self._shutdown_pending_translations()

    def _resolve_language(self, segment: TranscriptSegment) -> str:
        return segment.language or (
            self._config.language if self._config.language != "auto" else "und"
        )

    def _open_seg_id(self) -> str:
        if self._current_seg_id is None:
            self._seg_counter += 1
            self._current_seg_id = f"{self._config.id}-{self._seg_counter:06d}"
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
        timing = self._build_timing(segment)
        event = CaptionInterimEvent(
            stage_id=self._config.id,
            seg_id=self._open_seg_id(),
            text=segment.text,
            language=self._resolve_language(segment),
            timing=timing,
        )
        self._publish(event)
        if timing is not None:
            self.stats.record_interim(timing.asr_latency_ms)
        logger.info("[%s][INTERIM] %s", self._config.id, segment.text)

    def _handle_final(self, segment: TranscriptSegment) -> None:
        timing = self._build_timing(segment)
        seg_id = self._open_seg_id()
        source_language = self._resolve_language(segment)
        event = CaptionFinalEvent(
            stage_id=self._config.id,
            seg_id=seg_id,
            text=segment.text,
            language=source_language,
            timing=timing,
        )
        self._store.append(self._config.id, event)
        self._publish(event)
        if timing is not None:
            self.stats.record_final(timing.asr_latency_ms)
        logger.info("[%s][FINAL] %s", self._config.id, segment.text)
        self._current_seg_id = None

        self._schedule_translations(seg_id, segment.text, source_language)

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
