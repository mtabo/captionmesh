import logging
import time
from pathlib import Path
from typing import Optional

from app.broadcast import Broadcaster
from app.config import StageConfig
from app.events import CaptionFinalEvent, CaptionInterimEvent, EventTiming
from app.providers import TranscriptionProvider, TranscriptSegment
from app.sources.ffmpeg import FileAudioSource
from app.stats import StageLatencyStats
from app.store import JsonlEventStore

logger = logging.getLogger(__name__)


class StagePipeline:
    """Coordinates Source -> Transcription -> Events -> Broadcast/Store for one stage.

    Interim and finalized transcripts of the same logical segment share a
    stable `seg_id` so audience clients can correlate them. Only finalized
    events are persisted; interim events are broadcast only.

    A stage failure is caught and reflected in `status` rather than raised,
    so it can run as an isolated task without taking down other stages.
    """

    def __init__(
        self,
        config: StageConfig,
        transcriber: TranscriptionProvider,
        store: JsonlEventStore,
        broadcaster: Broadcaster,
    ) -> None:
        self._config = config
        self._transcriber = transcriber
        self._store = store
        self._broadcaster = broadcaster
        self._seg_counter = 0
        self._current_seg_id: str | None = None
        self.status = "created"
        self.stats = StageLatencyStats()

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
        event = CaptionFinalEvent(
            stage_id=self._config.id,
            seg_id=self._open_seg_id(),
            text=segment.text,
            language=self._resolve_language(segment),
            timing=timing,
        )
        self._store.append(self._config.id, event)
        self._publish(event)
        if timing is not None:
            self.stats.record_final(timing.asr_latency_ms)
        logger.info("[%s][FINAL] %s", self._config.id, segment.text)
        self._current_seg_id = None
