import logging
from pathlib import Path

from app.broadcast import Broadcaster
from app.config import StageConfig
from app.events import CaptionFinalEvent, CaptionInterimEvent
from app.providers import TranscriptionProvider
from app.sources.ffmpeg import FileAudioSource
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

    def _resolve_language(self, segment) -> str:
        return segment.language or (
            self._config.language if self._config.language != "auto" else "und"
        )

    def _open_seg_id(self) -> str:
        if self._current_seg_id is None:
            self._seg_counter += 1
            self._current_seg_id = f"{self._config.id}-{self._seg_counter:06d}"
        return self._current_seg_id

    def _handle_interim(self, segment) -> None:
        event = CaptionInterimEvent(
            stage_id=self._config.id,
            seg_id=self._open_seg_id(),
            text=segment.text,
            language=self._resolve_language(segment),
        )
        self._broadcaster.publish(event)
        logger.info("[%s][INTERIM] %s", self._config.id, segment.text)

    def _handle_final(self, segment) -> None:
        event = CaptionFinalEvent(
            stage_id=self._config.id,
            seg_id=self._open_seg_id(),
            text=segment.text,
            language=self._resolve_language(segment),
        )
        self._store.append(self._config.id, event)
        self._broadcaster.publish(event)
        logger.info("[%s][FINAL] %s", self._config.id, segment.text)
        self._current_seg_id = None
