import logging
from pathlib import Path

from app.config import StageConfig
from app.events import CaptionFinalEvent
from app.providers import TranscriptionProvider
from app.sources.ffmpeg import FileAudioSource
from app.store import JsonlEventStore

logger = logging.getLogger(__name__)


class StagePipeline:
    """Coordinates Source -> Transcription -> Events -> Store for one stage.

    A stage failure is caught and reflected in `status` rather than raised,
    so it can run as an isolated task without taking down other stages.
    """

    def __init__(
        self,
        config: StageConfig,
        transcriber: TranscriptionProvider,
        store: JsonlEventStore,
    ) -> None:
        self._config = config
        self._transcriber = transcriber
        self._store = store
        self._seg_counter = 0
        self.status = "created"

    async def run(self) -> None:
        self.status = "running"
        try:
            source = FileAudioSource(Path(self._config.source.path))
            async for segment in self._transcriber.transcribe(source.stream()):
                if segment.is_final:
                    self._handle_final(segment)
                else:
                    logger.info("[%s][INTERIM] %s", self._config.id, segment.text)
            self.status = "stopped"
        except Exception:
            self.status = "error"
            logger.exception("Stage %s failed", self._config.id)

    def _handle_final(self, segment) -> None:
        language = segment.language or (
            self._config.language if self._config.language != "auto" else "und"
        )
        self._seg_counter += 1
        seg_id = f"{self._config.id}-{self._seg_counter:06d}"
        event = CaptionFinalEvent(
            stage_id=self._config.id,
            seg_id=seg_id,
            text=segment.text,
            language=language,
        )
        self._store.append(self._config.id, event)
        logger.info("[%s][FINAL] %s", self._config.id, segment.text)
