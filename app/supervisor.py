import asyncio
from pathlib import Path
from typing import Optional

from app.broadcast import Broadcaster
from app.config import ConferenceConfig, FileSourceConfig, ReplaySourceConfig, StageConfig
from app.providers.gemini_asr import GeminiTranscriber
from app.providers.gemini_translate import GeminiTranslator
from app.providers import TranslationProvider
from app.providers.replay import ReplayTranscriber
from app.sources.ffmpeg import FileAudioSource
from app.stage import StagePipeline
from app.store import JsonlEventStore

# Translation candidate under evaluation (see benchmark notes): far cheaper
# Free Tier quota (per-minute) than the gemini-3.5-flash default (per-day,
# easily exhausted by a single stage), and measured faster/more reliable in
# a real multi-segment run. Explicit override via GeminiTranslator's existing
# `model` param — the class default is untouched.
TRANSLATOR_MODEL = "gemini-3.5-flash-lite"


class StageSupervisor:
    """Creates and supervises one independent StagePipeline per configured stage.

    Each pipeline runs as its own asyncio.Task with entirely independent
    state (source, transcriber, seg_id sequence, pending translations,
    stats, lifecycle status) — a failure or slow operation in one stage
    cannot block another. `broadcaster` and `store` are shared, but both are
    already keyed by `stage_id` internally, so sharing them does not create
    cross-stage coupling.
    """

    def __init__(
        self,
        conference: ConferenceConfig,
        api_key: str,
        broadcaster: Optional[Broadcaster] = None,
        store: Optional[JsonlEventStore] = None,
        translator: Optional[TranslationProvider] = None,
    ) -> None:
        self._conference = conference
        self._api_key = api_key
        self.broadcaster = broadcaster or Broadcaster()
        self.store = store or JsonlEventStore()
        self._translator = translator or GeminiTranslator(api_key=api_key, model=TRANSLATOR_MODEL)
        self.pipelines: dict[str, StagePipeline] = {}
        self._tasks: list[asyncio.Task] = []

    def _build_pipeline(self, stage_config: StageConfig) -> StagePipeline:
        source = stage_config.source
        if isinstance(source, FileSourceConfig):
            transcriber = GeminiTranscriber(
                api_key=self._api_key,
                language=stage_config.language,
                session_rotation_seconds=self._conference.gemini.session_rotation_seconds,
            )
            audio_source = FileAudioSource(Path(source.path))
        elif isinstance(source, ReplaySourceConfig):
            transcriber = ReplayTranscriber(Path(source.path))
            audio_source = None
        else:
            raise ValueError(f"Unsupported source type for stage {stage_config.id!r}: {source.type!r}")

        return StagePipeline(
            stage_config,
            transcriber,
            self.store,
            self.broadcaster,
            self._translator,
            audio_source=audio_source,
        )

    def start_all(self) -> None:
        for stage_config in self._conference.stages:
            pipeline = self._build_pipeline(stage_config)
            self.pipelines[stage_config.id] = pipeline
            self._tasks.append(asyncio.create_task(pipeline.run()))

    async def stop_all(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
