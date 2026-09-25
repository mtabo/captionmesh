import asyncio
from pathlib import Path
from typing import Optional

from app.broadcast import Broadcaster
from app.config import ConferenceConfig, FileSourceConfig, ReplaySourceConfig, StageConfig
from app.providers.gemini_asr import GeminiTranscriber
from app.providers.gemini_translate import GeminiTranslator
from app.providers import SegmentationProvider, TranslationProvider
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

# Gemini Live transcription mode. Measured back-to-back on the same demo
# audio: SMART (the class default) buffered heavily and emitted a single
# ~160-word paragraph final ~47s after the audio ended; VERBATIM produced
# interims every ~0.5s and sentence-sized finals in real time. Live captions
# need the latter; the trade-off is that disfluencies are no longer removed.
TRANSCRIBE_MODE = "VERBATIM"


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
        segmenter: Optional[SegmentationProvider] = None,
    ) -> None:
        self._conference = conference
        self._api_key = api_key
        self.broadcaster = broadcaster or Broadcaster()
        self.store = store or JsonlEventStore()
        self._translator = translator or GeminiTranslator(api_key=api_key, model=TRANSLATOR_MODEL)
        # No default segmenter: a per-final Gemini segmentation call sits on
        # the caption's critical path and was measured timing out (10s) under
        # load. Finals are emitted as Gemini produced them unless one is
        # explicitly injected.
        self._segmenter = segmenter
        self.pipelines: dict[str, StagePipeline] = {}
        self.stage_configs: dict[str, StageConfig] = {s.id: s for s in conference.stages}
        self._tasks: list[asyncio.Task] = []

    def _build_pipeline(self, stage_config: StageConfig) -> StagePipeline:
        source = stage_config.source
        if isinstance(source, FileSourceConfig):
            transcriber = GeminiTranscriber(
                api_key=self._api_key,
                language=stage_config.language,
                mode=TRANSCRIBE_MODE,
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
            segmenter=self._segmenter,
            audio_source=audio_source,
        )

    def start_all(self) -> None:
        stagger = self._conference.gemini.session_start_stagger_seconds
        for index, stage_config in enumerate(self._conference.stages):
            pipeline = self._build_pipeline(stage_config)
            self.pipelines[stage_config.id] = pipeline
            self._tasks.append(asyncio.create_task(self._run_after(pipeline, index * stagger)))

    @staticmethod
    async def _run_after(pipeline: StagePipeline, delay_seconds: float) -> None:
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        await pipeline.run()

    async def stop_all(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
