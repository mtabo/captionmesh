import asyncio
import logging
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

logger = logging.getLogger(__name__)

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

# How long restart_stage() waits for the previous session to actually stop
# after cancelling it, before giving up on waiting and starting the new
# session anyway. See restart_stage's own comment for why this can't be
# unbounded.
RESTART_CANCEL_GRACE_SECONDS = 10


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
        vtt_output_dir: Optional[Path] = None,
        restart_cancel_grace_seconds: float = RESTART_CANCEL_GRACE_SECONDS,
    ) -> None:
        self._conference = conference
        self._api_key = api_key
        # Injectable so tests can exercise the "previous session didn't stop
        # in time" branch without a real 10s wait; production always uses
        # the module default above.
        self._restart_cancel_grace_seconds = restart_cancel_grace_seconds
        self.broadcaster = broadcaster or Broadcaster()
        self.store = store or JsonlEventStore()
        # None (the default) means no stage writes a per-session VTT archive
        # — same explicit-opt-in reasoning as StagePipeline's own param this
        # is forwarded to. Only app.api's lifespan (the real app) passes
        # VTT_OUTPUT_DIR; tests constructing a StageSupervisor directly stay
        # filesystem-safe without needing to know about VTT at all.
        self._vtt_output_dir = vtt_output_dir
        self._translator = translator or GeminiTranslator(
            api_key=api_key, model=TRANSLATOR_MODEL, glossary=conference.gemini.glossary
        )
        # No default segmenter: a per-final Gemini segmentation call sits on
        # the caption's critical path and was measured timing out (10s) under
        # load. Finals are emitted as Gemini produced them unless one is
        # explicitly injected.
        self._segmenter = segmenter
        self.pipelines: dict[str, StagePipeline] = {}
        self.stage_configs: dict[str, StageConfig] = {s.id: s for s in conference.stages}
        self._tasks: list[asyncio.Task] = []
        # Mirrors `_tasks`, keyed by stage_id — lets restart_stage() cancel
        # and replace exactly one stage's task without touching `_tasks`
        # (existing tests iterate that list directly) or any other stage.
        self._tasks_by_stage: dict[str, asyncio.Task] = {}

    def _build_pipeline(self, stage_config: StageConfig) -> StagePipeline:
        source = stage_config.source
        if isinstance(source, FileSourceConfig):
            transcriber = GeminiTranscriber(
                api_key=self._api_key,
                language=stage_config.language,
                mode=TRANSCRIBE_MODE,
                session_rotation_seconds=self._conference.gemini.session_rotation_seconds,
                custom_vocabulary=self._conference.gemini.glossary,
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
            vtt_output_dir=self._vtt_output_dir,
        )

    def start_all(self) -> None:
        stagger = self._conference.gemini.session_start_stagger_seconds
        for index, stage_config in enumerate(self._conference.stages):
            self._start_stage(stage_config, delay_seconds=index * stagger)

    def _start_stage(self, stage_config: StageConfig, delay_seconds: float = 0.0) -> StagePipeline:
        pipeline = self._build_pipeline(stage_config)
        self.pipelines[stage_config.id] = pipeline
        task = asyncio.create_task(self._run_after(pipeline, delay_seconds))
        self._tasks.append(task)
        self._tasks_by_stage[stage_config.id] = task
        return pipeline

    @staticmethod
    async def _run_after(pipeline: StagePipeline, delay_seconds: float) -> None:
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        await pipeline.run()

    async def restart_stage(self, stage_id: str) -> StagePipeline:
        """Cleanly stops the stage's current pipeline (if still running) and
        starts a fresh one from the same, unmodified config — same
        transcriber/translator/segmenter/audio source construction as
        start_all(), via the existing `_build_pipeline`. History (JSONL,
        past VTT snapshots) is untouched: the new StagePipeline appends to
        the same `store`, and gets its own timestamped VTT on completion,
        same as any other session (see StagePipeline._write_session_vtt).

        `broadcaster`/`store` are shared across the whole supervisor and
        keyed by `stage_id`, not by StagePipeline instance, so an already
        `/ws/audience/{stage_id}`-connected client keeps working through
        the restart with no reconnect needed — it just starts receiving
        the new pipeline's events.
        """
        stage_config = self.stage_configs.get(stage_id)
        if stage_config is None:
            raise KeyError(stage_id)

        old_task = self._tasks_by_stage.get(stage_id)
        if old_task is not None and not old_task.done():
            old_task.cancel()
            # Bounded, not indefinite: cancelling a task stuck inside a live
            # Gemini send() can itself hang past this grace window (measured
            # for real, elsewhere in this project — the write can be stuck
            # on kernel-level TCP flow control, which task cancellation
            # alone doesn't unblock). RESTART_CANCEL_GRACE_SECONDS covers
            # the normal case (its own segmentation/translation shutdown
            # grace windows total 10s); past that, proceed and start the
            # new session anyway rather than leave this HTTP request (and
            # the demo) hanging — the old task keeps winding down on its
            # own in the background.
            # Shielded: asyncio.wait_for's own timeout handling cancels
            # *and awaits* whatever it's given, so waiting on old_task
            # directly would still block forever if old_task keeps
            # swallowing CancelledError (measured for real: it can catch it
            # and re-enter another blocking await). Shielding means the
            # timeout only ever cancels the shield wrapper, which returns
            # immediately regardless of old_task's own state — old_task
            # itself keeps winding down in the background, cancelled or not.
            try:
                await asyncio.wait_for(asyncio.shield(old_task), timeout=self._restart_cancel_grace_seconds)
            except asyncio.TimeoutError:
                logger.warning(
                    "[%s] previous session did not stop within %ss of cancellation; "
                    "starting the new one anyway",
                    stage_id, self._restart_cancel_grace_seconds,
                )
            except asyncio.CancelledError:
                pass  # expected: this is the cancellation we just requested
            except Exception:
                logger.exception("[%s] previous session raised while stopping for restart", stage_id)

        return self._start_stage(stage_config)

    async def stop_all(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
