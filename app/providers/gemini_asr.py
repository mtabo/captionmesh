import asyncio
import logging
import time
from typing import AsyncIterator, Callable, Optional

from google import genai
from google.genai import types

from app.providers import TranscriptSegment
from app.sources.ffmpeg import SAMPLE_RATE, SAMPLE_WIDTH_BYTES

DEFAULT_MODEL = "gemini-3.5-transcribe-live"
DEFAULT_MODE = "SMART"
RECEIVE_GRACE_SECONDS = 15
BYTES_PER_MS = SAMPLE_RATE * SAMPLE_WIDTH_BYTES / 1000
DRIFT_SAMPLE_INTERVAL_MS = 1000

# Gemini Live Transcribe sessions have a documented ~10 minute limit; the
# caller (StageSupervisor, from config `gemini.session_rotation_seconds`)
# is expected to pass a threshold comfortably under that. This module-level
# default is only used if a caller constructs GeminiTranscriber directly
# without one.
DEFAULT_SESSION_ROTATION_SECONDS = 540.0

# Bounded backoff for reconnecting after an unexpected session error (not
# used for planned rotation, which reconnects immediately). After these are
# exhausted, the failure is raised and the stage ends with status "error".
RECONNECT_BACKOFF_SCHEDULE = (1.0, 2.0, 4.0, 8.0)

logger = logging.getLogger(__name__)


class GeminiTranscriber:
    """TranscriptionProvider backed by Gemini Live Transcription.

    Transparently rotates the underlying session before Gemini's documented
    ~10 minute session limit, and reconnects (bounded backoff) if a session
    ends unexpectedly — the caller only ever sees one continuous
    `transcribe()` stream; audio already pulled from `audio_chunks` is never
    replayed, and subsequent chunks simply continue into the new session.
    """

    def __init__(
        self,
        api_key: str,
        language: str = "auto",
        model: str = DEFAULT_MODEL,
        mode: str = DEFAULT_MODE,
        on_send_sample: Optional[Callable[[float, float], None]] = None,
        session_rotation_seconds: float = DEFAULT_SESSION_ROTATION_SECONDS,
    ) -> None:
        """`on_send_sample`, if given, is called roughly every second of audio
        content sent with (audio_elapsed_ms, wall_elapsed_ms) — diagnostic
        only, unused by default and not wired into production call sites.
        """
        self._api_key = api_key
        self._language = language
        self._model = model
        self._mode = mode
        self._on_send_sample = on_send_sample
        self._session_rotation_seconds = session_rotation_seconds
        self._stream_started_at: Optional[float] = None
        self._audio_elapsed_ms: float = 0.0
        self._last_sampled_audio_ms: float = 0.0

    async def transcribe(self, audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[TranscriptSegment]:
        chunk_iter = audio_chunks.__aiter__()
        session_index = 0
        consecutive_failures = 0

        while True:
            session_index += 1
            outcome = {"reason": "error"}
            try:
                async for segment in self._run_session(chunk_iter, session_index, outcome):
                    yield segment
            except Exception:
                logger.exception("[asr] session #%d ended with an error", session_index)
                outcome["reason"] = "error"

            if outcome["reason"] == "exhausted":
                logger.info("[asr] audio source exhausted after session #%d; ending", session_index)
                return
            elif outcome["reason"] == "rotate":
                logger.info("[asr] session #%d rotation threshold reached; reconnecting", session_index)
                consecutive_failures = 0
                continue
            else:
                consecutive_failures += 1
                if consecutive_failures > len(RECONNECT_BACKOFF_SCHEDULE):
                    logger.error(
                        "[asr] giving up after %d failed reconnect attempts", consecutive_failures - 1
                    )
                    raise RuntimeError(
                        f"Gemini ASR session failed after {consecutive_failures - 1} reconnect attempts"
                    )
                delay = RECONNECT_BACKOFF_SCHEDULE[consecutive_failures - 1]
                logger.warning(
                    "[asr] reconnect attempt %d/%d in %.0fs",
                    consecutive_failures,
                    len(RECONNECT_BACKOFF_SCHEDULE),
                    delay,
                )
                await asyncio.sleep(delay)
                continue

    def _connect(self):
        """Isolated so tests can substitute a fake session without the real SDK."""
        client = genai.Client(api_key=self._api_key)
        language_codes = [] if self._language == "auto" else [self._language]
        config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            input_audio_transcription=types.AudioTranscriptionConfig(
                language_codes=language_codes,
                mode=self._mode,
            ),
        )
        return client.aio.live.connect(model=self._model, config=config)

    async def _run_session(
        self, chunk_iter, session_index: int, outcome: dict
    ) -> AsyncIterator[TranscriptSegment]:
        if self._stream_started_at is None:
            self._stream_started_at = time.monotonic()
        session_started_at = time.monotonic()
        is_first_segment = True

        async with self._connect() as session:
            logger.info("[asr] session #%d started", session_index)
            sender = asyncio.create_task(
                self._send_audio_for_session(session, chunk_iter, session_started_at, outcome)
            )
            receive_iter = session.receive().__aiter__()
            try:
                while True:
                    try:
                        # Bounded regardless of sender state: if the sender is
                        # still active but the connection has died silently
                        # (observed in production as a server-initiated close
                        # the SDK never surfaced as an exception — the socket
                        # sat in CLOSE_WAIT), this must not hang forever. On
                        # timeout we break out below, the existing `finally`
                        # cancels the sender, and the existing outer
                        # reconnect/backoff loop in transcribe() takes over.
                        message = await asyncio.wait_for(
                            receive_iter.__anext__(), timeout=RECEIVE_GRACE_SECONDS
                        )
                    except (StopAsyncIteration, asyncio.TimeoutError):
                        break

                    content = message.server_content
                    if content is None:
                        continue

                    audio_elapsed_ms, asr_latency_ms = self._current_timing()

                    interim = content.interim_input_transcription
                    if interim is not None and interim.text:
                        yield TranscriptSegment(
                            text=interim.text,
                            is_final=False,
                            language=interim.language_code,
                            audio_elapsed_ms=audio_elapsed_ms,
                            asr_latency_ms=asr_latency_ms,
                            session_boundary=is_first_segment,
                        )
                        is_first_segment = False

                    final = content.input_transcription
                    if final is not None and final.text:
                        # `finished` is often None rather than True for a completed
                        # segment; only `False` marks it as still-interim content
                        # streamed through this field (observed in the spike).
                        if final.finished is False:
                            yield TranscriptSegment(
                                text=final.text,
                                is_final=False,
                                language=final.language_code,
                                audio_elapsed_ms=audio_elapsed_ms,
                                asr_latency_ms=asr_latency_ms,
                                session_boundary=is_first_segment,
                            )
                        else:
                            yield TranscriptSegment(
                                text=final.text,
                                is_final=True,
                                language=final.language_code,
                                audio_elapsed_ms=audio_elapsed_ms,
                                asr_latency_ms=asr_latency_ms,
                                session_boundary=is_first_segment,
                            )
                        is_first_segment = False

                    if content.turn_complete:
                        break
            finally:
                if not sender.done():
                    sender.cancel()
                try:
                    await sender
                except asyncio.CancelledError:
                    pass
                logger.info("[asr] session #%d closed", session_index)

    def _current_timing(self) -> tuple[Optional[float], Optional[float]]:
        """Snapshot of (audio_elapsed_ms, asr_latency_ms) at the moment of a receive.

        Both are cumulative across the whole transcribe() call (all
        sessions), not reset per session — they describe the stage's
        overall audio timeline, not any one session's.

        asr_latency_ms is how far the wall clock has moved past the amount of
        audio content already sent — i.e. how far behind real time this
        transcript event arrived. None before any audio has been sent.
        """
        if self._stream_started_at is None:
            return None, None
        wall_elapsed_ms = (time.monotonic() - self._stream_started_at) * 1000
        return self._audio_elapsed_ms, wall_elapsed_ms - self._audio_elapsed_ms

    async def _send_audio_for_session(
        self, session, chunk_iter, session_started_at: float, outcome: dict
    ) -> None:
        while True:
            if time.monotonic() - session_started_at >= self._session_rotation_seconds:
                outcome["reason"] = "rotate"
                break
            try:
                chunk = await chunk_iter.__anext__()
            except StopAsyncIteration:
                outcome["reason"] = "exhausted"
                break

            await session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={SAMPLE_RATE}")
            )
            self._audio_elapsed_ms += len(chunk) / BYTES_PER_MS

            if (
                self._on_send_sample is not None
                and self._audio_elapsed_ms - self._last_sampled_audio_ms >= DRIFT_SAMPLE_INTERVAL_MS
            ):
                self._last_sampled_audio_ms = self._audio_elapsed_ms
                wall_elapsed_ms = (time.monotonic() - self._stream_started_at) * 1000
                self._on_send_sample(self._audio_elapsed_ms, wall_elapsed_ms)

        await session.send_realtime_input(audio_stream_end=True)
