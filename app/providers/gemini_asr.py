import asyncio
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


class GeminiTranscriber:
    """TranscriptionProvider backed by Gemini Live Transcription."""

    def __init__(
        self,
        api_key: str,
        language: str = "auto",
        model: str = DEFAULT_MODEL,
        mode: str = DEFAULT_MODE,
        on_send_sample: Optional[Callable[[float, float], None]] = None,
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
        self._stream_started_at: Optional[float] = None
        self._audio_elapsed_ms: float = 0.0
        self._last_sampled_audio_ms: float = 0.0

    async def transcribe(self, audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[TranscriptSegment]:
        client = genai.Client(api_key=self._api_key)
        language_codes = [] if self._language == "auto" else [self._language]
        config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            input_audio_transcription=types.AudioTranscriptionConfig(
                language_codes=language_codes,
                mode=self._mode,
            ),
        )

        async with client.aio.live.connect(model=self._model, config=config) as session:
            sender = asyncio.create_task(self._send_audio(session, audio_chunks))
            receive_iter = session.receive().__aiter__()
            try:
                while True:
                    try:
                        if sender.done():
                            message = await asyncio.wait_for(
                                receive_iter.__anext__(), timeout=RECEIVE_GRACE_SECONDS
                            )
                        else:
                            message = await receive_iter.__anext__()
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
                        )

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
                            )
                        else:
                            yield TranscriptSegment(
                                text=final.text,
                                is_final=True,
                                language=final.language_code,
                                audio_elapsed_ms=audio_elapsed_ms,
                                asr_latency_ms=asr_latency_ms,
                            )

                    if content.turn_complete:
                        break
            finally:
                if not sender.done():
                    sender.cancel()
                try:
                    await sender
                except asyncio.CancelledError:
                    pass

    def _current_timing(self) -> tuple[Optional[float], Optional[float]]:
        """Snapshot of (audio_elapsed_ms, asr_latency_ms) at the moment of a receive.

        asr_latency_ms is how far the wall clock has moved past the amount of
        audio content already sent — i.e. how far behind real time this
        transcript event arrived. None before any audio has been sent.
        """
        if self._stream_started_at is None:
            return None, None
        wall_elapsed_ms = (time.monotonic() - self._stream_started_at) * 1000
        return self._audio_elapsed_ms, wall_elapsed_ms - self._audio_elapsed_ms

    async def _send_audio(self, session, audio_chunks: AsyncIterator[bytes]) -> None:
        self._stream_started_at = time.monotonic()
        async for chunk in audio_chunks:
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
