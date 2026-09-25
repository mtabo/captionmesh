import asyncio
from typing import AsyncIterator

from google import genai
from google.genai import types

from app.providers import TranscriptSegment
from app.sources.ffmpeg import SAMPLE_RATE

DEFAULT_MODEL = "gemini-3.5-transcribe-live"
DEFAULT_MODE = "SMART"
RECEIVE_GRACE_SECONDS = 15


class GeminiTranscriber:
    """TranscriptionProvider backed by Gemini Live Transcription."""

    def __init__(
        self,
        api_key: str,
        language: str = "auto",
        model: str = DEFAULT_MODEL,
        mode: str = DEFAULT_MODE,
    ) -> None:
        self._api_key = api_key
        self._language = language
        self._model = model
        self._mode = mode

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

                    interim = content.interim_input_transcription
                    if interim is not None and interim.text:
                        yield TranscriptSegment(
                            text=interim.text, is_final=False, language=interim.language_code
                        )

                    final = content.input_transcription
                    if final is not None and final.text:
                        # `finished` is often None rather than True for a completed
                        # segment; only `False` marks it as still-interim content
                        # streamed through this field (observed in the spike).
                        if final.finished is False:
                            yield TranscriptSegment(
                                text=final.text, is_final=False, language=final.language_code
                            )
                        else:
                            yield TranscriptSegment(
                                text=final.text, is_final=True, language=final.language_code
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

    @staticmethod
    async def _send_audio(session, audio_chunks: AsyncIterator[bytes]) -> None:
        async for chunk in audio_chunks:
            await session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={SAMPLE_RATE}")
            )
        await session.send_realtime_input(audio_stream_end=True)
