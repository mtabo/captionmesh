from typing import AsyncIterator, Optional, Protocol

from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    text: str
    is_final: bool
    language: Optional[str] = None
    # Honest, provider-measured timing (see app.events.EventTiming). None when
    # a provider (e.g. a test stub or a future ReplayTranscriber) cannot
    # supply it.
    audio_elapsed_ms: Optional[float] = None
    asr_latency_ms: Optional[float] = None
    # True on the first segment of a new underlying provider session (e.g.
    # after Gemini ASR session rotation/reconnect). Lets StagePipeline close
    # any dangling open seg_id from the previous session without fabricating
    # a final caption for it. False for every other segment and for
    # providers that have no session concept (e.g. ReplayTranscriber).
    session_boundary: bool = False


class TranscriptionProvider(Protocol):
    def transcribe(self, audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[TranscriptSegment]:
        ...


class TranslationProvider(Protocol):
    async def translate(self, text: str, source_language: str, target_language: str) -> str:
        ...


class SegmentationProvider(Protocol):
    async def segment(self, text: str) -> list[str]:
        ...
