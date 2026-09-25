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


class TranscriptionProvider(Protocol):
    def transcribe(self, audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[TranscriptSegment]:
        ...


class TranslationProvider(Protocol):
    async def translate(self, text: str, source_language: str, target_language: str) -> str:
        ...
