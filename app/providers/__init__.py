from typing import AsyncIterator, Optional, Protocol

from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    text: str
    is_final: bool
    language: Optional[str] = None


class TranscriptionProvider(Protocol):
    def transcribe(self, audio_chunks: AsyncIterator[bytes]) -> AsyncIterator[TranscriptSegment]:
        ...
