import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

from app.providers import TranscriptSegment


class ReplayTranscriber:
    """TranscriptionProvider that replays a previously recorded transcript
    with its original timing.

    A real provider, not a placeholder: deterministic and quota-free, used
    to develop/test/demo downstream processing (including multiple
    concurrent stages) without multiplying live Gemini sessions.

    Ignores the `audio_chunks` argument entirely — replay reproduces a
    transcript timeline directly from its fixture, it does not consume
    audio. This keeps it interchangeable with any other TranscriptionProvider
    at the StagePipeline boundary.

    Fixture format: a JSON array of segments, each
        {"delay_ms": <int>, "text": <str>, "is_final": <bool>, "language": <str, optional>}
    `delay_ms` is the wait before emitting that segment, relative to the
    previous one (or to replay start, for the first).
    """

    def __init__(self, path: Path, realtime: bool = True) -> None:
        self._path = path
        self._realtime = realtime

    async def transcribe(self, audio_chunks) -> AsyncIterator[TranscriptSegment]:
        if not self._path.exists():
            raise FileNotFoundError(f"Replay fixture not found: {self._path}")

        fixture = json.loads(self._path.read_text())
        for entry in fixture:
            if self._realtime:
                delay_s = entry.get("delay_ms", 0) / 1000
                if delay_s > 0:
                    await asyncio.sleep(delay_s)
            yield TranscriptSegment(
                text=entry["text"],
                is_final=entry.get("is_final", False),
                language=entry.get("language"),
            )
