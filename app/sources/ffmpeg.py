import asyncio
import shutil
import time
from pathlib import Path
from typing import AsyncIterator

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
CHUNK_MS = 100
CHUNK_BYTES = int(SAMPLE_RATE * SAMPLE_WIDTH_BYTES * CHANNELS * CHUNK_MS / 1000)


class FileAudioSource:
    """Decodes a local audio file into canonical PCM chunks, paced at real time.

    `chunk_ms` is configurable (used for the chunk-size latency experiment)
    but defaults to the production value; it is not exposed via stage config.

    `pacing` defaults to "deadline" (production): each sleep targets an
    absolute wall-clock deadline anchored to stream start
    (`stream_started_at + chunk_index * chunk_ms`), computed fresh from the
    current clock each time. Any per-iteration overhead elsewhere in the
    pipeline (downstream await time, read latency) is absorbed into a
    shorter next sleep instead of accumulating. Measured against the real
    Gemini Live API, the previous "naive" pacing (fixed `sleep(chunk_ms)`
    after every chunk, blind to how much wall-clock time already elapsed)
    accumulated ~18ms of drift per second of audio (~1.2s after 67s);
    "deadline" pacing kept drift flat. "naive" is kept only for comparison/
    tests, not used by any production call site.
    """

    def __init__(
        self,
        path: Path,
        realtime: bool = True,
        chunk_ms: int = CHUNK_MS,
        pacing: str = "deadline",
    ) -> None:
        self._path = path
        self._realtime = realtime
        self._chunk_ms = chunk_ms
        self._chunk_bytes = int(SAMPLE_RATE * SAMPLE_WIDTH_BYTES * CHANNELS * chunk_ms / 1000)
        self._pacing = pacing

    @staticmethod
    def _pacing_delay_seconds(
        pacing: str,
        stream_started_at: float,
        chunk_index: int,
        chunk_ms: int,
        now: float,
    ) -> float:
        """Seconds to sleep before the next read, given `now` (monotonic clock).

        "deadline": sleep only until the absolute schedule position for
        `chunk_index` (clamped to 0 if already behind) — self-correcting,
        does not accumulate drift.
        "naive": always sleep the full `chunk_ms`, regardless of `now` — any
        overhead elsewhere in the loop compounds every iteration.
        """
        if pacing == "deadline":
            target = stream_started_at + chunk_index * (chunk_ms / 1000)
            return max(0.0, target - now)
        return chunk_ms / 1000

    async def stream(self) -> AsyncIterator[bytes]:
        if not self._path.exists():
            raise FileNotFoundError(f"Audio file not found: {self._path}")

        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError("ffmpeg is not available on PATH.")

        process = await asyncio.create_subprocess_exec(
            ffmpeg_path,
            "-i", str(self._path),
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ac", str(CHANNELS),
            "-ar", str(SAMPLE_RATE),
            "-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert process.stdout is not None
        stream_started_at = time.monotonic()
        chunk_index = 0
        try:
            while True:
                chunk = await process.stdout.read(self._chunk_bytes)
                if not chunk:
                    break
                yield chunk
                chunk_index += 1
                if self._realtime:
                    delay = self._pacing_delay_seconds(
                        self._pacing, stream_started_at, chunk_index, self._chunk_ms, time.monotonic()
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
        finally:
            stderr = await process.stderr.read()
            returncode = await process.wait()
            if returncode != 0:
                raise RuntimeError(
                    f"ffmpeg exited with code {returncode}: {stderr.decode(errors='replace')}"
                )
