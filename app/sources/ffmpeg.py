import asyncio
import shutil
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
    """

    def __init__(self, path: Path, realtime: bool = True, chunk_ms: int = CHUNK_MS) -> None:
        self._path = path
        self._realtime = realtime
        self._chunk_ms = chunk_ms
        self._chunk_bytes = int(SAMPLE_RATE * SAMPLE_WIDTH_BYTES * CHANNELS * chunk_ms / 1000)

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
        try:
            while True:
                chunk = await process.stdout.read(self._chunk_bytes)
                if not chunk:
                    break
                yield chunk
                if self._realtime:
                    await asyncio.sleep(self._chunk_ms / 1000)
        finally:
            stderr = await process.stderr.read()
            returncode = await process.wait()
            if returncode != 0:
                raise RuntimeError(
                    f"ffmpeg exited with code {returncode}: {stderr.decode(errors='replace')}"
                )
