"""CaptionMesh — Gemini Live Transcription spike.

Proves the core assumption end-to-end:

    audio file -> ffmpeg -> PCM 16kHz mono -> Gemini Live Transcription
        -> interim/finalized transcript -> console output

This is a standalone runtime smoke test, independent of the future
StagePipeline. It requires network access and a valid GEMINI_API_KEY.
"""

import asyncio
import os
import shutil
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

MODEL = "gemini-3.5-transcribe-live"
TRANSCRIPTION_MODE = "SMART"
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2
CHUNK_MS = 100
CHUNK_BYTES = int(SAMPLE_RATE * SAMPLE_WIDTH_BYTES * CHANNELS * CHUNK_MS / 1000)
RECEIVE_GRACE_SECONDS = 15


class SpikeFailure(Exception):
    """Raised for any condition that should end the spike with RESULT: FAIL."""


def require_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise SpikeFailure("GEMINI_API_KEY is not set. Add it to .env.")
    return api_key


def require_audio_file(path: Path) -> Path:
    if not path.exists():
        raise SpikeFailure(
            f"Audio file not found: {path}. "
            "Run `docker compose run --rm gen-audio` to generate it."
        )
    return path


def require_ffmpeg() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise SpikeFailure("ffmpeg is not available on PATH.")
    return ffmpeg_path


async def pcm_chunks(ffmpeg_path: str, audio_path: Path):
    """Decode audio_path to raw PCM (s16le/16kHz/mono) and yield fixed-size chunks."""
    process = await asyncio.create_subprocess_exec(
        ffmpeg_path,
        "-i", str(audio_path),
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
            chunk = await process.stdout.read(CHUNK_BYTES)
            if not chunk:
                break
            yield chunk
    finally:
        stderr = await process.stderr.read()
        returncode = await process.wait()
        if returncode != 0:
            raise SpikeFailure(
                f"ffmpeg exited with code {returncode}: {stderr.decode(errors='replace')}"
            )


class SpikeState:
    def __init__(self) -> None:
        self.audio_started_at: float | None = None
        self.first_interim_at: float | None = None
        self.first_final_at: float | None = None
        self.interim_seen = False
        self.final_seen = False
        self.final_text_parts: list[str] = []
        self.last_message_at: float | None = None


async def send_audio(session, audio_path: Path, ffmpeg_path: str, state: SpikeState) -> None:
    state.audio_started_at = time.monotonic()
    async for chunk in pcm_chunks(ffmpeg_path, audio_path):
        await session.send_realtime_input(
            audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={SAMPLE_RATE}")
        )
        await asyncio.sleep(CHUNK_MS / 1000)
    await session.send_realtime_input(audio_stream_end=True)


async def receive_transcripts(session, state: SpikeState) -> None:
    async for message in session.receive():
        state.last_message_at = time.monotonic()
        content = message.server_content
        if content is None:
            continue

        interim = content.interim_input_transcription
        if interim is not None and interim.text:
            if not state.interim_seen:
                state.interim_seen = True
                state.first_interim_at = time.monotonic()
            print(f"[INTERIM] {interim.text}")

        final = content.input_transcription
        if final is not None and final.text:
            if final.finished is False:
                # Some server builds stream interim updates through this field too.
                if not state.interim_seen:
                    state.interim_seen = True
                    state.first_interim_at = time.monotonic()
                print(f"[INTERIM] {final.text}")
            else:
                if not state.final_seen:
                    state.final_seen = True
                    state.first_final_at = time.monotonic()
                print(f"[FINAL] {final.text}")
                state.final_text_parts.append(final.text)

        if content.turn_complete:
            break


async def run_spike(audio_path: Path) -> SpikeState:
    api_key = require_api_key()
    require_audio_file(audio_path)
    ffmpeg_path = require_ffmpeg()

    print("CaptionMesh — Gemini Live Transcription Spike\n")
    print(f"Audio: {audio_path}")
    print(f"Model: {MODEL}")
    print(f"Mode: {TRANSCRIPTION_MODE}")
    print("Language: auto\n")

    client = genai.Client(api_key=api_key)
    config = types.LiveConnectConfig(
        response_modalities=["TEXT"],
        input_audio_transcription=types.AudioTranscriptionConfig(
            language_codes=[],
            mode=TRANSCRIPTION_MODE,
        ),
    )

    state = SpikeState()

    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            print("Connected to Gemini Live.\n")

            sender = asyncio.create_task(send_audio(session, audio_path, ffmpeg_path, state))
            receiver = asyncio.create_task(receive_transcripts(session, state))

            await sender
            try:
                await asyncio.wait_for(receiver, timeout=RECEIVE_GRACE_SECONDS)
            except asyncio.TimeoutError:
                receiver.cancel()
    except SpikeFailure:
        raise
    except Exception as exc:
        raise SpikeFailure(f"Gemini session error: {exc}") from exc

    print("\nSession completed.\n")
    return state


def report(state: SpikeState) -> bool:
    final_transcript = " ".join(state.final_text_parts).strip()

    print("Final transcript:")
    print("-----------------")
    print(final_transcript or "(empty)")
    print("-----------------\n")

    if state.audio_started_at and state.first_interim_at:
        print(f"time_to_first_interim: {state.first_interim_at - state.audio_started_at:.2f}s")
    if state.audio_started_at and state.first_final_at:
        print(f"time_to_first_final:   {state.first_final_at - state.audio_started_at:.2f}s")
    print()

    passed = state.interim_seen and state.final_seen and bool(final_transcript)
    return passed


def main() -> int:
    audio_path = Path(os.environ.get("SPIKE_AUDIO_PATH", "data/audio/test-en.wav"))

    try:
        state = asyncio.run(run_spike(audio_path))
    except SpikeFailure as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print("\nRESULT: FAIL")
        return 1

    passed = report(state)
    print("RESULT: PASS" if passed else "RESULT: FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
