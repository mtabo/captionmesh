"""Generate a synthetic test speech fixture for the Gemini transcription spike.

Not part of the CaptionMesh runtime. Produces data/audio/test-en.wav, a local
development fixture that is not committed to Git (see .gitignore).
"""

import subprocess
import sys
from pathlib import Path

from gtts import gTTS

SCRIPT = """
Welcome to CaptionMesh, an open source real-time captioning and translation
platform built during the Nerdearla Vibeathon.

Today we are testing real-time transcription using Gemini, streamed over a
WebSocket connection to the audience.

CaptionMesh runs as a FastAPI application and uses Docker for local
development, so contributors do not need to install dependencies on their
own machines.

The system is designed to scale from a single conference stage running on
one laptop, all the way up to a Kubernetes deployment handling many stages
at once.

Every audio source is normalized with ffmpeg before it reaches the
transcription pipeline, and finalized transcripts are translated and
broadcast to audience clients as live subtitles.

This is only a technical spike to validate that live transcription works
end to end before we build the full CaptionMesh architecture.
""".strip()

AUDIO_DIR = Path("data/audio")
MP3_PATH = AUDIO_DIR / "test-en.mp3"
WAV_PATH = AUDIO_DIR / "test-en.wav"


def main() -> int:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating speech with gTTS...")
    try:
        tts = gTTS(text=SCRIPT, lang="en")
        tts.save(str(MP3_PATH))
    except Exception as exc:  # network failure, etc.
        print(f"ERROR: gTTS generation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Converting {MP3_PATH} -> {WAV_PATH} (PCM 16-bit / 16kHz / mono)...")
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(MP3_PATH),
            "-ac", "1",
            "-ar", "16000",
            "-sample_fmt", "s16",
            str(WAV_PATH),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("ERROR: ffmpeg conversion failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return 1

    MP3_PATH.unlink(missing_ok=True)

    print(f"Wrote {WAV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
