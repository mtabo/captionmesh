# CaptionMesh

CaptionMesh is an open-source real-time conference captioning and translation
system, built during the Nerdearla Vibeathon 2026. See `docs/spec.md` for the
full architecture and `CLAUDE.md` for development rules.

## Running the App (single stage)

The real `StagePipeline` runs inside the FastAPI process defined in `app/api.py`.
It wires together `FileAudioSource` (ffmpeg) → `GeminiTranscriber` → finalized
`caption.final` events, persisted as JSONL. Translation and the audience
UI/broadcast are not implemented yet — see `docs/spec.md` for the full pipeline.

Stages are configured in `config/stages.yaml`:

```yaml
conference:
  name: CaptionMesh Demo

stages:
  - id: main
    name: Main Stage
    language: auto
    targets:
      - es
    source:
      type: file
      path: data/audio/test-en.wav
```

Generate the fixture audio first (see below), then start the app:

```bash
docker compose up app
```

- `GET /health` reports each configured stage's lifecycle status
  (`created` → `running` → `stopped`/`error`).
- Finalized transcript segments are appended to
  `data/stages/<stage_id>.jsonl` as they complete, each with a stable
  `seg_id` (e.g. `main-000001`).
- If Gemini does not return a language code under `language: auto`, the
  event's `language` falls back to `"und"` rather than falsely claiming
  `"auto"` — the app never assumes the provider supplies a reliable code.
- A stage's transcription failure is caught inside `StagePipeline.run()` and
  reflected as `status: "error"` rather than crashing the process, so other
  stages (once multi-stage support lands) stay unaffected.

Session rotation/reconnect for the ~minutes-long Gemini Live session limit is
not implemented yet — a long-running stage will simply stop once the session
ends.

## Gemini Live Transcription Spike

A standalone technical spike that proves the core assumption: an audio file
can be streamed through ffmpeg into Gemini Live Transcription and produce
real-time interim and finalized captions. This is **not** yet the
`StagePipeline` — it is an isolated runtime smoke test at `app/spike_transcription.py`.

### Prerequisites

- Docker and Docker Compose
- A Gemini API key with access to `gemini-3.5-transcribe-live`

### Configuration

Create a `.env` file in the project root (never committed):

```text
GEMINI_API_KEY=your-key-here
```

### Generate the test audio fixture

The spike uses a short synthetic speech fixture (not committed to Git,
excluded via `.gitignore`) rather than copyrighted conference audio:

```bash
docker compose run --rm gen-audio
```

This writes `data/audio/test-en.wav` (PCM 16-bit / 16 kHz / mono), generated
with gTTS and normalized with ffmpeg.

### Run the spike

```bash
docker compose run --rm spike
```

- **Model:** `gemini-3.5-transcribe-live`
- **Mode:** `SMART` (disfluency removal, formatting cleanup)
- **Language:** `auto` (automatic language detection, no hardcoded language)
- **Audio format:** PCM 16-bit / 16 kHz / mono / little-endian

### Expected output

```text
CaptionMesh — Gemini Live Transcription Spike

Audio: data/audio/test-en.wav
Model: gemini-3.5-transcribe-live
Mode: SMART
Language: auto

Connected to Gemini Live.

[INTERIM] Welcome to CaptionMesh...
[FINAL] Welcome to CaptionMesh, an open source real-time captioning...
...

Session completed.

Final transcript:
-----------------
...
-----------------

time_to_first_interim: 1.45s
time_to_first_final:   32.00s

RESULT: PASS
```

### Known limitations

- Gemini Live Transcription sessions are limited to a finite continuous
  streaming duration (on the order of minutes). Session rotation/reconnect
  is not implemented in this spike — it belongs in the production
  `GeminiTranscriber` provider.
- `SMART` mode finalizes segments only once it has enough context, so
  `time_to_first_final` is noticeably higher than `time_to_first_interim`.
  This is expected buffering behavior, not a bug.
- This is a technical spike, not the final `StagePipeline`. It is not wired
  into translation, broadcast, or persistence.

### Run tests

```bash
docker compose run --rm --entrypoint bash spike -c "python -m pytest tests/ -v"
```
