# CaptionMesh

CaptionMesh is an open-source real-time conference captioning and translation
system, built during the Nerdearla Vibeathon 2026. See `docs/spec.md` for the
full architecture and `CLAUDE.md` for development rules.

## Running the App (single stage)

The real `StagePipeline` runs inside the FastAPI process defined in `app/api.py`.
It wires together `FileAudioSource` (ffmpeg) → `GeminiTranscriber` → `caption.interim`
/ `caption.final` events, broadcast live to audience clients and, for finals only,
persisted as JSONL. Translation is not implemented yet — see `docs/spec.md` for
the full pipeline.

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

### Audience view

Open `http://localhost:8000/audience/main` in a browser (or any `stage_id`
from `config/stages.yaml`). The page connects to `/ws/audience/{stage_id}`, a
read-only WebSocket fed by an in-memory `Broadcaster` (`app/broadcast.py`).

- Interim captions update a single "live" line as they arrive.
- When a final event with the same `seg_id` arrives, it replaces the live
  line's content and the finalized text is pushed into a short on-page
  history (last 20 segments).
- The connection badge reflects WebSocket state; the stage badge polls
  `GET /health` every 5s.
- `Broadcaster` only fans out to currently-connected subscribers — it does
  not buffer events for clients that join late, and per-subscriber queues
  are bounded (oldest event dropped under backpressure) so a slow client
  cannot grow an unbounded backlog.

### Latency instrumentation

Every event carries an optional `timing` object (`{audio_elapsed_ms,
asr_latency_ms}`), `None` when a provider can't supply it:

- `audio_elapsed_ms` — elapsed *source-relative* audio content sent to
  Gemini so far. Not a wall-clock capture timestamp (the P0 source is a
  prerecorded file, not a live mic).
- `asr_latency_ms` — how far behind that audio timeline this transcript
  arrived: `(wall-clock elapsed since streaming started) - audio_elapsed_ms`.
  Computed in `GeminiTranscriber` from actual bytes sent per chunk, so it
  stays correct regardless of chunk size.

`GET /health` includes a `latency` object per stage with count/first/min/avg/max
for interim and final events separately (`app/stats.py`). The audience page
shows a small dev-only indicator (`LIVE · ASR 1.4s · delivery(same-host) 2ms`);
the delivery number assumes server and browser share a clock, which holds for
this local dev setup but not for a real remote audience client.

`FileAudioSource` accepts an optional `chunk_ms` (default unchanged at 100ms)
used only for the chunk-size experiment; it is not exposed through stage
config.

`FileAudioSource` paces sends with absolute wall-clock deadlines
(`pacing="deadline"`, the production default), not repeated fixed-duration
sleeps. Measured against the real Gemini Live API, the old fixed-sleep
approach (`pacing="naive"`, kept only for comparison/tests) accumulated
~18ms of drift per second of audio (~1.2s after 67s) because it never
accounted for time spent elsewhere in the send loop; deadline-based pacing
holds drift flat instead.
- `StagePipeline` has no dependency on FastAPI or WebSockets; it only calls
  `Broadcaster.publish(event)`, so the transport is fully swappable.

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
