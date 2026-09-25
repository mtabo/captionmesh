# CaptionMesh

CaptionMesh is an open-source real-time conference captioning and translation
system, built during the Nerdearla Vibeathon 2026. See `docs/spec.md` for the
full architecture and `CLAUDE.md` for development rules.

## WebVTT Export

`GET /api/stages/{stage_id}/captions.vtt` (`app/vtt.py`) builds a WebVTT
document from a stage's persisted finalized captions —
`supervisor.store.read_events(stage_id)` (new method on the existing
`JsonlEventStore`, not a new storage abstraction) → `build_vtt(events)`.
404 for an unconfigured `stage_id`; 200 with a valid, empty (`WEBVTT\n`)
document for a configured stage with no finals yet.

- Only `caption.final` events produce cues — interim is never persisted, so
  it never reaches this path.
- When a `caption.translation` exists for a `seg_id`, its text is preferred
  over the original (MVP choice: an EN→ES stage's VTT shows Spanish).
  Falls back to the original final's text when no translation exists.
- **Cue timestamps always come from the original final's `timing`, never
  from the translation** — confirmed by inspection: `StagePipeline._translate_one`
  never sets `timing` on a `CaptionTranslationEvent`, so it is always `None`.
  A translation can substitute a cue's text but never its timing.
- A cue's END is its final's `timing.audio_elapsed_ms`; its START is the
  previous cue's END (0 for the first) — the closest boundary this event
  model actually has, since only finals are persisted (no true per-utterance
  start is recorded). Documented approximation, not measured data.
- **A final with no `timing` is skipped, never given a fabricated
  timestamp.** This means stages sourced from `ReplayTranscriber` (which
  does not supply timing) currently export a valid but *empty* VTT even
  when they have finals — confirmed directly against a real run. Only
  `file`-sourced (real Gemini) stages produce populated VTT today.

## Multi-Stage Support

`StageSupervisor` (`app/supervisor.py`) creates and runs one independent
`StagePipeline` per configured stage, each as its own `asyncio.Task`. Every
pipeline has fully independent state — source, transcriber, `seg_id`
sequence, pending translation tasks, stats, lifecycle status — a slow or
failing stage cannot block another. `Broadcaster` and `JsonlEventStore` are
shared instances but were already keyed by `stage_id` internally, so sharing
them creates no cross-stage coupling; each `/ws/audience/{stage_id}` client
only ever receives that stage's events.

Two source types are supported per stage, selected by `source.type`:

- `file` — the existing `GeminiTranscriber` + `FileAudioSource` (real Gemini
  Live Transcription).
- `replay` — `ReplayTranscriber` (`app/providers/replay.py`), a real
  provider (not a mock) that replays a recorded transcript from a JSON
  fixture with its original timing, ignoring the audio input entirely. Used
  for quota-free, deterministic development and demos — see
  `data/replay/*.json` and `config/stages.multi.yaml`.

Run a two-stage demo (no Gemini ASR quota consumed — translation still uses
the real API, per-stage, ~2 calls each):

```bash
docker compose run -d --rm -p 8000:8000 -e CAPTIONMESH_CONFIG=config/stages.multi.yaml app
```

Then open `http://localhost:8000/audience` (no `stage_id` — this route
auto-discovers every configured stage from `/health` and renders one panel
per stage side by side; the existing single-stage `/audience/{stage_id}`
page is unchanged).

**Known limitation:** the shutdown grace period (5s, see below) is fixed
regardless of how short a stage's content is. A very short replay stage
(a few seconds) can outrun a real translation call that happens to take
longer than 5s, cancelling it — observed directly in testing. This is the
same disclosed trade-off as before, just more visible on short demo content
than on a multi-minute real talk.

## ASR Session Rotation

Gemini Live Transcribe sessions have a documented ~10 minute limit.
`GeminiTranscriber` (`app/providers/gemini_asr.py`) rotates to a fresh
session before that limit is reached, and reconnects (bounded backoff:
1s/2s/4s/8s, then gives up) if a session drops unexpectedly — entirely
internal to the transcriber. `StagePipeline` never restarts, broadcaster/
persistence/`seg_id` sequence/translation state are all untouched by a
rotation.

Configured globally (applies to every `file`-source stage) via a new
top-level `gemini:` section — no such section existed before this:

```yaml
gemini:
  session_rotation_seconds: 540  # default: a minute of margin before the ~10 min limit
```

An in-flight interim that never resolved to a final before a rotation
simply disappears (never fabricated into a final); the next session's first
segment is marked internally so `StagePipeline` starts a fresh `seg_id`
instead of reusing the old, abandoned one.

**Known limitation, confirmed in testing:** `asr_latency_ms` (see below) is
measured against the stage's overall stream-start time, not reset per
session — across a rotation, the receive-side draining wait and reconnect
handshake both advance wall-clock time while no new audio is being sent,
inflating this metric for messages received during/after that gap. This is
a measurement artifact of rotation, not a transcription problem — real
E2E latency (persisted finals, translations) was unaffected.

## Running the App (single stage)

The real `StagePipeline` runs inside the FastAPI process defined in `app/api.py`.
It wires together `FileAudioSource` (ffmpeg) → `GeminiTranscriber` → `caption.interim`
/ `caption.final` events, broadcast live to audience clients and, for finals only,
persisted as JSONL. Finalized segments are then translated into each configured
`target` (via `GeminiTranslator`) into `caption.translation` events, sharing the
same `seg_id` and also broadcast + persisted. See `docs/spec.md` for the full
pipeline.

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

### Translation

`GeminiTranslator` (`app/providers/gemini_translate.py`) implements a
minimal `TranslationProvider` Protocol (`app/providers/__init__.py`):
`translate(text, source_language, target_language) -> str`, one stateless
call per finalized segment. `StagePipeline` calls it only from `_handle_final`
(never for interim events), once per configured `targets` entry, skipping any
target equal to the source language. A per-target translation failure is
logged and skipped — it does not stop other targets or crash the stage.

Translation runs as an independent, tracked `asyncio.Task` per (segment,
target) pair — `_handle_final` schedules it via `asyncio.create_task` and
never awaits it, so a slow or failing translation call (observed 10–42s
against `gemini-3.5-flash` under real service load) cannot block reading
further transcription messages. This fixed a real, measured problem from an
earlier version that awaited translation inline: it inflated subsequent ASR
latency and, in one run, caused the Gemini Live transcription session itself
to be aborted (`1008: operation aborted`) from inactivity while a 42s
translation call was in flight. Verified fixed against the real API: a
second `caption.final` was processed and persisted while the first one's
translation was still 10s from completing, with ASR latency staying low
throughout.

When `StagePipeline.run()`'s main loop ends, pending translation tasks get a
short grace window (`TRANSLATION_SHUTDOWN_GRACE_SECONDS = 5`, overridable
per instance) to finish before being cancelled — bounded so shutdown can't
hang indefinitely on a stuck call, at the cost of occasionally cancelling a
translation that was still legitimately in flight (observed for real: one
run cancelled a translation started near the end of the session once the
5s grace elapsed, logged clearly, no error). `GET /health` exposes
`pending_translations` per stage.

The audience page does not yet render `caption.translation` events (only
`caption.interim`/`caption.final`) — they flow over the existing WebSocket
and get persisted, but displaying them is unimplemented UI work.

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
