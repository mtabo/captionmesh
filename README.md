# CaptionMesh

CaptionMesh is an open-source real-time conference captioning and translation
system, built during the Nerdearla Vibeathon 2026. It streams audio through
Gemini Live for live transcription, translates each caption asynchronously,
and delivers both to an audience page over WebSocket — all as a single
FastAPI process (a modular monolith, not microservices).

## Features

- **Live transcription** of an audio source via Gemini Live
  (`gemini-3.5-transcribe-live`), decoded to PCM through ffmpeg.
- **Sentence-level captions**: transcript sentences are committed and shown
  as soon as they're complete, instead of waiting for large paragraph-sized
  blocks.
- **English → Spanish translation** (or any language pair configured per
  stage), run asynchronously per caption so a slow translation call never
  blocks transcription.
- **Live audience subtitles** delivered over WebSocket, with original and
  translated captions shown together.
- **Technical glossary / custom vocabulary**: a configurable term list
  (e.g. `Firebase`, `Kubernetes`, `CaptionMesh`) biases ASR recognition and
  is preserved untranslated.
- **WebVTT export** of a stage's finalized captions.
- **Independent, multi-stage pipelines**: each configured stage runs its own
  audio source, Gemini session, translation tasks and WebSocket subscribers,
  isolated from every other stage.
- **In-browser audio playback** synced to the position the pipeline is
  currently transcribing, not just started from 0:00.

## Quick Start

**Requirements:** Docker and Docker Compose. Nothing else — Python, ffmpeg
and all dependencies run inside the container; you don't need them on your
host.

```bash
git clone <this-repo-url>
cd captionmesh
cp .env.example .env
```

Edit `.env` and set your Gemini API key (see [Configuration](#configuration)
below), then start the app:

```bash
docker compose up -d app
```

The first run builds the image, which can take a minute. Verify it's up:

```bash
curl localhost:8000/health
```

Then open the audience page in a browser:

- **`http://localhost:8000/audience`** — all configured stages, side by side.
- **`http://localhost:8000/audience/main`** — a single stage by id.

Follow the logs:

```bash
docker compose logs -f app
```

Stop everything:

```bash
docker compose down
```

## Configuration

```bash
cp .env.example .env
```

`.env.example` currently defines a single variable:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace the value with a real Gemini API key that has access to
`gemini-3.5-transcribe-live`. `.env` is gitignored — never commit it.

Which audio stages run, and in which languages, is controlled separately by
YAML files under `config/` (see [Demo](#demo) and
[Multi-stage / Scaling](#multi-stage--scaling)), selected via the
`CAPTIONMESH_CONFIG` environment variable (default: `config/stages.yaml`).

## Demo

The default config (`config/stages.yaml`) is already set up to caption a
demo fixture included in the repo:

```
data/audio/nerdearla-demo-main.wav
```

This is a ~60-second excerpt from a real Nerdearla conference talk, included
as a demo/test fixture for the project — no separate upload or setup is
needed. Running `docker compose up -d app` and opening
`http://localhost:8000/audience/main` is enough to see it transcribed and
translated to Spanish live.

For a two-stage demo (`main` + `devroom`, both using their own demo audio
file), see [Multi-stage / Scaling](#multi-stage--scaling).

## How it works

```
Audio source (file, via ffmpeg)
    → Gemini Live transcription (interim + final segments)
    → sentence commit (publish each completed sentence as soon as it's done)
    → async translation (one task per sentence/target language)
    → WebSocket broadcast + JSONL persistence
    → audience UI
```

Translation never blocks transcription: each finalized sentence schedules an
independent, tracked `asyncio.Task` per target language, and the pipeline
keeps reading further Gemini messages without waiting for it to finish.

## Architecture

`StageSupervisor` (`app/supervisor.py`) creates one `StagePipeline`
(`app/stage.py`) per configured stage, each running as its own
`asyncio.Task`. A stage's state is fully independent of every other stage:

- its own audio source (`FileAudioSource`, `app/sources/ffmpeg.py`)
- its own Gemini Live transcription session
- its own translation tasks
- its own event stream (interim/final/translation events)
- its own WebSocket subscribers
- its own persisted event log (`data/stages/<stage_id>.jsonl`)

A failure or slow operation in one stage cannot block another. Everything
runs inside one FastAPI process — this is a modular monolith, not a set of
separate services.

## Multi-stage / Scaling

Adding a stage means adding an entry to the YAML config — no code change.
`config/stages.multi.yaml` configures two stages already, both using demo
audio fixtures included in the repo:

```yaml
stages:
  - id: main
    name: Main Stage
    language: en
    targets: [es]
    source:
      type: file
      path: data/audio/nerdearla-demo-main.wav

  - id: devroom
    name: Dev Room
    language: en
    targets: [es]
    source:
      type: file
      path: data/audio/nerdearla-demo-devroom.wav
```

Run it with:

```bash
docker compose run -d --rm -p 8000:8000 -e CAPTIONMESH_CONFIG=config/stages.multi.yaml app
```

**Known limitation:** two Gemini Live sessions opened on the same API key at
nearly the same time were measured to sometimes degrade — one stage's
session can stall with no captions while the other works fine. This wasn't
fully solved; `gemini.session_start_stagger_seconds` in the config
(currently 15s in `stages.multi.yaml`) delays starting the second stage's
session, which reduces but does not guarantee against this. No specific
number of reliable concurrent stages is claimed.

Conceptually, this same per-stage-pipeline model could scale by
distributing stages across multiple application instances for larger
deployments — that is not implemented today; the current architecture runs
every configured stage in one process.

## API / Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Per-stage status, latency stats, pending translations, audio URLs/positions. |
| `GET /audience` | Audience UI showing all configured stages at once. |
| `GET /audience/{stage_id}` | Audience UI for a single stage. |
| `WS /ws/audience/{stage_id}` | Read-only WebSocket stream of a stage's caption/translation events. |
| `GET /api/stages/{stage_id}/audio` | Streams the stage's source audio file (only for `file`-type stages). |
| `GET /api/stages/{stage_id}/captions.vtt` | WebVTT export of the stage's finalized captions (always fresh, also saved as `data/vtt/{stage_id}.vtt`). |
| `GET /docs` | Auto-generated FastAPI/Swagger API explorer. |

Beyond the on-demand endpoint above, every stage that finishes a session
cleanly also gets its own archived, timestamped WebVTT file written
automatically to `data/vtt/{stage_id}_{YYYYMMDD-HHMMSS}.vtt` — no manual
request needed.

## Testing

```bash
docker compose run --rm --entrypoint bash spike -c "python -m pytest tests/ -v"
```

This reuses the `spike` service's image (same build as `app`) just to run
`pytest` with an overridden entrypoint. At the time of writing this runs
130 tests covering config parsing, the Gemini ASR/translation providers
(against fake sessions, no real API calls), stage pipeline behavior (seg_id
assignment, sentence commit, duplicate handling), multi-stage isolation,
the VTT/store/broadcast layers, and the HTTP API.

## Project Structure

```text
app/
  api.py              FastAPI app: routes, lifespan, WebSocket endpoint
  supervisor.py        StageSupervisor: creates/runs one StagePipeline per stage
  stage.py              StagePipeline: transcription → captions → translation
  config.py             YAML config models (ConferenceConfig, StageConfig, GeminiConfig)
  providers/            GeminiTranscriber, GeminiTranslator, GeminiSegmenter, ReplayTranscriber
  sources/               FileAudioSource (ffmpeg decoding)
  static/                  audience.html, audience_multi.html
  vtt.py, store.py, broadcast.py, stats.py, events.py

config/                  YAML stage configs (stages.yaml, stages.multi.yaml)
data/
  audio/                   Audio fixtures (some gitignored, demo files tracked)
  replay/                   Replay transcript fixtures (deterministic, no Gemini quota)
  stages/                   Persisted per-stage JSONL event logs (generated at runtime)
  vtt/                       Generated WebVTT snapshots (generated at runtime)

tests/                   pytest suite (see Testing)
docker/Dockerfile        Single image used by all docker-compose services
```

## Troubleshooting

- **Nothing happens / stage status is `error`:** check the container logs
  (`docker compose logs -f app`) and confirm `GEMINI_API_KEY` in `.env` is
  set and valid.
- **Port 8000 already in use:** stop whatever is using it, or change the
  `"8000:8000"` port mapping in `docker-compose.yml`.
- **A stage never produces captions in the two-stage demo:** see the known
  concurrent-session limitation in
  [Multi-stage / Scaling](#multi-stage--scaling) — restart
  (`docker compose down && docker compose up -d app`) to retry.
- **Stale test/build errors after pulling changes:** rebuild the images —
  `docker compose build`.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
