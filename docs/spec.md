# CaptionMesh — SPEC v0.1

## 1. Purpose

CaptionMesh is an open-source real-time conference captioning and translation system designed for open conferences.

The system receives live audio from conference stages, produces real-time transcription in the original language, translates speech into configured target languages, and distributes synchronized captions to audience clients.

The hackathon MVP prioritizes reliability, low latency, simple deployment, and support for multiple simultaneous stages.

## 2. Hackathon Goal

Build a working open-source prototype during the Vibeathon that demonstrates:

- Live audio ingestion.
- Real-time transcription.
- English → Spanish translation.
- Original and translated captions displayed simultaneously.
- At least two concurrent stages.
- A clear architecture that can scale to more stages.
- Public repository.
- OSI-approved open-source license.
- Reproducible setup documented in the README.
- A 1–2 minute demo using real talk audio.

The architecture should also support other source/target language combinations without changing the core pipeline.

## 3. Architecture

CaptionMesh is a **modular monolith**.

All MVP components run inside a single FastAPI process.

The system is organized around independently supervised conference stages rather than independent services.

```text
                         CaptionMesh
                              │
                     StageSupervisor
                              │
             ┌────────────────┼────────────────┐
             ▼                ▼                ▼
         Stage A           Stage B           Stage N
             │
        StagePipeline
             │
     ┌───────┼────────────┐
     ▼       ▼            ▼
  Source  Transcriber  Translator
             │            │
             └──────┬─────┘
                    ▼
                  Events
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
      Broadcast    Store      VTT
          │
          ▼
       Audience
```

The core architecture must remain simple enough to develop and operate as one process.

Do not introduce distributed infrastructure unless an actual requirement demonstrates that it is necessary.

## 4. Core Concepts

### Conference

A conference contains multiple stages.

### Stage

A stage represents one independent live audio/captioning pipeline.

Each stage has:

- configuration
- audio source
- transcription provider
- translation provider(s)
- event stream
- broadcaster
- persistence
- runtime status

A failure in one stage must not terminate other stages.

### StageSupervisor

The StageSupervisor creates, starts, monitors, and stops StagePipeline instances based on configuration.

Adding a stage should require configuration rather than application-code changes.

### StagePipeline

A StagePipeline coordinates the processing flow for one stage:

```text
Source
  ↓
Transcription
  ↓
Translation
  ↓
Events
  ↓
Broadcast / Store
```

The pipeline is an orchestration component, not a generic workflow framework.

## 5. Project Structure

The initial application structure should remain intentionally small:

```text
app/
├── config.py
├── events.py
├── stage.py
├── providers/
│   ├── gemini_asr.py
│   ├── gemini_translate.py
│   └── replay.py
├── sources/
│   ├── ffmpeg.py
│   └── web.py
├── broadcast.py
├── store.py
└── api.py
```

This structure represents architectural boundaries without introducing unnecessary layers.

Use composition and `typing.Protocol` for replaceable providers.

Do not create generic domain/application/infrastructure frameworks solely for architectural appearance.

## 6. Configuration

Conference and stage configuration should be external to application code.

Example:

```yaml
conference:
  name: Nerdearla 2026

stages:
  - id: main
    name: Main Stage
    language: auto
    targets:
      - es
    source:
      type: file
      path: /data/audio/main.wav

  - id: devroom
    name: Dev Room
    language: es
    targets:
      - en
    source:
      type: file
      path: /data/audio/devroom.wav
```

### `language`

`language` defines the source language.

Allowed behavior:

- explicit language such as `en` or `es`
- `auto` for automatic detection

Default:

```yaml
language: auto
```

The application must not assume that transcription automatically provides a reliable language code.

When `auto` is used, language identification must be handled by the appropriate model/component before target translation is selected.

When the conference schedule already provides the source language, it should be possible to explicitly configure it. This avoids unnecessary ambiguity during the first seconds of a session.

### `targets`

`targets` defines the languages to generate.

A target language equal to the source language does not require translation.

Example:

```yaml
language: en
targets:
  - es
```

means English transcription with Spanish translation.

This model also supports the optional ES → EN use case:

```yaml
language: es
targets:
  - en
```

and future multi-language output:

```yaml
language: en
targets:
  - es
  - pt
```

## 7. Providers

External capabilities must be represented through small provider interfaces.

Use `typing.Protocol` rather than inheritance-heavy abstraction.

### TranscriptionProvider

P0 implementations:

```text
GeminiTranscriber
ReplayTranscriber
```

### GeminiTranscriber

Production realtime transcription using Google's Gemini Live Transcription API.

### ReplayTranscriber

A deterministic transcription provider that replays a previously recorded transcript with timing information.

Replay exists to:

- develop without API usage
- test downstream components deterministically
- reproduce known scenarios
- test multiple simultaneous stages without requiring multiple Gemini sessions
- demonstrate scaling without making unnecessary external API calls

This makes the provider boundary operationally useful rather than merely hypothetical.

### TranslationProvider

P0 implementation:

```text
GeminiTranslator
```

The provider interface should remain small and replaceable.

No provider factory or dependency injection container is required.

## 8. Transcription

Use Google's Gemini Live Transcription model:

```text
gemini-3.5-transcribe-live
```

Canonical input format:

```text
PCM 16-bit
16 kHz
mono
little-endian
```

The transcription pipeline must distinguish:

- interim transcription
- finalized transcription

Finalized transcription segments become inputs to translation.

Do not translate every interim transcription update.

### Session lifecycle

Gemini Live Transcription sessions have a finite continuous streaming duration.

CaptionMesh must therefore support explicit reconnect/rotation.

Session lifecycle is isolated per stage.

A transcription failure in Stage A must not terminate Stage B.

## 9. Audio

P0 audio ingestion is file/stream based.

The canonical internal format is:

```text
PCM 16-bit
16 kHz
mono
little-endian
```

ffmpeg is responsible for normalization when necessary.

Browser microphone ingestion is P1.

The system should use bounded queues and must not allow an unbounded audio backlog.

If realtime processing falls behind, stale audio should not accumulate indefinitely.

For realtime captioning, bounded latency is more important than processing every delayed packet.

## 10. Translation

Translation operates on finalized transcription segments.

```text
Final transcript
      ↓
TranslationProvider
      ↓
Target language captions
```

Translation must not be performed on every interim transcript.

The system should generate translations only for configured `targets`.

Each translation must reference the original segment using `seg_id`.

## 11. Events

Events are the internal contract between processing and output components.

Each finalized transcription segment receives a stable `seg_id`.

Example:

```json
{
  "type": "caption.final",
  "stage_id": "main",
  "seg_id": "main-000123",
  "text": "Welcome to the conference.",
  "language": "en"
}
```

Translation is a separate event:

```json
{
  "type": "caption.translation",
  "stage_id": "main",
  "seg_id": "main-000123",
  "language": "es",
  "text": "Bienvenidos a la conferencia."
}
```

Consumers correlate original and translated captions through `seg_id`.

The event model must remain explicit and small.

Do not introduce a generic enterprise-style EventBus.

A stage may use an asyncio queue or equivalent bounded mechanism for its internal event flow.

## 12. Audience Interface

The audience interface must provide:

- stage selection
- original-language captions
- translated captions
- clear stage identification
- connection status
- readable typography
- automatic scrolling
- low visual jitter

The UI should remain intentionally simple.

A complex frontend framework is not required for P0.

## 13. Broadcast

Each stage provides a WebSocket endpoint for audience clients.

Broadcasting is stage-local.

The broadcaster performs fan-out to connected audience clients.

Audience clients are read-only consumers of caption events.

## 14. Persistence

The MVP does not require a database.

Stage events may be persisted as JSONL:

```text
data/stages/<stage_id>.jsonl
```

This provides:

- debugging
- replay
- basic persistence
- a source for future VTT generation

## 15. VTT

VTT export is an optional P1 capability.

The event model and persisted data should contain enough information to make VTT generation possible without changing the transcription or translation pipeline.

## 16. Optional Features — P1

Only implement after P0 is working:

- technical glossary
- custom vocabulary
- VTT export
- latency/status dashboard
- browser microphone input
- measured 10-stage test
- prepared session rotation
- OBS overlay

## 17. Explicitly Out of Scope

Do not implement during the initial MVP:

- Kubernetes
- Redis
- Kafka
- RabbitMQ
- Celery
- PostgreSQL
- distributed workers
- service mesh
- multi-instance orchestration
- audience authentication
- user accounts
- dependency injection containers
- repository pattern
- Unit of Work
- generic EventBus
- unnecessary abstract base classes
- complex frontend build systems
- vMix integration
- Gemma/local inference
- speaker diarization
- word-level timestamps
- enterprise-grade observability
- billing system

These may be considered only if a concrete requirement appears.

## 18. Scaling Model

The system scales conceptually by adding independent stage pipelines:

```text
Stage A ──┐
Stage B ──┤
Stage C ──┤
Stage D ──┤
Stage N ──┘
```

For the MVP, all stages may execute within one process.

The README should explain that larger deployments could distribute stages across multiple application instances if required.

The architecture should not require distributed infrastructure to demonstrate the scaling model.

## 19. Reliability

Each stage must have:

- supervised lifecycle
- explicit runtime status
- reconnect handling
- bounded audio buffering
- error logging

A stage failure must be isolated.

The application must remain operational when another stage fails.

## 20. Security

Gemini API credentials must remain server-side.

Audience clients must never receive the Gemini API key.

Untrusted caption content must not be rendered through raw HTML.

Environment variables should be used for secrets.

## 21. Demo Acceptance Criteria

The final demo should demonstrate:

1. Two simultaneous conference stages.
2. Live original-language transcription.
3. English → Spanish translation.
4. Switching between stages.
5. Stable audience captions.
6. Public open-source repository.
7. Simple deployment.
8. Explanation of scaling to additional stages.

If time permits:

- 10 simultaneous replay stages
- glossary
- VTT export
- measured latency
- CaptionMesh-generated subtitles on the demo itself

## 22. Development Strategy

Development order:

1. Gemini transcription spike.
2. One complete stage.
3. Translation.
4. Audience WebSocket/UI.
5. Two concurrent stages.
6. Replay provider.
7. Reconnect handling.
8. Packaging and documentation.
9. Optional features.
10. Demo.

Do not optimize or generalize before the end-to-end path works.

Every architectural abstraction should pay for itself through a concrete use case, testability requirement, or operational benefit.