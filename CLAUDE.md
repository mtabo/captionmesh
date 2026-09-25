# CaptionMesh — Claude Code Instructions

## Project

CaptionMesh is an open-source real-time conference captioning and translation system being developed during the Nerdearla Vibeathon 2026.

The priority is to deliver a working, demonstrable MVP before adding architectural complexity.

Read `docs/spec.md` before making architectural changes.

## Core Principle

Optimize for:

1. Working end-to-end system.
2. Low latency.
3. Reliability.
4. Simplicity.
5. Demonstrability.

**Boundaries yes, ceremony no.**

Every abstraction must have a concrete reason to exist.

Do not optimize for hypothetical future scale before the MVP works.

## Architecture

CaptionMesh is a modular monolith.

All MVP components run in one FastAPI process.

The primary runtime concept is a conference `Stage`.

```text
StageSupervisor
      │
      ├── StagePipeline
      │
      ├── StagePipeline
      │
      └── StagePipeline
```

Each StagePipeline coordinates:

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

Each stage is independently supervised.

A failure in one stage must not terminate other stages.

## Project Structure

Prefer this initial structure:

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

Keep the structure small.

Do not create separate `domain`, `application`, `infrastructure`, and `interfaces` layers merely for architectural appearance.

## Configuration

Conference stages are configuration-driven.

Adding a stage should not require application-code changes.

Example:

```yaml
stages:
  - id: main
    name: Main Stage
    language: auto
    targets:
      - es
    source:
      type: file
      path: /data/audio/main.wav
```

`language` describes the source language.

Default:

```text
auto
```

The application must not assume that the transcription provider automatically supplies a reliable language code.

`targets` defines the languages to generate.

A target equal to the source language does not require translation.

The architecture must support both:

```text
EN → ES
ES → EN
```

and future multi-language output.

## Providers

Use small `typing.Protocol` interfaces for external capabilities.

### TranscriptionProvider

P0 implementations:

```text
GeminiTranscriber
ReplayTranscriber
```

### TranslationProvider

P0 implementation:

```text
GeminiTranslator
```

Gemini is an infrastructure adapter, not the conceptual center of CaptionMesh.

Do not hard-code Gemini assumptions into domain/runtime objects when a provider boundary is sufficient.

### ReplayTranscriber

Replay is a real provider, not a fake placeholder.

It should replay recorded transcript events according to their timing.

Use it to:

- test downstream processing
- develop without API calls
- reproduce scenarios
- run multiple stages without multiplying Gemini sessions
- demonstrate scaling

## Anti-Ceremony Rules

Do NOT introduce the following without a concrete demonstrated need:

- dependency injection containers
- repository pattern
- Unit of Work
- generic EventBus
- service locator
- factory hierarchies
- abstract base class hierarchies
- domain event frameworks
- generic workflow engines
- ORM/database layer
- microservices

`typing.Protocol`, dataclasses, Pydantic models, asyncio queues, and small compositional classes are encouraged where they simplify the implementation.

## Gemini

The transcription model is:

```text
gemini-3.5-transcribe-live
```

Canonical audio format:

```text
PCM 16-bit
16 kHz
mono
little-endian
```

Use the Google GenAI Python SDK.

The transcription pipeline must distinguish:

- interim transcription
- finalized transcription

Finalized segments are inputs to translation.

Do not translate every interim transcript.

## Segment Identity

Every finalized transcript receives a stable `seg_id`.

Original and translated captions are separate events correlated through `seg_id`.

Never make the frontend match transcript text to associate translations.

## Events

Events are an explicit internal contract.

Keep event types small and typed.

A stage may use an asyncio queue or equivalent mechanism for internal event flow.

Do not create a generic EventBus abstraction.

## Realtime Behavior

Latency matters.

Use bounded queues.

Never allow an unbounded audio backlog.

If the system falls behind, prefer dropping stale audio over creating ever-growing latency.

## Session Lifecycle

Gemini Live Transcription sessions have a finite continuous streaming duration.

Implement explicit reconnect/rotation.

Do not assume that a transcription session can remain open indefinitely.

Session lifecycle is isolated per stage.

A failure in Stage A must not terminate Stage B.

## Audio

P0 uses file/stream input.

Use ffmpeg for normalization.

Canonical format:

```text
PCM 16-bit / 16 kHz / mono / little-endian
```

Browser microphone support is P1.

Do not spend P0 time building AudioWorklet infrastructure unless explicitly requested.

## Translation

Translate finalized transcript segments.

Translation must be implemented behind a small provider interface.

Translate only to configured `targets`.

Do not build a translation orchestration framework.

## Frontend

Keep the audience UI simple.

The UI must prioritize:

- readability
- stable captions
- low jitter
- stage identification
- connection status

Avoid complex frontend frameworks unless they provide a concrete time-saving benefit.

## Persistence

No database is required for P0.

JSONL is sufficient for stage events and debugging.

VTT generation should consume persisted/events data rather than become part of the transcription pipeline.

## Testing

Prioritize useful integration tests.

Important test areas:

- audio normalization
- event parsing
- segment IDs
- translation correlation
- stage isolation
- replay provider
- stage lifecycle

Replay should be used wherever deterministic downstream tests are useful.

Do not spend excessive time building a large test suite during the hackathon.

## Development Workflow

Implement incrementally:

```text
implement
    ↓
run
    ↓
observe
    ↓
fix
    ↓
commit
```

Preferred order:

1. Gemini transcription spike.
2. One stage end-to-end.
3. Translation.
4. Audience WebSocket/UI.
5. Two stages.
6. Replay provider.
7. Reconnect.
8. Documentation.
9. Optional features.
10. Demo.

Do not build the entire architecture before validating the first realtime path.

## Git

Use small, meaningful commits.

Examples:

```text
feat: add Gemini live transcription spike
feat: add stage pipeline
feat: add translation pipeline
feat: add audience websocket
feat: support concurrent stages
feat: add replay transcription provider
fix: reconnect transcription session
docs: document deployment
```

Avoid unrelated changes in the same commit.

## Scope Control

If a proposed feature is not required for P0, classify it as P1/P2 before implementing it.

When uncertain, choose the simpler implementation that satisfies the acceptance criteria.

Do not redesign the architecture without evidence from:

- an actual requirement
- a failing test
- an implementation constraint
- an observed operational problem

## Hackathon Constraint

This repository is being created and developed during the Vibeathon.

Do not copy implementation code from pre-existing private projects.

General knowledge, open-source dependencies, documented patterns, and newly written implementation are allowed.

## Definition of Done

A feature is not done because the code compiles.

For realtime features, verify actual runtime behavior.

Prefer:

```text
implement
→ run
→ observe
→ fix
→ commit
```

over:

```text
implement everything
→ test at the end
```