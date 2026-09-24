# CaptionMesh — Claude Code Instructions

## Project

CaptionMesh is an open-source real-time conference captioning system being developed during the Nerdearla Vibeathon 2026.

The priority is to deliver a working, demonstrable MVP before adding architectural complexity.

Read `docs/spec.md` before making architectural changes.

## Core Principle

Optimize for:

1. Working end-to-end system.
2. Low latency.
3. Reliability.
4. Simplicity.
5. Demonstrability.

Do not optimize for hypothetical future scale before the MVP works.

## Architecture

The MVP is a single FastAPI process containing independent stage pipelines:

```text
Source
  ↓
Audio Queue
  ↓
Transcriber
  ↓
Translator
  ↓
Broadcaster
```

Each stage has independent state and failure handling.

Do not introduce microservices unless explicitly requested.

## Forbidden Complexity for P0

Do NOT introduce:

- Redis
- Kafka
- RabbitMQ
- Celery
- Kubernetes
- PostgreSQL
- distributed workers
- service mesh
- React
- complex frontend build systems
- authentication systems
- multi-instance orchestration

unless the user explicitly requests them.

## Gemini

The transcription model is:

```text
gemini-3.5-transcribe-live
```

The canonical audio format is:

```text
PCM 16-bit
16 kHz
mono
little-endian
```

Use the Google GenAI Python SDK.

The transcription pipeline must distinguish:

- interim transcription
- final transcription

Final transcription segments are the input to translation.

Do not translate every interim transcript.

## Segment Identity

Every finalized transcript must receive a stable `seg_id`.

Original transcript and translation are separate events correlated through `seg_id`.

Never rely on frontend text matching to associate translations with transcripts.

## Realtime Behavior

Latency matters.

Use bounded queues.

Never allow an unbounded backlog of audio.

If the system falls behind, prefer dropping stale audio over creating an ever-growing latency buffer.

## Session Lifecycle

Gemini Live Transcription sessions have a finite continuous streaming duration.

Implement reconnect/rotation explicitly.

Do not assume a transcription session can remain open indefinitely.

Session lifecycle must be isolated per stage.

A session failure in Stage A must not terminate Stage B.

## Audio

P0 uses file/stream input.

Use ffmpeg for normalization.

Canonical internal format:

```text
PCM 16-bit / 16 kHz / mono / little-endian
```

Browser microphone support is optional P1.

Do not spend P0 time building AudioWorklet infrastructure unless explicitly requested.

## Translation

Translate finalized transcript segments.

Translation must be implemented behind an internal interface.

Keep the implementation replaceable.

## Frontend

Keep the audience UI simple.

Avoid unnecessary frontend frameworks.

The UI must prioritize:

- readability
- stable captions
- low jitter
- stage identification
- connection status

## Persistence

No database is required for P0.

JSONL is sufficient for stage event persistence and debugging.

## Configuration

Keep configuration explicit and simple.

Prefer environment variables for secrets.

Never commit API keys.

## Testing

Every new component should have the smallest useful test.

Prioritize integration tests for:

- audio normalization
- transcription event parsing
- segment IDs
- translation correlation
- stage isolation

Do not spend excessive time building a large test suite during the hackathon.

## Development Workflow

Work incrementally.

Preferred order:

1. Gemini transcription spike.
2. One stage end-to-end.
3. Translation.
4. Audience UI.
5. Two stages.
6. Reconnect.
7. Documentation.
8. Optional features.

After each milestone, run the application and verify actual behavior.

## Git

Use small, meaningful commits.

Examples:

```text
feat: add Gemini live transcription spike
feat: add stage pipeline
feat: add translation pipeline
feat: add audience websocket
feat: support concurrent stages
fix: reconnect transcription session
docs: document deployment
```

Do not make large unrelated commits.

## Scope Control

If a proposed feature is not required for the MVP, classify it as P1/P2 before implementing it.

When uncertain, choose the simpler implementation that satisfies the acceptance criteria.

Do not redesign the architecture without evidence from an actual requirement or implementation problem.

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