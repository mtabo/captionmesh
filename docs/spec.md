# CaptionMesh — SPEC v0.1

## 1. Purpose

CaptionMesh is an open-source real-time captioning and translation system designed for open conferences.

The system receives live audio from conference stages, produces real-time transcription in the original language, translates English speech into Spanish, and distributes synchronized captions to audience clients.

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

## 3. MVP Scope — P0

### Required

1. Audio source from a local audio file or stream.
2. Audio normalization using ffmpeg.
3. Streaming PCM audio to Gemini Live Transcription.
4. Real-time interim transcription.
5. Finalized transcription segments.
6. English → Spanish translation of finalized segments.
7. Stable segment identifiers (`seg_id`).
8. WebSocket distribution to audience clients.
9. Two simultaneous stages.
10. Audience web UI.
11. Stage status.
12. Basic session reconnect handling.
13. Docker Compose deployment.
14. README with complete setup instructions.
15. OSI-approved open-source license.

### P0 Architecture

```text
Audio Source
    │
    ▼
Source Adapter
    │
    ▼
Audio Queue
    │
    ▼
Gemini Transcriber
    │
    ├── interim transcript
    │
    └── final transcript
              │
              ▼
         Translator
              │
              ▼
         Broadcaster
              │
       ┌──────┴──────┐
       ▼             ▼
   Audience       Stage State
    WebSocket       JSONL
```

Each stage runs independently inside the same backend process.

## 4. Architecture Principles

### Single-process MVP

The MVP uses a single FastAPI process.

Do not introduce:

- Redis
- Kafka
- RabbitMQ
- Kubernetes
- Celery
- distributed workers
- external databases
- microservices

unless a concrete requirement emerges later.

### Stage isolation

Each conference stage has its own:

- source
- audio queue
- transcription session
- translation flow
- broadcast channel
- state

Failure of one stage must not terminate other stages.

### Event-driven captions

Caption events use a stable `seg_id`.

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

The audience UI must correlate both events using `seg_id`.

## 5. Transcription

Use Google's Gemini Live Transcription model:

```text
gemini-3.5-transcribe-live
```

Input:

```text
PCM 16-bit
16 kHz
mono
little-endian
```

The system must support:

- interim transcription
- final transcription
- reconnect after session termination

Live Transcription sessions are limited to approximately 10 minutes of continuous streaming.

The MVP should therefore implement reconnect/rotation rather than depend on indefinite sessions.

## 6. Translation

Translation occurs after a finalized transcription segment.

P0:

```text
final transcript
      ↓
translation model
      ↓
Spanish translation
```

Translation should not be performed on every interim transcript.

The UI must prioritize stable captions over character-by-character translation updates.

The translation component should be isolated behind an internal interface so the provider can be changed later without modifying the transcription pipeline.

## 7. Audio

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

## 8. Audience Interface

The audience interface must provide:

- stage selection
- original-language captions
- Spanish translation
- clear indication of stage
- connection status
- readable typography
- automatic scrolling
- low visual jitter

The interface should remain intentionally simple.

## 9. Persistence

The MVP does not require a database.

Stage events may be persisted as JSONL:

```text
data/stages/<stage_id>.jsonl
```

This provides:

- debugging
- replay
- basic persistence
- future VTT generation

## 10. Optional Features — P1

Only implement after P0 is working:

- technical glossary
- custom vocabulary
- VTT export
- latency/status dashboard
- browser microphone input
- measured 10-stage test
- prepared session rotation
- OBS overlay

## 11. Explicitly Out of Scope

Do not implement during the initial MVP:

- Kubernetes
- Redis
- Kafka
- multi-instance orchestration
- authentication for audience users
- user accounts
- complex frontend frameworks
- React unless a concrete requirement appears
- vMix integration
- Portuguese
- Gemma/local inference
- speaker diarization
- word-level timestamps
- enterprise-grade observability
- billing system

## 12. Scaling Model

The system must conceptually scale by adding independent stage pipelines:

```text
Stage A ──┐
Stage B ──┤
Stage C ──┤
Stage D ──┤
Stage N ──┘
```

For the MVP, all stages may execute within one process.

The README must explain how the architecture could evolve toward multiple backend instances if required.

## 13. Reliability Requirements

A failure in one stage must not crash the complete application.

Each stage should have:

- supervised task lifecycle
- explicit status
- reconnect handling
- bounded audio buffering
- error logging

Stale audio should not be allowed to accumulate indefinitely.

For realtime captioning, bounded latency is more important than processing every delayed audio packet.

## 14. Security

Gemini API credentials must remain server-side.

Audience clients must never receive the Gemini API key.

The application should avoid rendering untrusted caption content through raw HTML.

## 15. Demo Acceptance Criteria

The final demo should demonstrate:

1. Two simultaneous conference stages.
2. Live original-language transcription.
3. English → Spanish translation.
4. Switching between stages.
5. Stable audience captions.
6. Open-source repository.
7. Simple deployment.
8. Explanation of how the system scales to additional stages.

If time permits, demonstrate:

- 10 simultaneous stage pipelines
- glossary
- VTT export
- CaptionMesh-generated subtitles on the demo itself

## 16. Development Strategy

Development order:

1. Gemini transcription spike.
2. One complete stage.
3. Translation.
4. Audience WebSocket/UI.
5. Two concurrent stages.
6. Reconnect handling.
7. Packaging and documentation.
8. Optional features.
9. Demo.

Do not optimize or generalize before the end-to-end path works.