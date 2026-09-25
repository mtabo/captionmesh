"""Minimal WebVTT export for a stage's finalized captions.

Reads the persisted event stream (via JsonlEventStore.read_events — no new
storage abstraction) and builds a WebVTT document from it.

Timing is intentionally honest, not invented:

- Only `caption.final` events with `timing.audio_elapsed_ms` present become
  cues. `caption.interim` events are never persisted to begin with, so they
  never reach this function.
- `caption.translation` events NEVER carry timing — StagePipeline builds
  them (`_translate_one`) without a `timing` argument, so it is always
  `None`. A translation can therefore only ever supply substitute cue TEXT
  for a seg_id that already has a timed original final; it can never define
  a cue's timestamps.
- A final's own `audio_elapsed_ms` (elapsed source-audio time when the
  segment was finalized) is treated as that cue's END. Its START is the
  previous included cue's END (0.0 for the first) — the closest boundary
  this event model actually has, since interim events (which would hint at
  a true start) are never persisted. This is an approximation of the true
  utterance start, not measured data; documented, not hidden.
- A final without `timing` (e.g. from a provider that doesn't supply it,
  such as ReplayTranscriber) is skipped rather than given a fabricated
  timestamp. See README for this limitation.
"""

WEBVTT_HEADER = "WEBVTT"


def _format_timestamp(ms: float) -> str:
    """WebVTT requires HH:MM:SS.mmm cue timestamps."""
    total_ms = max(0, int(round(ms)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def build_vtt(events: list[dict]) -> str:
    """Build a WebVTT document from one stage's raw persisted events.

    Prefers a translation's text for a cue when one exists for that
    seg_id (MVP choice: EN->ES stages show the Spanish translation);
    otherwise falls back to the original final's own text.
    """
    finals = [
        e for e in events
        if e.get("type") == "caption.final" and e.get("timing") is not None
    ]
    translation_text_by_seg = {
        e["seg_id"]: e["text"] for e in events if e.get("type") == "caption.translation"
    }

    lines = [WEBVTT_HEADER, ""]
    previous_end_ms = 0.0
    for cue_index, final in enumerate(finals, start=1):
        end_ms = final["timing"]["audio_elapsed_ms"]
        if end_ms <= previous_end_ms:
            # Keep the timeline monotonically increasing even if source
            # timing data is out of order or duplicated.
            end_ms = previous_end_ms + 1.0
        start_ms = previous_end_ms
        text = translation_text_by_seg.get(final["seg_id"], final["text"])

        lines.append(str(cue_index))
        lines.append(f"{_format_timestamp(start_ms)} --> {_format_timestamp(end_ms)}")
        lines.append(text)
        lines.append("")

        previous_end_ms = end_ms

    return "\n".join(lines)
