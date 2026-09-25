from datetime import datetime, timezone
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


class EventTiming(BaseModel):
    """Honest, measured timing for a transcript event.

    `audio_elapsed_ms` is source-relative (elapsed audio content sent to the
    transcription provider so far), not a wall-clock capture timestamp — the
    canonical P0 source is a prerecorded file, not a live microphone.
    `asr_latency_ms` is how far behind that audio timeline this event arrived,
    i.e. (wall-clock elapsed since streaming started) - audio_elapsed_ms.
    """

    audio_elapsed_ms: float
    asr_latency_ms: float


class _CaptionEventBase(BaseModel):
    stage_id: str
    seg_id: str
    text: str
    language: str
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    timing: Optional[EventTiming] = None


class CaptionInterimEvent(_CaptionEventBase):
    type: Literal["caption.interim"] = "caption.interim"


class CaptionFinalEvent(_CaptionEventBase):
    type: Literal["caption.final"] = "caption.final"


StageEvent = Union[CaptionInterimEvent, CaptionFinalEvent]
