from datetime import datetime, timezone
from typing import Literal, Union

from pydantic import BaseModel, Field


class _CaptionEventBase(BaseModel):
    stage_id: str
    seg_id: str
    text: str
    language: str
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CaptionInterimEvent(_CaptionEventBase):
    type: Literal["caption.interim"] = "caption.interim"


class CaptionFinalEvent(_CaptionEventBase):
    type: Literal["caption.final"] = "caption.final"


StageEvent = Union[CaptionInterimEvent, CaptionFinalEvent]
