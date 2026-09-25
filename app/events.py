from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class CaptionFinalEvent(BaseModel):
    type: Literal["caption.final"] = "caption.final"
    stage_id: str
    seg_id: str
    text: str
    language: str
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
