from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, Field


class FileSourceConfig(BaseModel):
    type: Literal["file"] = "file"
    path: Path


class ReplaySourceConfig(BaseModel):
    type: Literal["replay"] = "replay"
    path: Path


SourceConfig = Annotated[Union[FileSourceConfig, ReplaySourceConfig], Field(discriminator="type")]


class StageConfig(BaseModel):
    id: str
    name: str
    language: str = "auto"
    targets: list[str] = Field(default_factory=list)
    source: SourceConfig


class GeminiConfig(BaseModel):
    # Gemini Live Transcribe sessions have a documented ~10 minute limit.
    # Default rotates a full minute before that, not at the exact boundary.
    session_rotation_seconds: float = 540.0


class ConferenceConfig(BaseModel):
    name: str = ""
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    stages: list[StageConfig]


def load_conference_config(path: Path) -> ConferenceConfig:
    raw = yaml.safe_load(path.read_text())
    conference = raw.get("conference") or {}
    gemini = raw.get("gemini") or {}
    return ConferenceConfig(
        name=conference.get("name", ""),
        gemini=GeminiConfig(**gemini),
        stages=raw.get("stages", []),
    )
