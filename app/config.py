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


class ConferenceConfig(BaseModel):
    name: str = ""
    stages: list[StageConfig]


def load_conference_config(path: Path) -> ConferenceConfig:
    raw = yaml.safe_load(path.read_text())
    conference = raw.get("conference") or {}
    return ConferenceConfig(name=conference.get("name", ""), stages=raw.get("stages", []))
