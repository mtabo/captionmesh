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


# Technical terms/proper nouns biased for ASR recognition (via
# AudioTranscriptionConfig.custom_vocabulary) and preserved untranslated in
# translation (via GeminiTranslator's system_instruction) — same list, two
# native Gemini mechanisms, no extra model calls either side.
DEFAULT_GLOSSARY = [
    "Firebase",
    "Firebase Studio",
    "Google Cloud",
    "Gemini",
    "Nerdearla",
    "CaptionMesh",
    "FastAPI",
    "WebSocket",
    "Docker",
    "FFmpeg",
    "Kubernetes",
    "Python",
    "JavaScript",
    "TypeScript",
]


class GeminiConfig(BaseModel):
    # Gemini Live Transcribe sessions have a documented ~10 minute limit.
    # Default rotates a full minute before that, not at the exact boundary.
    session_rotation_seconds: float = 540.0
    # Delay between starting consecutive stages. Gemini Live sessions opened
    # within ~1s of each other on the same API key were measured producing
    # one degraded session (first interim after ~12s, then silence);
    # spacing session creation apart avoids that. 0 = start all at once.
    session_start_stagger_seconds: float = 0.0
    # Technical terms/proper nouns to bias ASR recognition toward and
    # preserve untranslated. Override per-deployment via YAML; empty list
    # disables both (no vocabulary bias, no glossary instruction).
    glossary: list[str] = Field(default_factory=lambda: list(DEFAULT_GLOSSARY))


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
