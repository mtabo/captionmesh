import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DEFAULT_GLOSSARY, load_conference_config


def test_load_conference_config_parses_stages(tmp_path):
    config_path = tmp_path / "stages.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            conference:
              name: Test Conference

            stages:
              - id: main
                name: Main Stage
                language: auto
                targets:
                  - es
                source:
                  type: file
                  path: data/audio/test-en.wav
            """
        )
    )

    conference = load_conference_config(config_path)

    assert conference.name == "Test Conference"
    assert len(conference.stages) == 1

    stage = conference.stages[0]
    assert stage.id == "main"
    assert stage.language == "auto"
    assert stage.targets == ["es"]
    assert stage.source.type == "file"
    assert stage.source.path == Path("data/audio/test-en.wav")


def test_load_conference_config_parses_two_stages_with_mixed_source_types(tmp_path):
    config_path = tmp_path / "stages.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            conference:
              name: Nerdearla 2026

            stages:
              - id: main
                name: Main Stage
                language: auto
                targets:
                  - es
                source:
                  type: file
                  path: data/audio/main.wav

              - id: devroom
                name: Dev Room
                language: es
                targets:
                  - en
                source:
                  type: replay
                  path: data/replay/devroom.json
            """
        )
    )

    conference = load_conference_config(config_path)

    assert [s.id for s in conference.stages] == ["main", "devroom"]

    main, devroom = conference.stages
    assert main.source.type == "file"
    assert main.source.path == Path("data/audio/main.wav")

    assert devroom.language == "es"
    assert devroom.targets == ["en"]
    assert devroom.source.type == "replay"
    assert devroom.source.path == Path("data/replay/devroom.json")


def test_language_defaults_to_auto_when_omitted(tmp_path):
    config_path = tmp_path / "stages.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            stages:
              - id: main
                name: Main Stage
                source:
                  type: file
                  path: data/audio/test-en.wav
            """
        )
    )

    conference = load_conference_config(config_path)

    assert conference.stages[0].language == "auto"
    assert conference.stages[0].targets == []


def test_glossary_defaults_when_gemini_section_omitted(tmp_path):
    config_path = tmp_path / "stages.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            stages:
              - id: main
                name: Main Stage
                source:
                  type: file
                  path: data/audio/test-en.wav
            """
        )
    )

    conference = load_conference_config(config_path)

    assert conference.gemini.glossary == DEFAULT_GLOSSARY
    assert "Firebase" in conference.gemini.glossary
    assert "Kubernetes" in conference.gemini.glossary


def test_glossary_can_be_overridden_from_yaml(tmp_path):
    config_path = tmp_path / "stages.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            gemini:
              glossary:
                - Nerdearla
                - CaptionMesh

            stages:
              - id: main
                name: Main Stage
                source:
                  type: file
                  path: data/audio/test-en.wav
            """
        )
    )

    conference = load_conference_config(config_path)

    assert conference.gemini.glossary == ["Nerdearla", "CaptionMesh"]


def test_glossary_can_be_disabled_with_an_empty_list(tmp_path):
    config_path = tmp_path / "stages.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            gemini:
              glossary: []

            stages:
              - id: main
                name: Main Stage
                source:
                  type: file
                  path: data/audio/test-en.wav
            """
        )
    )

    conference = load_conference_config(config_path)

    assert conference.gemini.glossary == []
