import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import load_conference_config


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
