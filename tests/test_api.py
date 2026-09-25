import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app import api as api_module
from app.config import ConferenceConfig, FileSourceConfig, ReplaySourceConfig, StageConfig
from app.events import CaptionFinalEvent, EventTiming
from app.store import JsonlEventStore
from app.supervisor import StageSupervisor
from app.vtt import write_vtt_file as real_write_vtt_file


class FakeTranslator:
    """Keeps StageSupervisor construction network-free (mirrors test_supervisor.py)."""

    async def translate(self, text, source_language, target_language):
        return text


def make_supervisor(tmp_path: Path, stages) -> StageSupervisor:
    conference = ConferenceConfig(stages=stages)
    # start_all() is never called: these tests only exercise routes that
    # read `stage_configs`, not a running pipeline.
    return StageSupervisor(
        conference, api_key="unused", store=JsonlEventStore(base_dir=tmp_path), translator=FakeTranslator()
    )


@pytest.fixture
def client(monkeypatch):
    """A TestClient over the real app, without running its `lifespan` (which
    would need a real GEMINI_API_KEY and load the real on-disk config) —
    confirmed that plain `TestClient(app)` (no `with` block) never triggers
    `lifespan`, so `api_module.supervisor` stays whatever each test sets it
    to via monkeypatch."""
    return TestClient(api_module.app)


def test_file_source_stage_audio_returns_the_configured_file(tmp_path, monkeypatch, client):
    audio_path = tmp_path / "main.wav"
    audio_path.write_bytes(b"RIFF-fake-wav-bytes-for-testing")
    stage = StageConfig(
        id="main", name="Main", language="en", targets=[],
        source=FileSourceConfig(type="file", path=audio_path),
    )
    monkeypatch.setattr(api_module, "supervisor", make_supervisor(tmp_path, [stage]))

    response = client.get("/api/stages/main/audio")

    assert response.status_code == 200
    assert response.content == audio_path.read_bytes()
    assert response.headers["content-type"] == "audio/wav"


def test_unknown_stage_returns_404(tmp_path, monkeypatch, client):
    monkeypatch.setattr(api_module, "supervisor", make_supervisor(tmp_path, []))

    response = client.get("/api/stages/does-not-exist/audio")

    assert response.status_code == 404


def test_replay_source_stage_has_no_audio_endpoint(tmp_path, monkeypatch, client):
    fixture_path = tmp_path / "devroom.json"
    fixture_path.write_text("[]")
    stage = StageConfig(
        id="devroom", name="Dev Room", language="es", targets=[],
        source=ReplaySourceConfig(type="replay", path=fixture_path),
    )
    monkeypatch.setattr(api_module, "supervisor", make_supervisor(tmp_path, [stage]))

    response = client.get("/api/stages/devroom/audio")

    assert response.status_code == 404
    # The audience UI must not be told to render a player for it either.
    health = client.get("/health").json()
    assert "devroom" not in health["audio_url"]


def test_audio_endpoint_ignores_client_supplied_path_and_only_serves_the_configured_file(
    tmp_path, monkeypatch, client
):
    """Regression guard: there is no `path`-style query parameter to attack —
    `stage_id` only ever selects among server-configured stages, and the
    served file always comes from that stage's own `FileSourceConfig`."""
    audio_path = tmp_path / "main.wav"
    audio_path.write_bytes(b"expected-bytes-only")
    secret_path = tmp_path / "secret.txt"
    secret_path.write_text("must never be served")
    stage = StageConfig(
        id="main", name="Main", language="en", targets=[],
        source=FileSourceConfig(type="file", path=audio_path),
    )
    monkeypatch.setattr(api_module, "supervisor", make_supervisor(tmp_path, [stage]))

    response = client.get("/api/stages/main/audio", params={"path": str(secret_path)})
    assert response.status_code == 200
    assert response.content == audio_path.read_bytes()  # the query param had zero effect

    # A path-traversal-shaped stage_id can't escape the single `{stage_id}`
    # path segment either — it simply fails the stage lookup and 404s.
    traversal_response = client.get("/api/stages/..%2f..%2fetc%2fpasswd/audio")
    assert traversal_response.status_code == 404


def test_health_exposes_a_distinct_audio_url_per_file_source_stage(tmp_path, monkeypatch, client):
    """The audience UI decides whether to render a stage's <audio> player
    entirely from this field — one real regression would be two stages
    accidentally sharing a URL, or a replay stage getting one at all."""
    main_audio = tmp_path / "main.wav"
    main_audio.write_bytes(b"main")
    devroom_fixture = tmp_path / "devroom.json"
    devroom_fixture.write_text("[]")
    stages = [
        StageConfig(
            id="main", name="Main", language="en", targets=[],
            source=FileSourceConfig(type="file", path=main_audio),
        ),
        StageConfig(
            id="devroom", name="Dev Room", language="es", targets=[],
            source=ReplaySourceConfig(type="replay", path=devroom_fixture),
        ),
    ]
    monkeypatch.setattr(api_module, "supervisor", make_supervisor(tmp_path, stages))

    audio_url = client.get("/health").json()["audio_url"]

    assert audio_url == {"main": "/api/stages/main/audio"}


def _with_vtt_written_to(monkeypatch, vtt_dir: Path):
    """The endpoint's default write target is data/vtt (relative to the
    process cwd) — redirect it to an isolated tmp_path directory for tests
    by wrapping the real write_vtt_file with a fixed base_dir, rather than
    monkeypatching VTT_OUTPUT_DIR (which wouldn't work: it's only read once,
    as write_vtt_file's default argument, at import time)."""
    monkeypatch.setattr(
        api_module,
        "write_vtt_file",
        lambda stage_id, content: real_write_vtt_file(stage_id, content, base_dir=vtt_dir),
    )


def _stage_with_one_final(store, stage_id, text, audio_elapsed_ms):
    store.append(
        stage_id,
        CaptionFinalEvent(
            stage_id=stage_id,
            seg_id=f"{stage_id}-000001",
            text=text,
            language="en",
            timing=EventTiming(audio_elapsed_ms=audio_elapsed_ms, asr_latency_ms=5.0),
        ),
    )


def test_captions_vtt_endpoint_persists_a_snapshot_to_disk(tmp_path, monkeypatch, client):
    vtt_dir = tmp_path / "vtt_out"
    _with_vtt_written_to(monkeypatch, vtt_dir)
    stage = StageConfig(
        id="main", name="Main", language="en", targets=[],
        source=FileSourceConfig(type="file", path=tmp_path / "unused.wav"),
    )
    supervisor = make_supervisor(tmp_path, [stage])
    supervisor.pipelines["main"] = object()  # only membership is checked by this endpoint
    _stage_with_one_final(supervisor.store, "main", "Hello.", 1000.0)
    monkeypatch.setattr(api_module, "supervisor", supervisor)

    response = client.get("/api/stages/main/captions.vtt")

    assert response.status_code == 200
    written = (vtt_dir / "main.vtt").read_text(encoding="utf-8")
    assert written == response.text  # the snapshot matches exactly what was returned
    assert "Hello." in written


def test_captions_vtt_endpoint_response_is_unaffected_by_a_stale_snapshot(tmp_path, monkeypatch, client):
    """The response must always reflect the current event store, never a
    stale file left over from a previous call."""
    vtt_dir = tmp_path / "vtt_out"
    vtt_dir.mkdir()
    (vtt_dir / "main.vtt").write_text("WEBVTT\n\nSTALE CONTENT\n", encoding="utf-8")
    _with_vtt_written_to(monkeypatch, vtt_dir)
    stage = StageConfig(
        id="main", name="Main", language="en", targets=[],
        source=FileSourceConfig(type="file", path=tmp_path / "unused.wav"),
    )
    supervisor = make_supervisor(tmp_path, [stage])
    supervisor.pipelines["main"] = object()
    _stage_with_one_final(supervisor.store, "main", "Fresh.", 1000.0)
    monkeypatch.setattr(api_module, "supervisor", supervisor)

    response = client.get("/api/stages/main/captions.vtt")

    assert "Fresh." in response.text
    assert "STALE CONTENT" not in response.text


def test_captions_vtt_endpoint_empty_stage_is_not_an_error(tmp_path, monkeypatch, client):
    vtt_dir = tmp_path / "vtt_out"
    _with_vtt_written_to(monkeypatch, vtt_dir)
    stage = StageConfig(
        id="main", name="Main", language="en", targets=[],
        source=FileSourceConfig(type="file", path=tmp_path / "unused.wav"),
    )
    supervisor = make_supervisor(tmp_path, [stage])
    supervisor.pipelines["main"] = object()
    monkeypatch.setattr(api_module, "supervisor", supervisor)

    response = client.get("/api/stages/main/captions.vtt")

    assert response.status_code == 200
    assert response.text == "WEBVTT\n"
    assert (vtt_dir / "main.vtt").read_text(encoding="utf-8") == "WEBVTT\n"


def test_captions_vtt_endpoint_keeps_two_stages_in_separate_files(tmp_path, monkeypatch, client):
    vtt_dir = tmp_path / "vtt_out"
    _with_vtt_written_to(monkeypatch, vtt_dir)
    stages = [
        StageConfig(id="main", name="Main", language="en", targets=[],
                    source=FileSourceConfig(type="file", path=tmp_path / "main.wav")),
        StageConfig(id="devroom", name="Dev Room", language="es", targets=[],
                    source=FileSourceConfig(type="file", path=tmp_path / "devroom.wav")),
    ]
    supervisor = make_supervisor(tmp_path, stages)
    supervisor.pipelines["main"] = object()
    supervisor.pipelines["devroom"] = object()
    _stage_with_one_final(supervisor.store, "main", "Main text.", 1000.0)
    _stage_with_one_final(supervisor.store, "devroom", "Devroom text.", 1000.0)
    monkeypatch.setattr(api_module, "supervisor", supervisor)

    client.get("/api/stages/main/captions.vtt")
    client.get("/api/stages/devroom/captions.vtt")

    main_vtt = (vtt_dir / "main.vtt").read_text(encoding="utf-8")
    devroom_vtt = (vtt_dir / "devroom.vtt").read_text(encoding="utf-8")
    assert "Main text." in main_vtt and "Devroom text." not in main_vtt
    assert "Devroom text." in devroom_vtt and "Main text." not in devroom_vtt


def test_captions_vtt_endpoint_unknown_stage_writes_nothing(tmp_path, monkeypatch, client):
    vtt_dir = tmp_path / "vtt_out"
    _with_vtt_written_to(monkeypatch, vtt_dir)
    monkeypatch.setattr(api_module, "supervisor", make_supervisor(tmp_path, []))

    response = client.get("/api/stages/does-not-exist/captions.vtt")
    traversal_response = client.get("/api/stages/..%2f..%2fetc%2fpasswd/captions.vtt")

    assert response.status_code == 404
    assert traversal_response.status_code == 404
    # The unknown-stage check happens before any file is ever touched.
    assert not vtt_dir.exists()
