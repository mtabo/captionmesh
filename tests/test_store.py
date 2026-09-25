import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.events import CaptionFinalEvent
from app.store import JsonlEventStore


def test_append_writes_one_json_line_per_event(tmp_path):
    store = JsonlEventStore(base_dir=tmp_path)
    event = CaptionFinalEvent(stage_id="main", seg_id="main-000001", text="Hello.", language="en")

    store.append("main", event)
    store.append("main", event)

    lines = (tmp_path / "main.jsonl").read_text().splitlines()
    assert len(lines) == 2

    parsed = json.loads(lines[0])
    assert parsed["type"] == "caption.final"
    assert parsed["stage_id"] == "main"
    assert parsed["seg_id"] == "main-000001"
    assert parsed["text"] == "Hello."
    assert parsed["language"] == "en"


def test_append_scopes_events_by_stage_id(tmp_path):
    store = JsonlEventStore(base_dir=tmp_path)
    event_a = CaptionFinalEvent(stage_id="a", seg_id="a-000001", text="A", language="en")
    event_b = CaptionFinalEvent(stage_id="b", seg_id="b-000001", text="B", language="en")

    store.append("a", event_a)
    store.append("b", event_b)

    assert (tmp_path / "a.jsonl").exists()
    assert (tmp_path / "b.jsonl").exists()
