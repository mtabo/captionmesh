import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.broadcast import Broadcaster
from app.config import ConferenceConfig, ReplaySourceConfig, StageConfig
from app.store import JsonlEventStore
from app.supervisor import StageSupervisor


class FakeTranslator:
    """A TranslationProvider stub — keeps supervisor tests network-free."""

    def __init__(self, delay: float = 0.0):
        self.calls = []
        self._delay = delay

    async def translate(self, text: str, source_language: str, target_language: str) -> str:
        self.calls.append((text, source_language, target_language))
        if self._delay:
            await asyncio.sleep(self._delay)
        return f"[{target_language}] {text}"


SEGMENTS_MAIN = [
    {"delay_ms": 0, "text": "Hi", "is_final": False},
    {"delay_ms": 0, "text": "Hi.", "is_final": True, "language": "en"},
]
SEGMENTS_DEVROOM = [
    {"delay_ms": 0, "text": "Hola", "is_final": False},
    {"delay_ms": 0, "text": "Hola.", "is_final": True, "language": "es"},
]


def make_replay_stage(tmp_path: Path, stage_id: str, entries, targets=()) -> StageConfig:
    fixture_path = tmp_path / f"{stage_id}-fixture.json"
    fixture_path.write_text(json.dumps(entries))
    return StageConfig(
        id=stage_id,
        name=stage_id,
        language="en",
        targets=list(targets),
        source=ReplaySourceConfig(type="replay", path=fixture_path),
    )


def make_two_stage_conference(tmp_path: Path, with_targets: bool = False) -> ConferenceConfig:
    return ConferenceConfig(
        stages=[
            make_replay_stage(tmp_path, "main", SEGMENTS_MAIN, targets=["es"] if with_targets else ()),
            make_replay_stage(tmp_path, "devroom", SEGMENTS_DEVROOM, targets=["en"] if with_targets else ()),
        ]
    )


async def test_two_stages_can_be_created_from_configuration(tmp_path):
    conference = make_two_stage_conference(tmp_path)
    supervisor = StageSupervisor(
        conference, api_key="unused", store=JsonlEventStore(base_dir=tmp_path), translator=FakeTranslator()
    )

    supervisor.start_all()

    assert set(supervisor.pipelines.keys()) == {"main", "devroom"}
    await supervisor.stop_all()


async def test_both_pipelines_run_concurrently_and_reach_stopped(tmp_path):
    conference = make_two_stage_conference(tmp_path)
    supervisor = StageSupervisor(
        conference, api_key="unused", store=JsonlEventStore(base_dir=tmp_path), translator=FakeTranslator()
    )

    supervisor.start_all()
    await asyncio.wait_for(asyncio.gather(*supervisor._tasks), timeout=2.0)

    assert supervisor.pipelines["main"].status == "stopped"
    assert supervisor.pipelines["devroom"].status == "stopped"


async def test_stage_events_never_cross_between_stages(tmp_path):
    conference = make_two_stage_conference(tmp_path)
    broadcaster = Broadcaster()
    main_queue = broadcaster.subscribe("main")
    devroom_queue = broadcaster.subscribe("devroom")

    supervisor = StageSupervisor(
        conference,
        api_key="unused",
        broadcaster=broadcaster,
        store=JsonlEventStore(base_dir=tmp_path),
        translator=FakeTranslator(),
    )
    supervisor.start_all()
    await asyncio.wait_for(asyncio.gather(*supervisor._tasks), timeout=2.0)

    def drain(queue):
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        return events

    main_events = drain(main_queue)
    devroom_events = drain(devroom_queue)

    assert main_events and devroom_events  # sanity: both actually produced something
    assert all(e.stage_id == "main" for e in main_events)
    assert all(e.stage_id == "devroom" for e in devroom_events)
    assert any(e.text == "Hi." for e in main_events)
    assert any(e.text == "Hola." for e in devroom_events)
    assert not any(e.text == "Hola." for e in main_events)
    assert not any(e.text == "Hi." for e in devroom_events)


async def test_seg_id_sequences_are_independent_per_stage(tmp_path):
    conference = make_two_stage_conference(tmp_path)
    supervisor = StageSupervisor(
        conference, api_key="unused", store=JsonlEventStore(base_dir=tmp_path), translator=FakeTranslator()
    )

    supervisor.start_all()
    await asyncio.wait_for(asyncio.gather(*supervisor._tasks), timeout=2.0)

    main_final = json.loads((tmp_path / "main.jsonl").read_text().splitlines()[0])
    devroom_final = json.loads((tmp_path / "devroom.jsonl").read_text().splitlines()[0])

    assert main_final["seg_id"] == "main-000001"
    assert devroom_final["seg_id"] == "devroom-000001"


async def test_each_stage_persists_independently(tmp_path):
    conference = make_two_stage_conference(tmp_path)
    supervisor = StageSupervisor(
        conference, api_key="unused", store=JsonlEventStore(base_dir=tmp_path), translator=FakeTranslator()
    )

    supervisor.start_all()
    await asyncio.wait_for(asyncio.gather(*supervisor._tasks), timeout=2.0)

    assert (tmp_path / "main.jsonl").exists()
    assert (tmp_path / "devroom.jsonl").exists()

    main_lines = (tmp_path / "main.jsonl").read_text().splitlines()
    devroom_lines = (tmp_path / "devroom.jsonl").read_text().splitlines()
    assert all(json.loads(line)["stage_id"] == "main" for line in main_lines)
    assert all(json.loads(line)["stage_id"] == "devroom" for line in devroom_lines)


async def test_a_slow_translation_in_one_stage_does_not_block_the_other(tmp_path):
    conference = make_two_stage_conference(tmp_path, with_targets=True)
    slow_translator = FakeTranslator(delay=999)  # deliberately never resolves in time
    supervisor = StageSupervisor(
        conference, api_key="unused", store=JsonlEventStore(base_dir=tmp_path), translator=slow_translator
    )

    supervisor.start_all()
    for pipeline in supervisor.pipelines.values():
        pipeline._translation_shutdown_grace_seconds = 0.05  # keep the test fast

    # Both stages' ASR/final processing (replay is near-instant) must complete
    # and reach "stopped" even though every translation call hangs — proving
    # a slow translation in either stage cannot block the other's pipeline.
    await asyncio.wait_for(asyncio.gather(*supervisor._tasks), timeout=2.0)

    assert supervisor.pipelines["main"].status == "stopped"
    assert supervisor.pipelines["devroom"].status == "stopped"
    assert len(slow_translator.calls) >= 2  # both stages did attempt translation


async def test_stopping_the_supervisor_stops_all_pipelines_cleanly(tmp_path):
    long_fixture = [{"delay_ms": 50, "text": f"segment {i}", "is_final": False} for i in range(100)]
    conference = ConferenceConfig(
        stages=[
            make_replay_stage(tmp_path, "main", long_fixture),
            make_replay_stage(tmp_path, "devroom", SEGMENTS_DEVROOM),
        ]
    )
    supervisor = StageSupervisor(
        conference, api_key="unused", store=JsonlEventStore(base_dir=tmp_path), translator=FakeTranslator()
    )

    supervisor.start_all()
    await asyncio.sleep(0.1)  # let both pipelines actually start running

    await asyncio.wait_for(supervisor.stop_all(), timeout=2.0)

    assert all(task.done() for task in supervisor._tasks)
