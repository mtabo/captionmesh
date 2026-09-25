import asyncio
import json
import sys
from pathlib import Path

import pytest

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


class FakeSegmenter:
    """A SegmentationProvider stub — keeps supervisor tests network-free.
    Never splits, so pre-existing seg_id/behavior assertions in these tests
    (written before segmentation existed) stay valid unchanged."""

    async def segment(self, text: str) -> list[str]:
        return [text]


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
        conference, api_key="unused", store=JsonlEventStore(base_dir=tmp_path),
        translator=FakeTranslator(), segmenter=FakeSegmenter()
    )

    supervisor.start_all()

    assert set(supervisor.pipelines.keys()) == {"main", "devroom"}
    await supervisor.stop_all()


async def test_both_pipelines_run_concurrently_and_reach_stopped(tmp_path):
    conference = make_two_stage_conference(tmp_path)
    supervisor = StageSupervisor(
        conference, api_key="unused", store=JsonlEventStore(base_dir=tmp_path),
        translator=FakeTranslator(), segmenter=FakeSegmenter()
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
        segmenter=FakeSegmenter(),
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
        conference, api_key="unused", store=JsonlEventStore(base_dir=tmp_path),
        translator=FakeTranslator(), segmenter=FakeSegmenter()
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
        conference, api_key="unused", store=JsonlEventStore(base_dir=tmp_path),
        translator=FakeTranslator(), segmenter=FakeSegmenter()
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
        conference, api_key="unused", store=JsonlEventStore(base_dir=tmp_path),
        translator=slow_translator, segmenter=FakeSegmenter()
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
        conference, api_key="unused", store=JsonlEventStore(base_dir=tmp_path),
        translator=FakeTranslator(), segmenter=FakeSegmenter()
    )

    supervisor.start_all()
    await asyncio.sleep(0.1)  # let both pipelines actually start running

    await asyncio.wait_for(supervisor.stop_all(), timeout=2.0)

    assert all(task.done() for task in supervisor._tasks)


# --- restart_stage() ---

async def test_restart_stage_cancels_the_previous_still_running_task(tmp_path):
    long_fixture = [{"delay_ms": 5000, "text": "late.", "is_final": True, "language": "en"}]
    conference = ConferenceConfig(stages=[make_replay_stage(tmp_path, "main", long_fixture)])
    supervisor = StageSupervisor(
        conference, api_key="unused", store=JsonlEventStore(base_dir=tmp_path),
        translator=FakeTranslator(), segmenter=FakeSegmenter(),
    )
    supervisor.start_all()
    await asyncio.sleep(0.05)  # let it actually start running
    old_task = supervisor._tasks_by_stage["main"]
    assert not old_task.done()

    await supervisor.restart_stage("main")

    assert old_task.done()
    assert old_task.cancelled()


async def test_restart_stage_creates_a_new_pipeline_instance_that_runs_to_completion(tmp_path):
    conference = ConferenceConfig(stages=[make_replay_stage(tmp_path, "main", SEGMENTS_MAIN)])
    supervisor = StageSupervisor(
        conference, api_key="unused", store=JsonlEventStore(base_dir=tmp_path),
        translator=FakeTranslator(), segmenter=FakeSegmenter(),
    )
    supervisor.start_all()
    await asyncio.wait_for(asyncio.gather(*supervisor._tasks), timeout=2.0)
    old_pipeline = supervisor.pipelines["main"]
    assert old_pipeline.status == "stopped"

    new_pipeline = await supervisor.restart_stage("main")
    await asyncio.wait_for(asyncio.gather(supervisor._tasks_by_stage["main"]), timeout=2.0)

    assert new_pipeline is not old_pipeline
    assert supervisor.pipelines["main"] is new_pipeline
    assert new_pipeline.status == "stopped"


async def test_restart_stage_uses_the_same_unmodified_config(tmp_path):
    conference = ConferenceConfig(stages=[make_replay_stage(tmp_path, "main", SEGMENTS_MAIN, targets=["es"])])
    supervisor = StageSupervisor(
        conference, api_key="unused", store=JsonlEventStore(base_dir=tmp_path),
        translator=FakeTranslator(), segmenter=FakeSegmenter(),
    )
    supervisor.start_all()
    await asyncio.wait_for(asyncio.gather(*supervisor._tasks), timeout=2.0)

    new_pipeline = await supervisor.restart_stage("main")
    await asyncio.wait_for(asyncio.gather(supervisor._tasks_by_stage["main"]), timeout=2.0)

    assert new_pipeline._config is conference.stages[0]  # same StageConfig object, untouched
    assert new_pipeline._config.targets == ["es"]


async def test_restart_unknown_stage_raises_keyerror(tmp_path):
    conference = ConferenceConfig(stages=[make_replay_stage(tmp_path, "main", SEGMENTS_MAIN)])
    supervisor = StageSupervisor(
        conference, api_key="unused", store=JsonlEventStore(base_dir=tmp_path),
        translator=FakeTranslator(), segmenter=FakeSegmenter(),
    )
    supervisor.start_all()

    with pytest.raises(KeyError):
        await supervisor.restart_stage("does-not-exist")


async def test_restart_stage_does_not_delete_prior_jsonl_history(tmp_path):
    conference = ConferenceConfig(stages=[make_replay_stage(tmp_path, "main", SEGMENTS_MAIN)])
    store = JsonlEventStore(base_dir=tmp_path)
    supervisor = StageSupervisor(
        conference, api_key="unused", store=store, translator=FakeTranslator(), segmenter=FakeSegmenter(),
    )
    supervisor.start_all()
    await asyncio.wait_for(asyncio.gather(*supervisor._tasks), timeout=2.0)
    first_run_events = len(store.read_events("main"))
    assert first_run_events > 0

    await supervisor.restart_stage("main")
    await asyncio.wait_for(asyncio.gather(supervisor._tasks_by_stage["main"]), timeout=2.0)

    # The new session's events were appended, not replacing the old ones.
    assert len(store.read_events("main")) == first_run_events * 2


async def test_two_successive_restarts_write_separate_timestamped_vtt_files(tmp_path):
    conference = ConferenceConfig(stages=[make_replay_stage(tmp_path, "main", SEGMENTS_MAIN)])
    vtt_dir = tmp_path / "vtt"
    supervisor = StageSupervisor(
        conference, api_key="unused", store=JsonlEventStore(base_dir=tmp_path / "stages"),
        translator=FakeTranslator(), segmenter=FakeSegmenter(), vtt_output_dir=vtt_dir,
    )
    supervisor.start_all()
    await asyncio.wait_for(asyncio.gather(*supervisor._tasks), timeout=2.0)

    await asyncio.sleep(1.1)  # ensure the next write lands in a different YYYYMMDD-HHMMSS second
    await supervisor.restart_stage("main")
    await asyncio.wait_for(asyncio.gather(supervisor._tasks_by_stage["main"]), timeout=2.0)

    written = sorted(vtt_dir.glob("main_*.vtt"))
    assert len(written) == 2
    assert written[0] != written[1]


async def test_restart_stage_does_not_hang_when_previous_task_ignores_cancellation(tmp_path):
    """Regression test for a real hang found in live testing: a task stuck
    inside a Gemini send()/drain() call can catch CancelledError and keep
    awaiting (e.g. blocked on kernel-level TCP flow control), so cancelling
    it does not guarantee it stops. restart_stage() must give up waiting
    after `restart_cancel_grace_seconds` and start the new session anyway,
    rather than hang the request forever."""

    async def uncancellable():
        # Swallows exactly the one cancellation restart_stage() itself sends
        # (simulating a call stuck on something cancellation alone can't
        # interrupt in time, e.g. real kernel-level TCP flow control), then
        # actually dies on a second cancel() so the test can still clean up
        # afterwards without hanging pytest-asyncio's own teardown.
        survived_once = False
        while True:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                if survived_once:
                    raise
                survived_once = True

    conference = ConferenceConfig(stages=[make_replay_stage(tmp_path, "main", SEGMENTS_MAIN)])
    supervisor = StageSupervisor(
        conference, api_key="unused", store=JsonlEventStore(base_dir=tmp_path),
        translator=FakeTranslator(), segmenter=FakeSegmenter(),
        restart_cancel_grace_seconds=0.1,
    )
    stuck_task = asyncio.create_task(uncancellable())
    supervisor._tasks_by_stage["main"] = stuck_task
    supervisor._tasks.append(stuck_task)
    await asyncio.sleep(0)  # let it actually reach its first `await` before cancelling it

    # The bug this guards against: without shielding, waiting on a task that
    # never actually finishes cancelling hangs forever, regardless of any
    # timeout wrapped around it — so the outer bound here is the real test.
    new_pipeline = await asyncio.wait_for(supervisor.restart_stage("main"), timeout=1.0)

    assert new_pipeline is supervisor.pipelines["main"]
    assert not stuck_task.done()  # confirms it really did ignore the first cancellation

    stuck_task.cancel()  # clean up: this second cancel is the one it actually respects
    with pytest.raises(asyncio.CancelledError):
        await stuck_task
