import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.broadcast import Broadcaster
from app.events import CaptionInterimEvent


def make_event(stage_id: str = "main", seg_id: str = "main-000001", text: str = "hi"):
    return CaptionInterimEvent(stage_id=stage_id, seg_id=seg_id, text=text, language="en")


async def test_subscriber_receives_published_event_for_its_stage():
    broadcaster = Broadcaster()
    queue = broadcaster.subscribe("main")

    broadcaster.publish(make_event(stage_id="main"))

    event = queue.get_nowait()
    assert event.text == "hi"


async def test_subscriber_does_not_receive_events_for_other_stages():
    broadcaster = Broadcaster()
    queue = broadcaster.subscribe("main")

    broadcaster.publish(make_event(stage_id="other"))

    assert queue.empty()


async def test_unsubscribe_stops_further_delivery():
    broadcaster = Broadcaster()
    queue = broadcaster.subscribe("main")
    broadcaster.unsubscribe("main", queue)

    broadcaster.publish(make_event(stage_id="main"))

    assert queue.empty()


async def test_publish_with_no_subscribers_does_not_raise():
    broadcaster = Broadcaster()

    broadcaster.publish(make_event())  # must not raise


async def test_full_queue_drops_oldest_event_in_favor_of_newest():
    broadcaster = Broadcaster(queue_size=1)
    queue = broadcaster.subscribe("main")

    broadcaster.publish(make_event(text="first"))
    broadcaster.publish(make_event(text="second"))

    assert queue.qsize() == 1
    event = queue.get_nowait()
    assert event.text == "second"
