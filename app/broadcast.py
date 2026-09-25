import asyncio

from app.events import StageEvent

DEFAULT_QUEUE_SIZE = 100


class Broadcaster:
    """Fans out stage events to subscribed audience clients.

    Each subscriber gets a bounded queue; if a slow consumer falls behind,
    the oldest queued event is dropped in favor of the newest one rather
    than growing an unbounded backlog.
    """

    def __init__(self, queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        self._queue_size = queue_size
        self._subscribers: dict[str, set[asyncio.Queue]] = {}

    def subscribe(self, stage_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.setdefault(stage_id, set()).add(queue)
        return queue

    def unsubscribe(self, stage_id: str, queue: asyncio.Queue) -> None:
        subscribers = self._subscribers.get(stage_id)
        if not subscribers:
            return
        subscribers.discard(queue)
        if not subscribers:
            del self._subscribers[stage_id]

    def publish(self, event: StageEvent) -> None:
        for queue in list(self._subscribers.get(event.stage_id, ())):
            self._put_dropping_oldest(queue, event)

    @staticmethod
    def _put_dropping_oldest(queue: asyncio.Queue, event: StageEvent) -> None:
        try:
            queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            pass
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass
