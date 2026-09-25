from typing import Optional

from pydantic import BaseModel


class LatencySummary(BaseModel):
    count: int
    first_ms: Optional[float] = None
    min_ms: Optional[float] = None
    avg_ms: Optional[float] = None
    max_ms: Optional[float] = None


class LatencySampler:
    """Minimal running min/avg/max/first over a stream of latency samples."""

    def __init__(self) -> None:
        self._count = 0
        self._sum_ms = 0.0
        self._first_ms: Optional[float] = None
        self._min_ms: Optional[float] = None
        self._max_ms: Optional[float] = None

    def record(self, latency_ms: float) -> None:
        if self._first_ms is None:
            self._first_ms = latency_ms
        self._count += 1
        self._sum_ms += latency_ms
        self._min_ms = latency_ms if self._min_ms is None else min(self._min_ms, latency_ms)
        self._max_ms = latency_ms if self._max_ms is None else max(self._max_ms, latency_ms)

    def summary(self) -> LatencySummary:
        avg_ms = self._sum_ms / self._count if self._count else None
        return LatencySummary(
            count=self._count,
            first_ms=self._first_ms,
            min_ms=self._min_ms,
            avg_ms=avg_ms,
            max_ms=self._max_ms,
        )


class StageLatencyStats:
    """Per-stage ASR latency stats, split by interim vs final events."""

    def __init__(self) -> None:
        self.interim = LatencySampler()
        self.final = LatencySampler()

    def record_interim(self, latency_ms: Optional[float]) -> None:
        if latency_ms is not None:
            self.interim.record(latency_ms)

    def record_final(self, latency_ms: Optional[float]) -> None:
        if latency_ms is not None:
            self.final.record(latency_ms)

    def summary(self) -> dict:
        return {
            "interim": self.interim.summary().model_dump(),
            "final": self.final.summary().model_dump(),
        }
