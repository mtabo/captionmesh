from typing import AsyncIterator, Protocol


class AudioSource(Protocol):
    def stream(self) -> AsyncIterator[bytes]:
        ...
