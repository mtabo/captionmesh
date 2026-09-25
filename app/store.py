from pathlib import Path

from pydantic import BaseModel


class JsonlEventStore:
    """Persists stage events to data/stages/<stage_id>.jsonl."""

    def __init__(self, base_dir: Path = Path("data/stages")) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def append(self, stage_id: str, event: BaseModel) -> None:
        path = self._base_dir / f"{stage_id}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json())
            f.write("\n")
