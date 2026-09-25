import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from app.broadcast import Broadcaster
from app.config import load_conference_config
from app.providers.gemini_asr import GeminiTranscriber
from app.providers.gemini_translate import GeminiTranslator
from app.stage import StagePipeline
from app.store import JsonlEventStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CONFIG_PATH = Path(os.environ.get("CAPTIONMESH_CONFIG", "config/stages.yaml"))
STATIC_DIR = Path(__file__).parent / "static"

pipelines: dict[str, StagePipeline] = {}
broadcaster = Broadcaster()
_tasks: list[asyncio.Task] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.environ["GEMINI_API_KEY"]
    conference = load_conference_config(CONFIG_PATH)
    store = JsonlEventStore()
    # Stateless per call (no session), so one instance is shared across stages.
    translator = GeminiTranslator(api_key=api_key)

    for stage_config in conference.stages:
        transcriber = GeminiTranscriber(api_key=api_key, language=stage_config.language)
        pipeline = StagePipeline(stage_config, transcriber, store, broadcaster, translator)
        pipelines[stage_config.id] = pipeline
        _tasks.append(asyncio.create_task(pipeline.run()))

    yield

    for task in _tasks:
        task.cancel()
    await asyncio.gather(*_tasks, return_exceptions=True)


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "stages": {stage_id: pipeline.status for stage_id, pipeline in pipelines.items()},
        "latency": {stage_id: pipeline.stats.summary() for stage_id, pipeline in pipelines.items()},
        "pending_translations": {
            stage_id: pipeline.pending_translation_count for stage_id, pipeline in pipelines.items()
        },
    }


@app.get("/audience/{stage_id}")
async def audience_page(stage_id: str):
    return FileResponse(STATIC_DIR / "audience.html")


@app.websocket("/ws/audience/{stage_id}")
async def audience_ws(websocket: WebSocket, stage_id: str):
    await websocket.accept()

    if stage_id not in pipelines:
        await websocket.close(code=4404, reason="unknown stage")
        return

    queue = broadcaster.subscribe(stage_id)
    try:
        while True:
            event = await queue.get()
            await websocket.send_text(event.model_dump_json())
    except WebSocketDisconnect:
        pass
    finally:
        broadcaster.unsubscribe(stage_id, queue)
