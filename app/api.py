import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.config import load_conference_config
from app.providers.gemini_asr import GeminiTranscriber
from app.stage import StagePipeline
from app.store import JsonlEventStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CONFIG_PATH = Path(os.environ.get("CAPTIONMESH_CONFIG", "config/stages.yaml"))

pipelines: dict[str, StagePipeline] = {}
_tasks: list[asyncio.Task] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.environ["GEMINI_API_KEY"]
    conference = load_conference_config(CONFIG_PATH)
    store = JsonlEventStore()

    for stage_config in conference.stages:
        transcriber = GeminiTranscriber(api_key=api_key, language=stage_config.language)
        pipeline = StagePipeline(stage_config, transcriber, store)
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
    }
