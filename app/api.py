import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response

from app.config import load_conference_config
from app.supervisor import StageSupervisor
from app.vtt import build_vtt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CONFIG_PATH = Path(os.environ.get("CAPTIONMESH_CONFIG", "config/stages.yaml"))
STATIC_DIR = Path(__file__).parent / "static"

supervisor: Optional[StageSupervisor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global supervisor
    api_key = os.environ["GEMINI_API_KEY"]
    conference = load_conference_config(CONFIG_PATH)
    supervisor = StageSupervisor(conference, api_key=api_key)
    supervisor.start_all()

    yield

    await supervisor.stop_all()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    pipelines = supervisor.pipelines if supervisor else {}
    return {
        "status": "ok",
        "stages": {stage_id: p.status for stage_id, p in pipelines.items()},
        "latency": {stage_id: p.stats.summary() for stage_id, p in pipelines.items()},
        "pending_translations": {
            stage_id: p.pending_translation_count for stage_id, p in pipelines.items()
        },
    }


@app.get("/audience/{stage_id}")
async def audience_page(stage_id: str):
    return FileResponse(STATIC_DIR / "audience.html")


@app.get("/audience")
async def audience_multi_page():
    return FileResponse(STATIC_DIR / "audience_multi.html")


@app.get("/api/stages/{stage_id}/captions.vtt")
async def stage_captions_vtt(stage_id: str):
    pipelines = supervisor.pipelines if supervisor else {}
    if stage_id not in pipelines:
        raise HTTPException(status_code=404, detail="unknown stage")

    events = supervisor.store.read_events(stage_id)
    return Response(content=build_vtt(events), media_type="text/vtt")


@app.websocket("/ws/audience/{stage_id}")
async def audience_ws(websocket: WebSocket, stage_id: str):
    await websocket.accept()

    pipelines = supervisor.pipelines if supervisor else {}
    if stage_id not in pipelines:
        await websocket.close(code=4404, reason="unknown stage")
        return

    queue = supervisor.broadcaster.subscribe(stage_id)
    try:
        while True:
            event = await queue.get()
            await websocket.send_text(event.model_dump_json())
    except WebSocketDisconnect:
        pass
    finally:
        supervisor.broadcaster.unsubscribe(stage_id, queue)
