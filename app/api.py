import logging
import mimetypes
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response

from app.config import FileSourceConfig, load_conference_config
from app.supervisor import StageSupervisor
from app.vtt import VTT_OUTPUT_DIR, build_vtt, write_vtt_file

# .wav is the canonical demo/fixture audio format (see docs/spec.md); map it
# explicitly to the modern IANA type rather than trusting the local system's
# mimetypes database, which commonly guesses the legacy "audio/x-wav" for it.
# Anything else falls back to mimetypes' own guess.
_AUDIO_MEDIA_TYPES = {".wav": "audio/wav"}


def _guess_audio_media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(str(path))
    return _AUDIO_MEDIA_TYPES.get(path.suffix.lower()) or guessed or "application/octet-stream"


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CONFIG_PATH = Path(os.environ.get("CAPTIONMESH_CONFIG", "config/stages.yaml"))
STATIC_DIR = Path(__file__).parent / "static"

supervisor: Optional[StageSupervisor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global supervisor
    api_key = os.environ["GEMINI_API_KEY"]
    conference = load_conference_config(CONFIG_PATH)
    supervisor = StageSupervisor(conference, api_key=api_key, vtt_output_dir=VTT_OUTPUT_DIR)
    supervisor.start_all()

    yield

    await supervisor.stop_all()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    pipelines = supervisor.pipelines if supervisor else {}
    stage_configs = supervisor.stage_configs if supervisor else {}

    # Only stages backed by a real audio file are playable — a replay
    # stage has no underlying file, so it's simply omitted from both of
    # these rather than the audience UI inventing a player/position for it.
    audio_url = {}
    audio_position_ms = {}
    for stage_id, config in stage_configs.items():
        if not isinstance(config.source, FileSourceConfig):
            continue
        audio_url[stage_id] = f"/api/stages/{stage_id}/audio"
        pipeline = pipelines.get(stage_id)
        if pipeline is not None and pipeline.current_audio_position_ms is not None:
            audio_position_ms[stage_id] = pipeline.current_audio_position_ms

    return {
        "status": "ok",
        "stages": {stage_id: p.status for stage_id, p in pipelines.items()},
        "latency": {stage_id: p.stats.summary() for stage_id, p in pipelines.items()},
        "pending_translations": {
            stage_id: p.pending_translation_count for stage_id, p in pipelines.items()
        },
        "audio_url": audio_url,
        # Best-effort current position in that same audio, for the audience
        # player to sync to (see StagePipeline.current_audio_position_ms).
        "audio_position_ms": audio_position_ms,
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
    vtt_content = build_vtt(events)
    # `stage_id` was just validated against known stages above, so this is
    # safe to use as a filename (see write_vtt_file's own docstring). Kept
    # as a snapshot on disk purely for convenience post-demo; the response
    # below is always generated fresh from the event store, never from
    # this file, so a stale file on disk can never cause a stale response.
    write_vtt_file(stage_id, vtt_content)
    return Response(content=vtt_content, media_type="text/vtt")


@app.post("/api/stages/{stage_id}/restart")
async def restart_stage(stage_id: str):
    """Stops the stage's current session (if any) and starts a fresh one
    from its existing config — no FastAPI/container restart, no config
    change, no history deleted. See StageSupervisor.restart_stage."""
    stage_configs = supervisor.stage_configs if supervisor else {}
    if stage_id not in stage_configs:
        raise HTTPException(status_code=404, detail="unknown stage")

    pipeline = await supervisor.restart_stage(stage_id)
    return {"stage_id": stage_id, "status": pipeline.status}


@app.get("/api/stages/{stage_id}/audio")
async def stage_audio(stage_id: str):
    """Streams the exact audio file `FileAudioSource` is transcribing for
    this stage, so the audience UI can play back what's actually being
    captioned. `stage_id` only ever selects among the server's own
    configured stages (never a client-supplied path), so there is no way
    to reach an arbitrary filesystem path through this endpoint."""
    stage_configs = supervisor.stage_configs if supervisor else {}
    stage_config = stage_configs.get(stage_id)
    if stage_config is None:
        raise HTTPException(status_code=404, detail="unknown stage")
    if not isinstance(stage_config.source, FileSourceConfig):
        raise HTTPException(status_code=404, detail="stage has no playable audio source")

    path = stage_config.source.path
    return FileResponse(path, media_type=_guess_audio_media_type(path))


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
