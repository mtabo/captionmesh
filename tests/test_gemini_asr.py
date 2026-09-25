import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.broadcast import Broadcaster
from app.config import FileSourceConfig, StageConfig
from app.providers import gemini_asr as gemini_asr_module
from app.providers.gemini_asr import GeminiTranscriber
from app.stage import StagePipeline
from app.store import JsonlEventStore


class FakeTranscription:
    def __init__(self, text, finished=None, language_code=None):
        self.text = text
        self.finished = finished
        self.language_code = language_code


class FakeServerContent:
    def __init__(self, interim=None, final=None, turn_complete=False):
        self.interim_input_transcription = interim
        self.input_transcription = final
        self.turn_complete = turn_complete


class FakeMessage:
    def __init__(self, server_content):
        self.server_content = server_content


class FakeSession:
    """Mimics enough of google.genai's Live session shape for GeminiTranscriber.

    `timed_messages`: (delay_s, FakeMessage) pairs, yielded in order.
    After exhausting them, receive() waits for `audio_stream_end` to be sent
    (mirroring the real server: the connection stays open for as long as we
    keep sending) then closes naturally — unless `abrupt_disconnect_after`
    is set, in which case it raises instead, regardless of stream-end state,
    simulating an unexpected mid-session drop.
    """

    def __init__(self, timed_messages=(), abrupt_disconnect_after=None, close_error=None, close_delay=0.005):
        self._timed_messages = list(timed_messages)
        self._abrupt_disconnect_after = abrupt_disconnect_after
        self._close_error = close_error
        self._close_delay = close_delay
        self._stream_end_event = asyncio.Event()
        self.sent_chunks = []
        self.stream_ended = False

    async def send_realtime_input(self, audio=None, audio_stream_end=None):
        if audio is not None:
            self.sent_chunks.append(audio.data)
        if audio_stream_end:
            self.stream_ended = True
            self._stream_end_event.set()

    async def receive(self):
        for delay_s, message in self._timed_messages:
            if delay_s:
                await asyncio.sleep(delay_s)
            yield message

        if self._abrupt_disconnect_after is not None:
            await asyncio.sleep(self._abrupt_disconnect_after)
            raise ConnectionError("simulated abrupt disconnect")

        await self._stream_end_event.wait()
        if self._close_delay:
            await asyncio.sleep(self._close_delay)
        if self._close_error is not None:
            raise self._close_error


class FakeConnectCM:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        return False


def make_connect(sessions_or_errors):
    """A `_connect` replacement: yields one fake session (or raises) per call, in order."""
    it = iter(sessions_or_errors)

    def _connect():
        item = next(it)
        if isinstance(item, BaseException):
            raise item
        return FakeConnectCM(item)

    return _connect


async def make_audio_source(n_chunks=20, chunk_bytes=b"\x00" * 32, delay=0.002):
    for _ in range(n_chunks):
        if delay:
            await asyncio.sleep(delay)
        yield chunk_bytes


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    """Every test in this module uses a millisecond-scale backoff schedule
    instead of production's (1, 2, 4, 8)s — same shape, fast to run."""
    monkeypatch.setattr(gemini_asr_module, "RECONNECT_BACKOFF_SCHEDULE", (0.001, 0.001, 0.001, 0.001))


async def test_rotation_occurs_before_configured_lifetime_and_a_new_session_continues():
    session1 = FakeSession(
        timed_messages=[(0.01, FakeMessage(FakeServerContent(
            final=FakeTranscription("Hello.", finished=True, language_code="en")
        )))]
    )
    session2 = FakeSession(
        timed_messages=[(0.005, FakeMessage(FakeServerContent(
            final=FakeTranscription("World.", finished=True, language_code="en")
        )))]
    )
    transcriber = GeminiTranscriber(api_key="unused", session_rotation_seconds=0.05)
    transcriber._connect = make_connect([session1, session2])

    segments = [
        seg async for seg in transcriber.transcribe(make_audio_source(n_chunks=16, delay=0.005))
    ]

    assert [s.text for s in segments] == ["Hello.", "World."]
    assert [s.is_final for s in segments] == [True, True]
    assert session1.stream_ended is True
    assert session2.stream_ended is True


async def test_a_new_session_is_actually_created_on_rotation():
    session1 = FakeSession(timed_messages=[])
    session2 = FakeSession(timed_messages=[])
    connect_calls = []

    def _connect():
        item = session1 if not connect_calls else session2
        connect_calls.append(item)
        return FakeConnectCM(item)

    transcriber = GeminiTranscriber(api_key="unused", session_rotation_seconds=0.03)
    transcriber._connect = _connect

    # session1 rotates at ~0.03s (~3 chunks); session2 then only gets the
    # remaining 2 chunks (~0.02s) — safely under its own threshold, so it
    # exhausts naturally instead of requesting a third session.
    async for _ in transcriber.transcribe(make_audio_source(n_chunks=5, delay=0.01)):
        pass

    assert len(connect_calls) == 2
    assert connect_calls[0] is session1
    assert connect_calls[1] is session2


async def test_seg_id_sequence_remains_continuous_across_rotation(tmp_path):
    """StagePipeline integration: rotation must not restart seg numbering."""
    session1 = FakeSession(
        timed_messages=[(0.01, FakeMessage(FakeServerContent(
            final=FakeTranscription("One.", finished=True, language_code="en")
        )))]
    )
    session2 = FakeSession(
        timed_messages=[(0.005, FakeMessage(FakeServerContent(
            final=FakeTranscription("Two.", finished=True, language_code="en")
        )))]
    )
    transcriber = GeminiTranscriber(api_key="unused", session_rotation_seconds=0.05)
    transcriber._connect = make_connect([session1, session2])

    store = JsonlEventStore(base_dir=tmp_path)
    config = StageConfig(
        id="main", name="Main", language="en", targets=[],
        source=FileSourceConfig(type="file", path="unused.wav"),
    )
    pipeline = StagePipeline(config, transcriber, store, Broadcaster(), audio_source=_FakeAudioSourceWrapper())

    await pipeline.run()

    import json
    lines = (tmp_path / "main.jsonl").read_text().splitlines()
    events = [json.loads(line) for line in lines]
    assert [e["seg_id"] for e in events] == ["main-000001", "main-000002"]
    assert [e["text"] for e in events] == ["One.", "Two."]


class _FakeAudioSourceWrapper:
    """Adapts make_audio_source() to the AudioSource Protocol (a `.stream()` method)."""

    def stream(self):
        return make_audio_source(n_chunks=16, delay=0.005)


async def test_interim_does_not_become_an_artificial_final_on_rotation():
    """Session 1 only ever produces an interim (never finalized) before
    rotating. That interim must simply disappear — no fabricated final."""
    session1 = FakeSession(
        timed_messages=[(0.01, FakeMessage(FakeServerContent(
            interim=FakeTranscription("partial thought", language_code="en")
        )))]
    )
    session2 = FakeSession(
        timed_messages=[(0.005, FakeMessage(FakeServerContent(
            final=FakeTranscription("Complete.", finished=True, language_code="en")
        )))]
    )
    transcriber = GeminiTranscriber(api_key="unused", session_rotation_seconds=0.05)
    transcriber._connect = make_connect([session1, session2])

    segments = [
        seg async for seg in transcriber.transcribe(make_audio_source(n_chunks=16, delay=0.005))
    ]

    assert [(s.text, s.is_final) for s in segments] == [
        ("partial thought", False),
        ("Complete.", True),
    ]
    # No fabricated final for "partial thought" ever appears.
    assert not any(s.is_final and s.text == "partial thought" for s in segments)
    assert segments[1].session_boundary is True  # new session's first segment


async def test_transient_disconnect_triggers_reconnect_and_transcription_continues():
    session1 = FakeSession(abrupt_disconnect_after=0.01)  # dies before ever yielding anything
    session2 = FakeSession(
        timed_messages=[(0.005, FakeMessage(FakeServerContent(
            final=FakeTranscription("Recovered.", finished=True, language_code="en")
        )))]
    )
    transcriber = GeminiTranscriber(api_key="unused", session_rotation_seconds=999)
    transcriber._connect = make_connect([session1, session2])

    segments = [
        seg async for seg in transcriber.transcribe(make_audio_source(n_chunks=20, delay=0.002))
    ]

    assert [s.text for s in segments] == ["Recovered."]
    assert segments[0].session_boundary is True


async def test_bounded_retry_eventually_fails_cleanly():
    def always_fail():
        raise ConnectionError("simulated connect failure")

    transcriber = GeminiTranscriber(api_key="unused", session_rotation_seconds=999)
    transcriber._connect = always_fail

    with pytest.raises(RuntimeError):
        async for _ in transcriber.transcribe(make_audio_source(n_chunks=5, delay=0)):
            pass


class _FakeHealthyTranscriber:
    """A minimal TranscriptionProvider stub, local to this file to keep it
    self-contained (mirrors the shape used across the test suite)."""

    def __init__(self, segments):
        self._segments = segments

    async def transcribe(self, audio_chunks):
        for segment in self._segments:
            yield segment


async def test_a_failing_stage_does_not_affect_another_stage(tmp_path):
    """GeminiTranscriber exhausting its reconnect budget must only fail its
    own StagePipeline — a sibling stage (using a different transcriber)
    must run to completion unaffected."""
    from app.providers import TranscriptSegment

    def always_fail():
        raise ConnectionError("simulated connect failure")

    broken_transcriber = GeminiTranscriber(api_key="unused", session_rotation_seconds=999)
    broken_transcriber._connect = always_fail

    healthy_transcriber = _FakeHealthyTranscriber(
        [TranscriptSegment(text="Hola.", is_final=True, language="es")]
    )

    store = JsonlEventStore(base_dir=tmp_path)
    broadcaster = Broadcaster()

    main_config = StageConfig(
        id="main", name="Main", language="en", targets=[],
        source=FileSourceConfig(type="file", path="unused.wav"),
    )
    devroom_config = StageConfig(
        id="devroom", name="Dev Room", language="es", targets=[],
        source=FileSourceConfig(type="file", path="unused.wav"),
    )

    main_pipeline = StagePipeline(
        main_config, broken_transcriber, store, broadcaster,
        audio_source=_FakeAudioSourceWrapper(),
    )
    devroom_pipeline = StagePipeline(devroom_config, healthy_transcriber, store, broadcaster)

    await asyncio.gather(main_pipeline.run(), devroom_pipeline.run())

    assert main_pipeline.status == "error"
    assert devroom_pipeline.status == "stopped"

    import json
    devroom_lines = (tmp_path / "devroom.jsonl").read_text().splitlines()
    assert [json.loads(line)["text"] for line in devroom_lines] == ["Hola."]
