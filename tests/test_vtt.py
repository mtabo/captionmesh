import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.vtt import _format_timestamp, build_vtt, write_timestamped_vtt_file, write_vtt_file


def final(seg_id, text, audio_elapsed_ms, stage_id="main"):
    return {
        "type": "caption.final",
        "stage_id": stage_id,
        "seg_id": seg_id,
        "text": text,
        "language": "en",
        "timing": {"audio_elapsed_ms": audio_elapsed_ms, "asr_latency_ms": 10.0},
    }


def final_without_timing(seg_id, text, stage_id="main"):
    return {
        "type": "caption.final",
        "stage_id": stage_id,
        "seg_id": seg_id,
        "text": text,
        "language": "en",
        "timing": None,
    }


def interim(seg_id, text, audio_elapsed_ms, stage_id="main"):
    return {
        "type": "caption.interim",
        "stage_id": stage_id,
        "seg_id": seg_id,
        "text": text,
        "language": "en",
        "timing": {"audio_elapsed_ms": audio_elapsed_ms, "asr_latency_ms": 10.0},
    }


def translation(seg_id, text, stage_id="main", language="es"):
    return {
        "type": "caption.translation",
        "stage_id": stage_id,
        "seg_id": seg_id,
        "text": text,
        "language": language,
        "source_language": "en",
        "timing": None,  # StagePipeline never sets this — see app/vtt.py
    }


def test_output_starts_with_a_valid_webvtt_header():
    vtt = build_vtt([final("main-000001", "Hello.", 1000.0)])
    assert vtt.startswith("WEBVTT\n\n")


def test_empty_stage_produces_a_valid_empty_vtt():
    vtt = build_vtt([])
    assert vtt == "WEBVTT\n"
    assert vtt.startswith("WEBVTT")


def test_finalized_captions_appear_as_cues():
    vtt = build_vtt([final("main-000001", "Hello world.", 2000.0)])
    assert "Hello world." in vtt
    assert "1" in vtt.splitlines()  # cue index
    assert "00:00:00.000 --> 00:00:02.000" in vtt


def test_interim_captions_never_appear_even_if_present_in_the_input():
    events = [
        interim("main-000001", "Hel", 500.0),
        interim("main-000001", "Hello", 800.0),
        final("main-000001", "Hello.", 1000.0),
    ]
    vtt = build_vtt(events)
    assert "Hel\n" not in vtt
    assert "Hello\n" not in vtt
    assert "Hello.\n" in vtt


def test_correct_ordering_of_multiple_cues():
    events = [
        final("main-000001", "First.", 1000.0),
        final("main-000002", "Second.", 3000.0),
        final("main-000003", "Third.", 4500.0),
    ]
    vtt = build_vtt(events)

    first_pos = vtt.index("First.")
    second_pos = vtt.index("Second.")
    third_pos = vtt.index("Third.")
    assert first_pos < second_pos < third_pos

    assert "00:00:00.000 --> 00:00:01.000" in vtt  # cue 1: 0 -> 1000ms
    assert "00:00:01.000 --> 00:00:03.000" in vtt  # cue 2: previous end -> 3000ms
    assert "00:00:03.000 --> 00:00:04.500" in vtt  # cue 3: previous end -> 4500ms


def test_translation_text_is_preferred_when_one_exists_for_the_seg_id():
    events = [
        final("main-000001", "Hello.", 1000.0),
        translation("main-000001", "Hola."),
    ]
    vtt = build_vtt(events)

    assert "Hola." in vtt
    assert "Hello.\n" not in vtt  # original text was replaced, not duplicated
    # The cue's timestamp still comes from the original final, not the
    # translation (which never carries timing) — verify it is still present.
    assert "00:00:00.000 --> 00:00:01.000" in vtt


def test_original_text_is_used_when_no_translation_exists_for_the_seg_id():
    events = [
        final("main-000001", "Hello.", 1000.0),
        final("main-000002", "World.", 2000.0),
        translation("main-000001", "Hola."),  # only seg 1 has a translation
    ]
    vtt = build_vtt(events)

    assert "Hola." in vtt
    assert "World." in vtt  # seg 2 falls back to its own original text


def test_a_final_without_timing_is_skipped_not_fabricated():
    events = [
        final_without_timing("main-000001", "No timing here."),
        final("main-000002", "Has timing.", 1500.0),
    ]
    vtt = build_vtt(events)

    assert "No timing here." not in vtt
    assert "Has timing." in vtt
    # Only one real cue was produced.
    assert vtt.count(" --> ") == 1


def test_events_from_another_stage_are_not_mixed_in():
    events = [
        final("main-000001", "Main stage text.", 1000.0, stage_id="main"),
        final("devroom-000001", "Devroom text.", 1000.0, stage_id="devroom"),
    ]
    # build_vtt itself does not filter by stage_id — that isolation comes
    # from JsonlEventStore.read_events only ever reading one stage's file
    # (see test_store.py). This test documents that build_vtt is a pure
    # function operating on whatever event list it's given.
    vtt = build_vtt(events)
    assert "Main stage text." in vtt
    assert "Devroom text." in vtt


def test_format_timestamp_produces_hh_mm_ss_mmm():
    assert _format_timestamp(0) == "00:00:00.000"
    assert _format_timestamp(1500) == "00:00:01.500"
    assert _format_timestamp(61_000) == "00:01:01.000"
    assert _format_timestamp(3_661_250) == "01:01:01.250"


def test_write_vtt_file_creates_the_directory_and_the_file(tmp_path):
    base_dir = tmp_path / "vtt"
    vtt = build_vtt([final("main-000001", "Hello.", 1000.0)])

    path = write_vtt_file("main", vtt, base_dir=base_dir)

    assert path == base_dir / "main.vtt"
    assert path.read_text(encoding="utf-8") == vtt


def test_write_vtt_file_is_a_valid_snapshot_even_when_empty(tmp_path):
    base_dir = tmp_path / "vtt"
    vtt = build_vtt([])  # no captions yet — not an error

    path = write_vtt_file("devroom", vtt, base_dir=base_dir)

    assert path.read_text(encoding="utf-8") == "WEBVTT\n"


def test_write_vtt_file_overwrites_the_previous_snapshot(tmp_path):
    base_dir = tmp_path / "vtt"
    write_vtt_file("main", build_vtt([final("main-000001", "First.", 1000.0)]), base_dir=base_dir)

    updated = build_vtt([
        final("main-000001", "First.", 1000.0),
        final("main-000002", "Second.", 2000.0),
    ])
    path = write_vtt_file("main", updated, base_dir=base_dir)

    assert "Second." in path.read_text(encoding="utf-8")


def test_write_vtt_file_keeps_different_stages_in_separate_files(tmp_path):
    base_dir = tmp_path / "vtt"
    main_path = write_vtt_file("main", build_vtt([final("main-000001", "Main text.", 1000.0)]), base_dir=base_dir)
    devroom_path = write_vtt_file(
        "devroom", build_vtt([final("devroom-000001", "Devroom text.", 1000.0)]), base_dir=base_dir
    )

    assert main_path != devroom_path
    assert "Main text." in main_path.read_text(encoding="utf-8")
    assert "Devroom text." in devroom_path.read_text(encoding="utf-8")
    assert "Devroom text." not in main_path.read_text(encoding="utf-8")


def test_write_timestamped_vtt_file_uses_stage_id_and_formatted_timestamp(tmp_path):
    base_dir = tmp_path / "vtt"
    vtt = build_vtt([final("main-000001", "Hello.", 1000.0)])

    path = write_timestamped_vtt_file("main", vtt, when=datetime(2026, 9, 25, 18, 34, 21), base_dir=base_dir)

    assert path == base_dir / "main_20260925-183421.vtt"
    assert path.read_text(encoding="utf-8") == vtt


def test_write_timestamped_vtt_file_name_has_no_colon_or_other_unsafe_characters(tmp_path):
    base_dir = tmp_path / "vtt"

    path = write_timestamped_vtt_file(
        "devroom", build_vtt([]), when=datetime(2026, 9, 25, 18, 42, 30), base_dir=base_dir
    )

    assert ":" not in path.name
    assert path.name == "devroom_20260925-184230.vtt"


def test_write_timestamped_vtt_file_defaults_to_the_real_clock_when_omitted(tmp_path):
    base_dir = tmp_path / "vtt"
    before = datetime.now()

    path = write_timestamped_vtt_file("main", build_vtt([]), base_dir=base_dir)

    after = datetime.now()
    timestamp_str = path.stem.split("_", 1)[1]
    parsed = datetime.strptime(timestamp_str, "%Y%m%d-%H%M%S")
    # Compare at second resolution — the format itself has no finer precision.
    assert before.replace(microsecond=0) <= parsed <= after.replace(microsecond=0)


def test_write_timestamped_vtt_file_two_distinct_timestamps_produce_two_files(tmp_path):
    base_dir = tmp_path / "vtt"

    first = write_timestamped_vtt_file(
        "main", build_vtt([final("main-000001", "First.", 1000.0)]),
        when=datetime(2026, 9, 25, 18, 0, 0), base_dir=base_dir,
    )
    second = write_timestamped_vtt_file(
        "main", build_vtt([final("main-000001", "Second.", 1000.0)]),
        when=datetime(2026, 9, 25, 18, 5, 0), base_dir=base_dir,
    )

    assert first != second
    assert first.exists() and second.exists()
    assert "First." in first.read_text(encoding="utf-8")
    assert "Second." in second.read_text(encoding="utf-8")


def test_write_timestamped_vtt_file_is_independent_from_the_plain_snapshot(tmp_path):
    """The on-demand endpoint's write_vtt_file (<stage_id>.vtt) and the
    per-session write_timestamped_vtt_file (<stage_id>_<timestamp>.vtt) must
    coexist without clobbering each other."""
    base_dir = tmp_path / "vtt"
    vtt = build_vtt([final("main-000001", "Hello.", 1000.0)])

    plain_path = write_vtt_file("main", vtt, base_dir=base_dir)
    timestamped_path = write_timestamped_vtt_file(
        "main", vtt, when=datetime(2026, 9, 25, 18, 0, 0), base_dir=base_dir
    )

    assert plain_path != timestamped_path
    assert plain_path.exists() and timestamped_path.exists()
