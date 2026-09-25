import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.providers.gemini_segment import GeminiSegmenter


def test_valid_list_of_strings_is_accepted_and_stripped():
    result = GeminiSegmenter._validate(["Hello.", " World. "], "Hello. World.")
    assert result == ["Hello.", "World."]


def test_single_unchanged_segment_is_accepted():
    """The common "no split needed" case: the model returns the original
    text as the only element."""
    result = GeminiSegmenter._validate(["Hi."], "Hi.")
    assert result == ["Hi."]


def test_non_list_output_is_rejected():
    with pytest.raises(ValueError):
        GeminiSegmenter._validate("Hello. World.", "Hello. World.")


def test_empty_list_is_rejected():
    with pytest.raises(ValueError):
        GeminiSegmenter._validate([], "Hello.")


def test_list_of_only_empty_or_whitespace_strings_is_rejected():
    with pytest.raises(ValueError):
        GeminiSegmenter._validate(["", "   "], "Hello.")


def test_non_string_entries_are_dropped_not_fatal_on_their_own():
    result = GeminiSegmenter._validate(["Hello.", 42, None, "World."], "Hello. World.")
    assert result == ["Hello.", "World."]


def test_output_much_shorter_than_original_is_rejected_as_likely_summarized():
    original = "This is a fairly long original sentence that should not be summarized away."
    with pytest.raises(ValueError):
        GeminiSegmenter._validate(["Short."], original)


def test_output_much_longer_than_original_is_rejected_as_likely_hallucinated():
    original = "Short."
    with pytest.raises(ValueError):
        GeminiSegmenter._validate(
            ["Short, but here is a very long fabricated explanation that was never in the original text at all."],
            original,
        )


def test_empty_original_text_skips_the_length_ratio_check():
    # Degenerate input — just must not crash on a divide-by-zero-shaped check.
    result = GeminiSegmenter._validate(["Something."], "")
    assert result == ["Something."]
