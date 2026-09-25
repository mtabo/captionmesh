import logging

from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-3.5-flash-lite"

SYSTEM_INSTRUCTION = (
    "You split a finalized live-caption transcript into short, natural "
    "subtitle-sized units (roughly one sentence or clause each), the way "
    "real subtitles are broken up. Rules: "
    "1) Preserve the original wording exactly - never paraphrase, "
    "translate, summarize, correct, or add/remove any words. "
    "2) Never split inside an abbreviation, a decimal number, or a name. "
    "3) If the input is already a single short sentence, return it "
    "unchanged as the only element. "
    "4) Output ONLY the ordered list of segments - no explanation, no "
    "numbering, no extra commentary."
)

logger = logging.getLogger(__name__)


class GeminiSegmenter:
    """SegmentationProvider backed by Gemini text generation.

    Splits one finalized ASR transcript into sentence/phrase-sized units
    for subtitle-style display. Stateless per call (like GeminiTranslator),
    so a single instance is safely shared across stages.

    Uses structured JSON output (`response_schema=list[str]`) rather than
    a free-text response, so there is no explanatory prose to strip out —
    the model can only return a plain list of strings. `segment()` raises
    on any invalid/implausible output; the caller (StagePipeline) decides
    the fallback policy (falling back to the original text as a single
    segment), not this class.
    """

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def segment(self, text: str) -> list[str]:
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=list[str],
        )
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=text,
            config=config,
        )
        return self._validate(response.parsed, text)

    @staticmethod
    def _validate(parsed, original_text: str) -> list[str]:
        """Raises ValueError on anything that isn't a plausible plain
        segmentation of `original_text` — never returns a fabricated or
        contaminated result for the caller to accidentally trust."""
        if not isinstance(parsed, list) or not parsed:
            raise ValueError(f"segmentation returned no valid list: {parsed!r}")

        segments = [s.strip() for s in parsed if isinstance(s, str) and s.strip()]
        if not segments:
            raise ValueError("segmentation returned only empty/invalid entries")

        # Sanity check against the model paraphrasing/summarizing instead of
        # just splitting: the total text length should stay in the same
        # ballpark as the original. Not a strict guarantee, but catches
        # gross deviations cheaply without re-parsing/diffing content.
        original_len = len(original_text)
        joined_len = sum(len(s) for s in segments)
        if original_len > 0 and not (0.5 * original_len <= joined_len <= 2.0 * original_len):
            raise ValueError(
                f"segmentation output length ({joined_len}) too different from "
                f"the original ({original_len}) - likely paraphrased, not segmented"
            )

        return segments
