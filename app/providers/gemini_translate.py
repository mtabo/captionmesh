from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-3.5-flash"

SYSTEM_INSTRUCTION = (
    "You are a live captioning translator for a technical conference. "
    "Translate the caption text from {source} to {target}. "
    "Output ONLY the translated text: no quotes, no labels, no explanation. "
    "Preserve meaning and tone; keep it concise and suitable for a live subtitle."
)


class GeminiTranslator:
    """TranslationProvider backed by Gemini text generation.

    Stateless per call (unlike GeminiTranscriber, which owns a live
    streaming session), so a single instance is safely shared across stages.
    """

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def translate(self, text: str, source_language: str, target_language: str) -> str:
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION.format(source=source_language, target=target_language),
            temperature=0.0,
        )
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=text,
            config=config,
        )
        return (response.text or "").strip()
