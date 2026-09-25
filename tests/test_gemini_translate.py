import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.providers import gemini_translate as gemini_translate_module
from app.providers.gemini_translate import GeminiTranslator


class FakeResponse:
    def __init__(self, text):
        self.text = text


def make_fake_client(captured, response_text="translated"):
    """Patches genai.Client so a real (unmonkeypatched) translate() call can
    be inspected: records the GenerateContentConfig built and returns a
    canned response, with no real API call."""

    class FakeModels:
        async def generate_content(self, model, contents, config):
            captured["model"] = model
            captured["contents"] = contents
            captured["config"] = config
            return FakeResponse(response_text)

    class FakeAio:
        def __init__(self):
            self.models = FakeModels()

    class FakeClient:
        def __init__(self, api_key):
            self.aio = FakeAio()

    return FakeClient


async def test_glossary_terms_are_appended_to_the_system_instruction(monkeypatch):
    captured = {}
    monkeypatch.setattr(gemini_translate_module.genai, "Client", make_fake_client(captured))
    translator = GeminiTranslator(api_key="unused", glossary=["Firebase", "Kubernetes"])

    await translator.translate("We used Firebase.", "en", "es")

    instruction = captured["config"].system_instruction
    assert "Firebase, Kubernetes" in instruction
    assert "never translated or transliterated" in instruction


async def test_no_glossary_instruction_when_glossary_is_empty_preserves_existing_behavior(monkeypatch):
    captured = {}
    monkeypatch.setattr(gemini_translate_module.genai, "Client", make_fake_client(captured))
    translator = GeminiTranslator(api_key="unused")  # glossary omitted — existing call signature

    await translator.translate("Hello.", "en", "es")

    instruction = captured["config"].system_instruction
    assert "never translated" not in instruction
    assert instruction == gemini_translate_module.SYSTEM_INSTRUCTION.format(source="en", target="es")


async def test_translate_still_returns_stripped_response_text(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        gemini_translate_module.genai, "Client", make_fake_client(captured, response_text="  Hola.  ")
    )
    translator = GeminiTranslator(api_key="unused", glossary=["Firebase"])

    result = await translator.translate("Hi.", "en", "es")

    assert result == "Hola."
    assert captured["contents"] == "Hi."
