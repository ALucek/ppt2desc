from pathlib import Path
from types import SimpleNamespace

import pytest

from llm.anthropic import AnthropicClient
from llm.aws import AWSClient
from llm.azure import AzureClient
from llm.google_unified import GoogleUnifiedClient
from llm.openai import OpenAIClient
from schemas.deck import DeckData, SlideData


def test_openai_client_generate_returns_text(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    image_path = tmp_path / "slide.png"
    image_path.write_bytes(b"fake-image")

    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="result"))])

    class FakeCompletions:
        def __init__(self, resp):
            self._resp = resp

        def create(self, **kwargs):
            return self._resp

    class FakeOpenAI:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(completions=FakeCompletions(response))

    monkeypatch.setattr("llm.openai.OpenAI", FakeOpenAI)

    client = OpenAIClient(model="gpt-test")
    text = client.generate("prompt", image_path)

    assert text == "result"


def test_openai_client_generate_handles_multimodal_list(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    image_path = tmp_path / "slide.png"
    image_path.write_bytes(b"fake-image")

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[SimpleNamespace(text="list-text")]
                )
            )
        ]
    )

    class FakeCompletions:
        def __init__(self, resp):
            self._resp = resp

        def create(self, **kwargs):
            return self._resp

    class FakeOpenAI:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(completions=FakeCompletions(response))

    monkeypatch.setattr("llm.openai.OpenAI", FakeOpenAI)

    client = OpenAIClient(model="gpt-test")
    text = client.generate("prompt", image_path)

    assert text == "list-text"


def test_openai_client_generate_missing_image_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setattr("llm.openai.OpenAI", lambda api_key: None)

    client = OpenAIClient(model="gpt-test")

    with pytest.raises(FileNotFoundError):
        client.generate("prompt", Path("missing.png"))


def test_anthropic_client_generate(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")
    image_path = tmp_path / "slide.png"
    image_path.write_bytes(b"fake")

    class FakeMessages:
        def create(self, **kwargs):
            return SimpleNamespace(content=[SimpleNamespace(text="anthropic")])

    class FakeAnthropic:
        def __init__(self, api_key):
            self.messages = FakeMessages()

    monkeypatch.setattr("llm.anthropic.anthropic.Anthropic", FakeAnthropic)

    client = AnthropicClient(model="claude")
    text = client.generate("prompt", image_path)

    assert text == "anthropic"


def test_azure_client_requires_endpoint(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")

    with pytest.raises(ValueError):
        AzureClient(deployment="dep", api_version="2023-12-01-preview")


def test_azure_client_generate_extracts_text(tmp_path, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example")
    image_path = tmp_path / "slide.png"
    image_path.write_bytes(b"fake")

    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=[{"text": "azure-text"}]
                )
            )
        ]
    )

    class FakeCompletions:
        def __init__(self, resp):
            self._resp = resp

        def create(self, **kwargs):
            return self._resp

    class FakeAzure:
        def __init__(self, api_key, api_version, base_url):
            self.chat = SimpleNamespace(completions=FakeCompletions(response))

    monkeypatch.setattr("llm.azure.AzureOpenAI", FakeAzure)

    client = AzureClient(deployment="dep", api_version="2023-12-01-preview")
    text = client.generate("prompt", image_path)

    assert text == "azure-text"


def test_aws_client_generate(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    image_path = tmp_path / "slide.png"
    image_path.write_bytes(b"fake")

    class FakeBedrock:
        def __init__(self, *args, **kwargs):
            pass

        def converse(self, modelId, messages):
            return {"output": {"message": {"content": [{"text": "bedrock"}]}}}

    monkeypatch.setattr("llm.aws.boto3.client", lambda *args, **kwargs: FakeBedrock())

    client = AWSClient(model="model-id")
    text = client.generate("prompt", image_path)

    assert text == "bedrock"


def test_aws_client_missing_region(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)

    with pytest.raises(ValueError):
        AWSClient(access_key_id="id", secret_access_key="secret", model="m")


def test_google_unified_client_gemini_generate(tmp_path, monkeypatch):
    from PIL import Image

    monkeypatch.setenv("GEMINI_API_KEY", "key")
    image_path = tmp_path / "slide.png"
    Image.new("RGB", (10, 10), (255, 255, 255)).save(image_path)

    class FakeModels:
        def generate_content(self, model, contents):
            assert contents[0] == "prompt"
            return SimpleNamespace(text="gemini")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.models = FakeModels()

    monkeypatch.setattr("llm.google_unified.genai.Client", FakeClient)

    client = GoogleUnifiedClient(model="gemini-model")
    text = client.generate("prompt", image_path)

    assert text == "gemini"


def test_google_unified_vertex_requires_credentials(tmp_path, monkeypatch):
    credentials = tmp_path / "creds.json"
    credentials.write_text("{}", encoding="utf-8")

    class FakeCredentials:
        def with_scopes(self, scopes):
            return self

    def fake_from_file(path):
        assert Path(path) == credentials
        return FakeCredentials()

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = SimpleNamespace(generate_content=lambda model, contents: SimpleNamespace(text="vertex"))

    monkeypatch.setattr("llm.google_unified.service_account.Credentials.from_service_account_file", fake_from_file)
    monkeypatch.setattr("llm.google_unified.genai.Client", FakeClient)

    client = GoogleUnifiedClient(
        credentials_path=str(credentials),
        project_id="pid",
        region="us-central1",
        model="vertex-model",
        use_vertex=True,
    )

    # Ensure generate works
    from PIL import Image

    image_path = tmp_path / "slide.png"
    Image.new("RGB", (10, 10), (255, 255, 255)).save(image_path)
    text = client.generate("prompt", image_path)
    assert text == "vertex"


def test_google_unified_vertex_missing_credentials_file(monkeypatch):
    with pytest.raises(FileNotFoundError):
        GoogleUnifiedClient(
            credentials_path="missing.json",
            project_id="pid",
            region="us",
            model="vertex",
            use_vertex=True,
        )


def test_deck_data_round_trip():
    deck = DeckData(
        deck="deck.pptx",
        model="model",
        slides=[SlideData(number=1, content="first"), SlideData(number=2, content="second")],
    )

    serialized = deck.model_dump_json()
    loaded = DeckData.model_validate_json(serialized)

    assert loaded == deck

