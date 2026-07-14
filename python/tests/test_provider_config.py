import json
from urllib.error import HTTPError

from pydantic import BaseModel, TypeAdapter
import pytest
from typer.testing import CliRunner

from sipz_agent.cli import app
from sipz_agent.core.config import load_config, resolve_model_config
from sipz_agent.core.models import AnthropicProvider, create_llm_provider, OpenAICompatibleProvider
from sipz_agent.core.orchestrator import run_study


def test_resolve_deepseek_provider_uses_deepseek_api_key(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)

    config = resolve_model_config(provider="deepseek", model=None)

    assert config.provider == "deepseek"
    assert config.model_name == "deepseek-chat"
    assert config.base_url == "https://api.deepseek.com"
    assert config.api_key == "test-key"


def test_load_config_reads_elsevier_api_key(monkeypatch) -> None:
    monkeypatch.setenv("ELSEVIER_API_KEY", "elsevier-key")

    config = load_config()

    assert config.elsevier_api_key == "elsevier-key"


def test_load_config_reads_unpaywall_email(monkeypatch) -> None:
    monkeypatch.setenv("UNPAYWALL_EMAIL", "research@example.com")

    assert load_config().unpaywall_email == "research@example.com"


def test_load_config_reads_openalex_api_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "openalex-key")

    assert load_config().openalex_api_key == "openalex-key"


def test_load_config_reads_ncbi_and_crossref_credentials(monkeypatch) -> None:
    monkeypatch.setenv("NCBI_API_KEY", "ncbi-key")
    monkeypatch.setenv("CROSSREF_MAILTO", "research@example.com")

    config = load_config()

    assert config.ncbi_api_key == "ncbi-key"
    assert config.crossref_mailto == "research@example.com"


def test_load_config_uses_ncbi_email_for_unpaywall_fallback(monkeypatch) -> None:
    # Keep dotenv from restoring the developer's local value during this test.
    monkeypatch.setenv("UNPAYWALL_EMAIL", "")
    monkeypatch.setenv("NCBI_EMAIL", "research@example.com")

    assert load_config().unpaywall_email == "research@example.com"


def test_create_deepseek_provider_returns_openai_compatible_provider(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    provider = create_llm_provider(resolve_model_config(provider="deepseek", model="deepseek-reasoner"))

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.base_url == "https://api.deepseek.com"
    assert provider.model == "deepseek-reasoner"


def test_resolve_openai_provider_uses_openai_api_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    config = resolve_model_config(provider="openai", model="gpt-5-mini")

    assert config.provider == "openai"
    assert config.model_name == "gpt-5-mini"
    assert config.base_url == "https://api.openai.com/v1"
    assert config.api_key == "openai-key"


def test_resolve_anthropic_provider_uses_anthropic_api_key(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

    config = resolve_model_config(provider="anthropic", model="claude-haiku-4-5")
    provider = create_llm_provider(config)

    assert config.provider == "anthropic"
    assert config.model_name == "claude-haiku-4-5"
    assert config.base_url == "https://api.anthropic.com"
    assert isinstance(provider, AnthropicProvider)


def test_direct_providers_require_their_api_keys(monkeypatch) -> None:
    monkeypatch.setattr("sipz_agent.core.config.load_dotenv", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="openai_api_key_missing"):
        resolve_model_config(provider="openai", model="gpt-5-mini")
    with pytest.raises(RuntimeError, match="anthropic_api_key_missing"):
        resolve_model_config(provider="anthropic", model="claude-haiku-4-5")


def test_study_packet_records_requested_provider_and_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    result = run_study(
        nutrient_name="fluoride",
        depth="standard",
        demo=True,
        out_dir=tmp_path,
        provider="deepseek",
        model="deepseek-chat",
    )

    packet = json.loads((result.run_dir / "packet.json").read_text(encoding="utf-8"))
    assert packet["model"]["provider"] == "deepseek"
    assert packet["model"]["model_name"] == "deepseek-chat"


def test_cli_accepts_provider_and_model_options(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "study",
            "fluoride",
            "--demo",
            "--provider",
            "deepseek",
            "--model",
            "deepseek-chat",
            "--out",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    run_dirs = list(tmp_path.iterdir())
    assert len(run_dirs) == 1


def test_deepseek_provider_requires_api_key(monkeypatch) -> None:
    monkeypatch.setattr("sipz_agent.core.config.load_dotenv", lambda: None)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="deepseek_api_key_missing"):
        resolve_model_config(provider="deepseek", model=None)


class ProviderSmokeResponse(BaseModel):
    effect: str


def test_anthropic_provider_posts_messages_and_validates_json(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps(
                {"content": [{"type": "text", "text": '```json\n{"effect":"supports bone health"}\n```'}]}
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = AnthropicProvider(
        base_url="https://api.anthropic.com",
        api_key="test-key",
        model="claude-haiku-4-5",
    )

    result = provider.complete_json("Return JSON.", TypeAdapter(ProviderSmokeResponse))

    assert result.effect == "supports bone health"
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["X-api-key"] == "test-key"
    assert captured["headers"]["Anthropic-version"] == "2023-06-01"
    assert captured["body"]["model"] == "claude-haiku-4-5"
    assert captured["body"]["temperature"] == 0


def test_openai_compatible_provider_posts_chat_completion_and_validates_json(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": '{"effect":"supports bone health"}'}}]}
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        base_url="https://api.deepseek.com",
        api_key="test-key",
        model="deepseek-chat",
    )

    result = provider.complete_json(
        "Return JSON for one magnesium health effect.",
        TypeAdapter(ProviderSmokeResponse),
    )

    assert result.effect == "supports bone health"
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["body"]["model"] == "deepseek-chat"
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["body"]["max_tokens"] == 8192
    assert captured["body"]["temperature"] == 0


def test_direct_openai_gpt5_uses_completion_tokens_and_minimal_reasoning(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": '{"effect":"supports bone health"}'}}]}
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-5-mini",
    )

    provider.complete_json("Return JSON.", TypeAdapter(ProviderSmokeResponse))

    assert captured["body"]["max_completion_tokens"] == 8192
    assert captured["body"]["reasoning_effort"] == "minimal"
    assert "max_tokens" not in captured["body"]
    assert "temperature" not in captured["body"]


def test_openai_compatible_provider_retries_truncated_json(monkeypatch) -> None:
    responses = [
        {"choices": [{"message": {"content": '{"effect":"supports'}}]},
        {"choices": [{"message": {"content": '{"effect":"supports bone health"}'}}]},
    ]
    request_bodies = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    def fake_urlopen(request, timeout):
        request_bodies.append(json.loads(request.data.decode("utf-8")))
        return FakeResponse(responses.pop(0))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        base_url="https://openrouter.ai/api/v1",
        api_key="test-key",
        model="deepseek/deepseek-v4-flash",
    )

    result = provider.complete_json(
        "Return JSON for one magnesium health effect.",
        TypeAdapter(ProviderSmokeResponse),
    )

    assert result.effect == "supports bone health"
    assert len(request_bodies) == 2
    assert "previous response was invalid or truncated" in request_bodies[1]["messages"][1]["content"]


def test_openai_compatible_provider_maps_payment_required(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        raise HTTPError(
            url=request.full_url,
            code=402,
            msg="Payment Required",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        base_url="https://api.deepseek.com",
        api_key="test-key",
        model="deepseek-chat",
    )

    with pytest.raises(RuntimeError, match="llm_provider_payment_required"):
        provider.complete_json(
            "Return JSON for one magnesium health effect.",
            TypeAdapter(ProviderSmokeResponse),
        )


def test_openai_compatible_provider_retries_rate_limit(monkeypatch) -> None:
    calls = 0

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": '{"effect":"supports health"}'}}]}
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HTTPError(request.full_url, 429, "Rate Limited", {"Retry-After": "0"}, None)
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("sipz_agent.core.models.time.sleep", lambda _: None)
    provider = OpenAICompatibleProvider(
        base_url="https://api.deepseek.com",
        api_key="test-key",
        model="deepseek-chat",
    )

    result = provider.complete_json("Return JSON.", TypeAdapter(ProviderSmokeResponse))

    assert result.effect == "supports health"
    assert calls == 2
