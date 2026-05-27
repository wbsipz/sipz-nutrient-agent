import json

from pydantic import BaseModel, TypeAdapter
import pytest
from typer.testing import CliRunner

from sipz_agent.cli import app
from sipz_agent.core.config import resolve_model_config
from sipz_agent.core.models import create_llm_provider, OpenAICompatibleProvider
from sipz_agent.core.orchestrator import run_study


def test_resolve_deepseek_provider_uses_deepseek_api_key(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)

    config = resolve_model_config(provider="deepseek", model=None)

    assert config.provider == "deepseek"
    assert config.model_name == "deepseek-chat"
    assert config.base_url == "https://api.deepseek.com"
    assert config.api_key == "test-key"


def test_create_deepseek_provider_returns_openai_compatible_provider(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    provider = create_llm_provider(resolve_model_config(provider="deepseek", model="deepseek-reasoner"))

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.base_url == "https://api.deepseek.com"
    assert provider.model == "deepseek-reasoner"


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
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="deepseek_api_key_missing"):
        resolve_model_config(provider="deepseek", model=None)


class ProviderSmokeResponse(BaseModel):
    effect: str


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
