import json
from typing import Protocol, TypeVar
from urllib import request

from pydantic import TypeAdapter

from sipz_agent.core.config import ModelConfig

T = TypeVar("T")


class LlmProvider(Protocol):
    def complete_json(self, prompt: str, adapter: TypeAdapter[T]) -> T:
        """Return validated JSON for a prompt."""

class HeuristicProvider:
    def complete_json(self, prompt: str, adapter: TypeAdapter[T]) -> T:
        _ = prompt
        _ = adapter
        raise NotImplementedError("heuristic_provider_does_not_generate_json")

class OpenAICompatibleProvider:
    def __init__(self, base_url: str | None, api_key: str | None, model: str | None) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def complete_json(self, prompt: str, adapter: TypeAdapter[T]) -> T:
        if not self.base_url or not self.api_key or not self.model:
            raise RuntimeError("openai_compatible_provider_not_configured")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only valid JSON matching the requested schema.",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=120) as response:
            response_payload = json.loads(response.read().decode("utf-8"))

        content = response_payload["choices"][0]["message"]["content"]
        return adapter.validate_json(content)


def create_llm_provider(config: ModelConfig) -> LlmProvider:
    if config.provider == "heuristic":
        return HeuristicProvider()
    return OpenAICompatibleProvider(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model_name,
    )
