from typing import Protocol, TypeVar

from pydantic import TypeAdapter

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
        _ = prompt
        _ = adapter
        if not self.base_url or not self.api_key:
            raise RuntimeError("openai_compatible_provider_not_configured")
        raise NotImplementedError("openai_compatible_provider_not_implemented")
