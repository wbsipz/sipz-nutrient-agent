import json
import random
import time
import re
from typing import Protocol, TypeVar
from urllib import error, request

from pydantic import TypeAdapter, ValidationError

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

        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        validation_error: ValidationError | None = None
        response_json_error: json.JSONDecodeError | None = None

        for attempt in range(2):
            retry_instruction = ""
            if attempt:
                retry_instruction = (
                    "\nYour previous response was invalid or truncated. Regenerate the complete JSON "
                    "from the beginning. Be concise and close every string, array, and object."
                )
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Return only complete, valid JSON matching the requested schema. "
                            "Keep explanatory strings concise."
                        ),
                    },
                    {"role": "user", "content": prompt + retry_instruction},
                ],
                "response_format": {"type": "json_object"},
            }
            direct_openai = self.base_url.rstrip("/") == "https://api.openai.com/v1"
            if direct_openai and self.model.startswith(("gpt-5", "o1", "o3", "o4")):
                payload["max_completion_tokens"] = 8192
                payload["reasoning_effort"] = "minimal"
            else:
                payload["max_tokens"] = 8192
                payload["temperature"] = 0
            req = request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            raw_response: str | None = None
            for transport_attempt in range(4):
                try:
                    with request.urlopen(req, timeout=120) as response:
                        raw_response = response.read().decode("utf-8")
                    break
                except error.HTTPError as exc:
                    if exc.code == 402:
                        raise RuntimeError("llm_provider_payment_required") from exc
                    if exc.code not in {408, 429, 502, 503, 504} or transport_attempt == 3:
                        raise
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        delay = float(retry_after) if retry_after else 0.0
                    except ValueError:
                        delay = 0.0
                    if delay <= 0:
                        delay = (1.0, 3.0, 8.0)[transport_attempt] + random.uniform(0, 1.5)
                    time.sleep(min(delay, 30.0))
                except (error.URLError, TimeoutError):
                    if transport_attempt == 3:
                        raise
                    delay = (1.0, 3.0, 8.0)[transport_attempt] + random.uniform(0, 1.5)
                    time.sleep(delay)
            if raw_response is None:
                raise RuntimeError("llm_provider_no_response")
            try:
                response_payload = json.loads(raw_response)
            except json.JSONDecodeError as exc:
                response_json_error = exc
                continue

            content = response_payload["choices"][0]["message"]["content"]
            try:
                return adapter.validate_json(content)
            except ValidationError as exc:
                validation_error = exc

        if validation_error is None:
            if response_json_error is not None:
                raise RuntimeError("llm_provider_invalid_response_json") from response_json_error
            raise RuntimeError("llm_provider_invalid_json")
        raise validation_error


class AnthropicProvider:
    def __init__(self, base_url: str | None, api_key: str | None, model: str | None) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def complete_json(self, prompt: str, adapter: TypeAdapter[T]) -> T:
        if not self.base_url or not self.api_key or not self.model:
            raise RuntimeError("anthropic_provider_not_configured")

        endpoint = f"{self.base_url.rstrip('/')}/v1/messages"
        validation_error: ValidationError | None = None
        response_json_error: json.JSONDecodeError | None = None
        for attempt in range(2):
            retry_instruction = ""
            if attempt:
                retry_instruction = (
                    "\nYour previous response was invalid or truncated. Regenerate the complete JSON "
                    "from the beginning. Be concise and close every string, array, and object."
                )
            payload = {
                "model": self.model,
                "system": (
                    "Return only complete, valid JSON matching the requested schema. "
                    "Do not use Markdown fences. Keep explanatory strings concise."
                ),
                "messages": [{"role": "user", "content": prompt + retry_instruction}],
                "max_tokens": 8192,
                "temperature": 0,
            }
            req = request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "X-Api-Key": self.api_key,
                    "Anthropic-Version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            raw_response: str | None = None
            for transport_attempt in range(4):
                try:
                    with request.urlopen(req, timeout=120) as response:
                        raw_response = response.read().decode("utf-8")
                    break
                except error.HTTPError as exc:
                    if exc.code == 402:
                        raise RuntimeError("llm_provider_payment_required") from exc
                    if exc.code not in {408, 429, 502, 503, 504} or transport_attempt == 3:
                        raise
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    try:
                        delay = float(retry_after) if retry_after else 0.0
                    except ValueError:
                        delay = 0.0
                    if delay <= 0:
                        delay = (1.0, 3.0, 8.0)[transport_attempt] + random.uniform(0, 1.5)
                    time.sleep(min(delay, 30.0))
                except (error.URLError, TimeoutError):
                    if transport_attempt == 3:
                        raise
                    delay = (1.0, 3.0, 8.0)[transport_attempt] + random.uniform(0, 1.5)
                    time.sleep(delay)
            if raw_response is None:
                raise RuntimeError("llm_provider_no_response")
            try:
                response_payload = json.loads(raw_response)
            except json.JSONDecodeError as exc:
                response_json_error = exc
                continue
            content = "".join(
                block.get("text", "")
                for block in response_payload.get("content", [])
                if block.get("type") == "text"
            ).strip()
            fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.DOTALL)
            if fenced:
                content = fenced.group(1)
            try:
                return adapter.validate_json(content)
            except ValidationError as exc:
                validation_error = exc

        if validation_error is None:
            if response_json_error is not None:
                raise RuntimeError("llm_provider_invalid_response_json") from response_json_error
            raise RuntimeError("llm_provider_invalid_json")
        raise validation_error


def create_llm_provider(config: ModelConfig) -> LlmProvider:
    if config.provider == "heuristic":
        return HeuristicProvider()
    if config.provider == "anthropic":
        return AnthropicProvider(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model_name,
        )
    return OpenAICompatibleProvider(
        base_url=config.base_url,
        api_key=config.api_key,
        model=config.model_name,
    )
