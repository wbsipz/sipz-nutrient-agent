from dataclasses import dataclass
import os
from typing import Literal

from dotenv import load_dotenv

ModelProviderName = Literal["heuristic", "deepseek", "openai-compatible"]


@dataclass(frozen=True)
class AppConfig:
    model_provider: str = "heuristic"
    model_name: str = "heuristic-demo"
    openai_compatible_base_url: str | None = None
    openai_compatible_api_key: str | None = None
    deepseek_api_key: str | None = None
    firecrawl_api_key: str | None = None
    ncbi_email: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    provider: ModelProviderName
    model_name: str
    base_url: str | None = None
    api_key: str | None = None


def load_config() -> AppConfig:
    load_dotenv()
    return AppConfig(
        model_provider=os.getenv("MODEL_PROVIDER", "heuristic"),
        model_name=os.getenv("MODEL_NAME", "heuristic-demo"),
        openai_compatible_base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL") or None,
        openai_compatible_api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY") or None,
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
        firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY") or None,
        ncbi_email=os.getenv("NCBI_EMAIL") or None,
    )


def resolve_model_config(provider: str | None = None, model: str | None = None) -> ModelConfig:
    config = load_config()
    selected_provider = (provider or config.model_provider).lower()

    if selected_provider == "heuristic":
        return ModelConfig(provider="heuristic", model_name=model or "heuristic-demo")

    if selected_provider == "deepseek":
        api_key = config.deepseek_api_key or config.openai_compatible_api_key
        if not api_key:
            raise RuntimeError("deepseek_api_key_missing")
        return ModelConfig(
            provider="deepseek",
            model_name=model or "deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key=api_key,
        )

    if selected_provider in {"openai-compatible", "openai_compatible"}:
        if not config.openai_compatible_base_url or not config.openai_compatible_api_key:
            raise RuntimeError("openai_compatible_provider_not_configured")
        return ModelConfig(
            provider="openai-compatible",
            model_name=model or config.model_name,
            base_url=config.openai_compatible_base_url,
            api_key=config.openai_compatible_api_key,
        )

    raise ValueError(f"unsupported_model_provider:{selected_provider}")
