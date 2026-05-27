from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    model_provider: str = "heuristic"
    model_name: str = "heuristic-demo"
    openai_compatible_base_url: str | None = None
    openai_compatible_api_key: str | None = None
    firecrawl_api_key: str | None = None
    ncbi_email: str | None = None


def load_config() -> AppConfig:
    load_dotenv()
    return AppConfig(
        model_provider=os.getenv("MODEL_PROVIDER", "heuristic"),
        model_name=os.getenv("MODEL_NAME", "heuristic-demo"),
        openai_compatible_base_url=os.getenv("OPENAI_COMPATIBLE_BASE_URL") or None,
        openai_compatible_api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY") or None,
        firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY") or None,
        ncbi_email=os.getenv("NCBI_EMAIL") or None,
    )
