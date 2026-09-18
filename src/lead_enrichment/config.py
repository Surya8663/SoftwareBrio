from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_fallback_models: str = Field(
        default="gemini-flash-latest,gemini-flash-lite-latest",
        alias="GEMINI_FALLBACK_MODELS",
    )
    gemini_input_usd_per_1m: float = Field(default=0.15, alias="GEMINI_INPUT_USD_PER_1M")
    gemini_output_usd_per_1m: float = Field(default=0.60, alias="GEMINI_OUTPUT_USD_PER_1M")

    max_pages_per_domain: int = Field(default=8, alias="MAX_PAGES_PER_DOMAIN")
    max_chars_per_page: int = Field(default=12_000, alias="MAX_CHARS_PER_PAGE")
    page_timeout_ms: int = Field(default=45_000, alias="PAGE_TIMEOUT_MS")
    request_retries: int = Field(default=2, alias="REQUEST_RETRIES")
    enable_linkedin_search: bool = Field(default=True, alias="ENABLE_LINKEDIN_SEARCH")

    project_root: Path = Field(default_factory=lambda: Path.cwd())


def load_settings() -> Settings:
    return Settings()
