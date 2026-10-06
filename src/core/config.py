from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated environment configuration shared by application components."""

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.1-flash-lite"
    bge_embedding_model: str = "BAAI/bge-base-en-v1.5"
    persist_directory: str = ".chroma"
    collection_name: str = "arxiv_abstracts_parent_child"
    semantic_cache_collection: str = "semantic_query_cache"
    log_level: str = "INFO"
    log_json: bool = False
    log_file: str = "logs/researchmind.log"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
