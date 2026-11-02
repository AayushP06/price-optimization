from functools import lru_cache

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str = "sqlite:///./instance/local.db"
    secret_key: str | None = None
    env: str = "development"
    admin_token: str | None = None
    enable_docs: bool | None = None

    @field_validator("database_url")
    @classmethod
    def normalize_postgres_scheme(cls, value: str) -> str:
        if value.startswith("postgres://"):
            return "postgresql://" + value.removeprefix("postgres://")
        return value

    @model_validator(mode="after")
    def set_docs_default(self):
        if self.enable_docs is None:
            self.enable_docs = self.env.casefold() not in {"prod", "production"}
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
