from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Server
    port: int = 8001

    # PostgreSQL / pgvector
    pgvector_host: str = "localhost"
    pgvector_port: int = 5432
    pgvector_db: str = "sheltertech"
    pgvector_user: str = "postgres"
    pgvector_password: str = "mypassword"
    pgvector_table: str = "service_snapshots"

    # Connection pool
    db_pool_min: int = 2
    db_pool_max: int = 10
    db_pool_timeout: float = 30.0

    # Ollama
    ollama_base_url: str = "http://172.26.64.1:11434"
    ollama_embedding_model: str = "nomic-embed-text"


settings = Settings()
