import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    supabase_url: str = os.environ["SUPABASE_URL"]
    supabase_service_key: str = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    openrouter_api_key: str = os.environ["OPENROUTER_API_KEY"]
    openrouter_model: str = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-3.5-haiku")
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    api_key: str = os.environ["API_KEY"]
    match_count: int = int(os.environ.get("MATCH_COUNT", "5"))


settings = Settings()
