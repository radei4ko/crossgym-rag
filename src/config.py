import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    supabase_url: str = os.environ["SUPABASE_URL"]
    supabase_service_key: str = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    openrouter_api_key: str = os.environ["OPENROUTER_API_KEY"]
    openrouter_model: str = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3-next-80b-a3b-instruct")
    # "low" cuts latency ~40% with no quality loss on this task (measured). Empty string = provider default.
    openrouter_reasoning_effort: str = os.environ.get("OPENROUTER_REASONING_EFFORT", "low")
    # Cap on output tokens (incl. reasoning). Without it OpenRouter reserves the model's max output
    # against the credit balance and returns 402 when the balance is low, even for tiny replies.
    openrouter_max_tokens: int = int(os.environ.get("OPENROUTER_MAX_TOKENS", "2000"))
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    api_key: str = os.environ["API_KEY"]
    match_count: int = int(os.environ.get("MATCH_COUNT", "60"))


settings = Settings()
