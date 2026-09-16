import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ---- Groq (LLM) ----
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    # Groq's free-tier catalog changes often. Override via env var without
    # touching code. "llama-3.3-70b-versatile" is a good general default;
    # "llama-3.1-8b-instant" is faster/cheaper for high-volume agent calls.
    GROQ_MODEL_HEAVY = os.getenv("GROQ_MODEL_HEAVY", "openai/gpt-oss-120b")  #"llama-3.3-70b-versatile")
    GROQ_MODEL_FAST = os.getenv("GROQ_MODEL_FAST", "openai/gpt-oss-20b")  #"llama-3.1-8b-instant")

    # ---- NEPSE data source ----
    # NEPSE has no official public API. This project reads a free,
    # community-maintained static JSON dataset (updated via GitHub Actions,
    # not real-time) published at:
    #   https://shubhamnpk.github.io/yonepse/data/...
    # See agents/data_agents.py::NepseAgent for the exact files it reads and
    # how to point this at a different mirror/fork if this one goes stale.
    NEPSE_DATA_BASE = os.getenv(
        "NEPSE_DATA_BASE",
        "https://shubhamnpk.github.io/yonepse/data",
    )
    # How many of the most recent monthly LTP shards to pull when building
    # a price-history series for technical analysis / forecasting. Each
    # NEPSE month has ~20 trading days, so 3 months gives ~60 data points.
    NEPSE_HISTORY_MONTHS = int(os.getenv("NEPSE_HISTORY_MONTHS", "3"))

    # ---- Optional free-tier keyed APIs (leave blank to use free fallbacks) ----
    NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")  # newsapi.org free dev tier (optional)
    GNEWS_KEY = os.getenv("GNEWS_KEY", "")      # gnews.io free tier (optional)

    # ---- App / DB ----
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data.db")
    CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))

    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "15"))
