"""
Thin wrapper around Groq's OpenAI-compatible chat completion endpoint.
Centralising this means every agent gets retry/error handling for free,
and swapping models or providers later only touches this one file.
"""
import logging
from groq import Groq
from config import Config

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        if not Config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
                "and set it as an environment variable."
            )
        _client = Groq(api_key=Config.GROQ_API_KEY)
    return _client


def chat(system_prompt: str, user_prompt: str, model: str = None,
         temperature: float = 0.3, max_tokens: int = 1024) -> str:
    """Single-turn chat completion. Returns plain text, or a clear
    error string on failure (agents should keep degrading gracefully
    rather than crashing the whole pipeline over one LLM hiccup)."""
    model = model or Config.GROQ_MODEL_HEAVY
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.exception("Groq call failed")
        return f"[LLM unavailable: {e}]"


def chat_json(system_prompt: str, user_prompt: str, model: str = None,
              temperature: float = 0.1, max_tokens: int = 1024) -> str:
    """Same as chat() but instructs the model to return raw JSON only.
    Caller is responsible for json.loads + try/except."""
    system_prompt = (
        system_prompt
        + "\n\nRespond with ONLY valid JSON. No markdown fences, no preamble, no commentary."
    )
    return chat(system_prompt, user_prompt, model=model, temperature=temperature, max_tokens=max_tokens)
