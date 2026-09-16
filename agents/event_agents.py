"""
Event Agents: pull recent headlines relevant to a target/topic.
Source is Google News RSS, which is free and needs no API key.
NEWSAPI_KEY / GNEWS_KEY are optional upgrades if the user adds them.
"""
import logging
import requests
import feedparser
from urllib.parse import quote_plus
from config import Config
from utils.db import cache_get, cache_set

logger = logging.getLogger(__name__)


def _cached(key, ttl, fn):
    hit = cache_get(key, ttl)
    if hit is not None:
        return hit
    result = fn()
    cache_set(key, result)
    return result


def _google_news_rss(query: str, limit: int = 8) -> list:
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    items = []
    for entry in feed.entries[:limit]:
        items.append({
            "title": entry.get("title"),
            "link": entry.get("link"),
            "published": entry.get("published"),
            "source": entry.get("source", {}).get("title") if entry.get("source") else None,
        })
    return items


class NewsAgent:
    """General headlines about a company/ticker/topic."""

    def get(self, query: str, limit: int = 8) -> dict:
        def fetch():
            try:
                if Config.NEWSAPI_KEY:
                    r = requests.get(
                        "https://newsapi.org/v2/everything",
                        params={"q": query, "sortBy": "publishedAt", "pageSize": limit,
                                "apiKey": Config.NEWSAPI_KEY},
                        timeout=Config.REQUEST_TIMEOUT,
                    )
                    r.raise_for_status()
                    arts = r.json().get("articles", [])
                    items = [{"title": a["title"], "link": a["url"], "published": a["publishedAt"],
                              "source": a.get("source", {}).get("name")} for a in arts]
                else:
                    items = _google_news_rss(query, limit)
                return {"ok": True, "query": query, "items": items}
            except Exception as e:
                logger.warning("NewsAgent fallback to RSS after error: %s", e)
                try:
                    return {"ok": True, "query": query, "items": _google_news_rss(query, limit)}
                except Exception as e2:
                    return {"ok": False, "error": str(e2)}

        return _cached(f"news:{query}:{limit}", 600, fetch)


class PoliticalAgent:
    """Political/regulatory headlines relevant to NEPSE."""

    def get(self) -> dict:
        query = "Nepal stock market NEPSE policy OR regulation OR government"
        def fetch():
            try:
                return {"ok": True, "items": _google_news_rss(query, 6)}
            except Exception as e:
                return {"ok": False, "error": str(e)}
        return _cached("political:nepal", 900, fetch)


class EconomicAgent:
    """Macro/economic-news headlines relevant to Nepal."""

    def get(self) -> dict:
        query = "Nepal economy inflation interest rate remittance"
        def fetch():
            try:
                return {"ok": True, "items": _google_news_rss(query, 6)}
            except Exception as e:
                return {"ok": False, "error": str(e)}
        return _cached("economic:nepal", 900, fetch)


class CompanyAgent:
    """Company-specific news for a given ticker/company name."""

    def get(self, company_name: str) -> dict:
        def fetch():
            try:
                return {"ok": True, "items": _google_news_rss(company_name, 8)}
            except Exception as e:
                return {"ok": False, "error": str(e)}
        return _cached(f"company_news:{company_name}", 600, fetch)
