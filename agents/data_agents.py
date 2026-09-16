"""
Data Agents for NEPSE.

NEPSE has no official public/free API. This module reads a free,
community-maintained static JSON dataset published on GitHub Pages
(updated periodically via GitHub Actions - not real-time):

    {NEPSE_DATA_BASE}/market/status.json      -> {is_open, last_checked}
    {NEPSE_DATA_BASE}/nepse_data.json         -> flat list of ALL securities,
                                                  current snapshot (ltp, change,
                                                  volume, turnover, market_cap...)
    {NEPSE_DATA_BASE}/ltp/manifest.json       -> {availableMonths, availableDays, ...}
    {NEPSE_DATA_BASE}/ltp/monthly/YYYY-MM.json -> per-symbol daily LTP series for
                                                  that month: {dates:[...],
                                                  series: {SYMBOL: [[dateIndex, ltp,
                                                  volume, turnover, trades], ...]}}

If this dataset goes stale or the project is discontinued, point
NEPSE_DATA_BASE (in config.py) at a fork/mirror, or swap the internals of
NepseAgent for another unofficial source (e.g. PyPI: nepse-sdk,
nepseman-api, NepseUnofficialApi). Every method here returns
{"ok": False, "error": ...} on failure instead of raising, so one bad
fetch never takes down the rest of the pipeline.
"""
import logging
import requests
from datetime import datetime
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


def _get_json(path: str):
    url = f"{Config.NEPSE_DATA_BASE}{path}"
    r = requests.get(url, timeout=Config.REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Cheap sector classifier: NEPSE company names follow fairly predictable
# naming conventions (e.g. "... Hydropower ...", "... Laghubitta ...",
# "... Life Insurance ..."), so a keyword match gets a usable sector label
# without needing an extra data source.
# ---------------------------------------------------------------------------
_SECTOR_KEYWORDS = [
    ("Hydropower / Energy", ["hydropower", "hydro", "energy", "jalvidhyut", "urja", "power", "jal bidyut", "vidhyut"]),
    ("Commercial Bank", ["bank limited", "bank ltd"]),
    ("Development Bank", ["development bank", "bikas bank"]),
    ("Microfinance (Laghubitta)", ["laghubitta", "microfinance", "bittiya sanstha"]),
    ("Life Insurance", ["life insurance"]),
    ("Non-Life Insurance", ["insurance co", "insurance company", "general insurance", "reinsurance"]),
    ("Finance Company", ["finance ltd", "finance limited", "finance company"]),
    ("Hotel / Tourism", ["hotel", "resort", "tourism", "cablecar"]),
    ("Manufacturing / Cement", ["cement", "industries", "distillery", "spinning", "paints"]),
    ("Investment / Mutual Fund", ["mutual fund", "fund", "yojana", "scheme", "investment trust"]),
    ("Telecom", ["doorsanchar", "telecom"]),
    ("Trading", ["trading"]),
]


def infer_sector(company_name: str) -> str:
    if not company_name:
        return "Other"
    name = company_name.lower()
    for sector, keywords in _SECTOR_KEYWORDS:
        if any(kw in name for kw in keywords):
            return sector
    return "Other"


class NepseAgent:
    def get_market_status(self) -> dict:
        def fetch():
            try:
                data = _get_json("/market/status.json")
                return {"ok": True, **data}
            except Exception as e:
                logger.warning("NEPSE market status unavailable: %s", e)
                return {"ok": False, "error": str(e)}
        return _cached("nepse_status", 120, fetch)

    def get_all_securities(self) -> dict:
        """Full current snapshot of every listed security. This is the
        single most useful call - most other methods derive from it."""
        def fetch():
            try:
                rows = _get_json("/nepse_data.json")
                by_symbol = {row["symbol"]: row for row in rows}
                return {"ok": True, "count": len(rows), "securities": by_symbol}
            except Exception as e:
                logger.warning("NEPSE securities snapshot unavailable: %s", e)
                return {"ok": False, "error": str(e)}
        return _cached("nepse_all_securities", 300, fetch)

    def get_security(self, symbol: str) -> dict:
        snap = self.get_all_securities()
        if not snap.get("ok"):
            return snap
        row = snap["securities"].get(symbol.upper())
        if not row:
            return {"ok": False, "error": f"Symbol '{symbol}' not found on NEPSE"}
        row = dict(row)
        row["sector"] = infer_sector(row.get("name", ""))
        return {"ok": True, **row}

    def get_market_breadth(self) -> dict:
        """Advancers/decliners/unchanged + average % change, computed from
        the current snapshot - a simple stand-in for a market index."""
        snap = self.get_all_securities()
        if not snap.get("ok"):
            return snap

        def fetch():
            rows = list(snap["securities"].values())
            advancers = sum(1 for r in rows if (r.get("percent_change") or 0) > 0)
            decliners = sum(1 for r in rows if (r.get("percent_change") or 0) < 0)
            unchanged = sum(1 for r in rows if (r.get("percent_change") or 0) == 0)
            changes = [r["percent_change"] for r in rows if r.get("percent_change") is not None]
            avg_change = round(sum(changes) / len(changes), 3) if changes else None
            total_turnover = sum(r.get("turnover") or 0 for r in rows)
            return {
                "ok": True,
                "advancers": advancers,
                "decliners": decliners,
                "unchanged": unchanged,
                "avg_percent_change": avg_change,
                "total_turnover": round(total_turnover, 2),
                "securities_count": len(rows),
            }
        return _cached("nepse_breadth", 300, fetch)

    def get_top_movers(self, n: int = 10) -> dict:
        snap = self.get_all_securities()
        if not snap.get("ok"):
            return snap

        def fetch():
            rows = [r for r in snap["securities"].values() if r.get("percent_change") is not None]
            gainers = sorted(rows, key=lambda r: r["percent_change"], reverse=True)[:n]
            losers = sorted(rows, key=lambda r: r["percent_change"])[:n]
            return {"ok": True, "gainers": gainers, "losers": losers}
        return _cached(f"nepse_movers:{n}", 300, fetch)

    def get_sector_peers(self, symbol: str, limit: int = 15) -> dict:
        """All securities inferred to be in the same sector as `symbol`,
        used for sector-context commentary and as a correlation basket."""
        sec = self.get_security(symbol)
        if not sec.get("ok"):
            return sec
        sector = sec["sector"]
        snap = self.get_all_securities()

        def fetch():
            peers = [
                r for r in snap["securities"].values()
                if infer_sector(r.get("name", "")) == sector and r["symbol"] != symbol.upper()
            ]
            peers = peers[:limit]
            avg_change = None
            changes = [p["percent_change"] for p in peers if p.get("percent_change") is not None]
            if changes:
                avg_change = round(sum(changes) / len(changes), 3)
            return {"ok": True, "sector": sector, "peers": peers, "peer_avg_percent_change": avg_change}
        return _cached(f"nepse_sector_peers:{symbol}:{limit}", 300, fetch)

    def _get_manifest(self) -> dict:
        def fetch():
            try:
                return {"ok": True, **_get_json("/ltp/manifest.json")}
            except Exception as e:
                return {"ok": False, "error": str(e)}
        return _cached("nepse_ltp_manifest", 3600, fetch)

    def _get_monthly_shard(self, month: str) -> dict:
        """month: 'YYYY-MM'. The current month's shard changes daily so it
        gets a short cache TTL; older months are immutable so they're
        cached for a full day."""
        is_current_month = month == datetime.utcnow().strftime("%Y-%m")
        ttl = 600 if is_current_month else 86400

        def fetch():
            try:
                return {"ok": True, **_get_json(f"/ltp/monthly/{month}.json")}
            except Exception as e:
                return {"ok": False, "error": str(e)}
        return _cached(f"nepse_shard:{month}", ttl, fetch)

    def get_price_history(self, symbol: str, months: int = None) -> dict:
        """Builds a chronological {dates, close, volume} series for one
        symbol by stitching together the last `months` monthly LTP shards."""
        months = months or Config.NEPSE_HISTORY_MONTHS
        symbol = symbol.upper()
        manifest = self._get_manifest()
        if not manifest.get("ok"):
            return manifest

        available = manifest.get("availableMonths", [])[-months:]
        if not available:
            return {"ok": False, "error": "No available months in NEPSE LTP manifest"}

        all_dates, all_close, all_volume = [], [], []
        for month in available:
            shard = self._get_monthly_shard(month)
            if not shard.get("ok"):
                continue
            dates = shard.get("dates", [])
            series = shard.get("series", {}).get(symbol)
            if not series:
                continue
            for row in series:
                # row is [dateIndex, ltp, volume, turnover, trades] - some
                # thinly-traded rows (mutual funds) only carry [dateIndex, ltp]
                idx = row[0]
                if idx >= len(dates):
                    continue
                all_dates.append(dates[idx])
                all_close.append(row[1])
                all_volume.append(row[2] if len(row) > 2 else 0)

        if not all_close:
            return {"ok": False, "error": f"No price history found for '{symbol}' in the last {months} month(s)"}

        # sort chronologically (months are stitched in order already, but
        # be defensive)
        paired = sorted(zip(all_dates, all_close, all_volume), key=lambda x: x[0])
        dates, close, volume = zip(*paired)
        return {"ok": True, "symbol": symbol, "dates": list(dates), "close": list(close), "volume": list(volume)}


class MacroAgent:
    """Nepal-level macro indicators from the World Bank's free, keyless API."""

    WB_INDICATORS = {
        "gdp_growth": "NY.GDP.MKTP.KD.ZG",
        "inflation": "FP.CPI.TOTL.ZG",
        "unemployment": "SL.UEM.TOTL.ZS",
    }

    def get_country_macro(self, country_code: str = "NP") -> dict:
        def fetch():
            out = {}
            for name, code in self.WB_INDICATORS.items():
                try:
                    url = (
                        f"https://api.worldbank.org/v2/country/{country_code}"
                        f"/indicator/{code}?format=json&per_page=5&mrnev=1"
                    )
                    r = requests.get(url, timeout=Config.REQUEST_TIMEOUT)
                    r.raise_for_status()
                    payload = r.json()
                    if len(payload) > 1 and payload[1]:
                        latest = payload[1][0]
                        out[name] = {"value": latest.get("value"), "year": latest.get("date")}
                except Exception:
                    continue
            return {"ok": True, "country": country_code, "indicators": out}
        return _cached(f"macro:{country_code}", 3600, fetch)
