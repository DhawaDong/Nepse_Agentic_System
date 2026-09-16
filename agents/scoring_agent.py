"""
Deterministic Scoring Engine + Verification Agent.

Per the project spec: the LLM must never invent the numerical score or
the BUY/HOLD/SELL call. Every number here comes from a plain Python
formula over data other agents already fetched. The LLM (Decision/Final
Agent) only *explains* this output afterward.

Category weights are redistributed from the full 8-category spec
(Fundamental 25% + Valuation 15% + Technical 15% + Sector 10% +
Market 10% + Macro 10% + News 5% + Risk 10%) down to the 6 categories
this app can actually back with verified free data today - Fundamental
and Valuation are dropped because the NEPSE free dataset in use has no
EPS/balance-sheet/financial-statement feed. See README for how to wire
those back in if you find/verify a free source.
"""
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# category -> weight (must sum to 100)
WEIGHTS = {
    "technical": 30,
    "sector": 15,
    "market": 15,
    "sentiment": 15,
    "macro": 10,
    "risk": 15,
}

RECOMMENDATION_BANDS = [
    (90, 100, "STRONG BUY"),
    (75, 90, "BUY"),
    (60, 75, "ACCUMULATE"),
    (45, 60, "HOLD / WATCH"),
    (30, 45, "REDUCE"),
    (0, 30, "AVOID / SELL"),
]


def _clip(x, lo=0, hi=100):
    return max(lo, min(hi, x))


class VerificationAgent:
    """Checks the raw pipeline output for missing/stale/failed data before
    scoring runs. Never invents a substitute value - only flags gaps."""

    def verify(self, context: dict) -> dict:
        warnings = []
        checks = {}

        security = context.get("security", {})
        checks["security_data"] = bool(security and security.get("symbol"))
        if not checks["security_data"]:
            warnings.append("No current security snapshot available for this symbol.")

        price_history = context.get("price_history", {})
        dates = price_history.get("dates", []) if price_history.get("ok") else []
        checks["price_history"] = bool(dates)
        if not checks["price_history"]:
            warnings.append("No price history available - technical analysis and forecast are skipped.")
        elif dates:
            try:
                last_date = datetime.strptime(dates[-1], "%Y-%m-%d")
                if datetime.utcnow() - last_date > timedelta(days=10):
                    warnings.append(f"Price data may be stale - last data point is {dates[-1]}.")
            except Exception:
                pass

        for key in ("sector", "sentiment", "macro"):
            ok = bool(context.get(key, {}).get("ok"))
            checks[key] = ok
            if not ok:
                warnings.append(f"'{key}' data unavailable for this run.")

        volumes = price_history.get("volume", []) if price_history.get("ok") else []
        avg_recent_volume = (sum(volumes[-10:]) / len(volumes[-10:])) if volumes else 0
        low_liquidity = avg_recent_volume < 500  # shares/day, conservative NEPSE threshold
        checks["liquidity_ok"] = not low_liquidity
        if low_liquidity:
            warnings.append(
                f"Low recent trading volume (avg ~{int(avg_recent_volume)} shares/day) - "
                "recommendation is capped regardless of score."
            )

        data_points_available = sum(1 for v in checks.values() if v)
        data_quality_score = round(100 * data_points_available / len(checks), 1)

        return {
            "ok": True,
            "checks": checks,
            "warnings": warnings,
            "data_quality_score": data_quality_score,
            "low_liquidity": low_liquidity,
            "avg_recent_volume": round(avg_recent_volume, 1),
        }


class ScoringAgent:
    def _technical_score(self, technical: dict):
        if not technical.get("ok"):
            return None
        score = 50
        if technical.get("trend") == "uptrend":
            score += 15
        else:
            score -= 10
        macd, macd_signal = technical.get("macd"), technical.get("macd_signal")
        if macd is not None and macd_signal is not None:
            score += 10 if macd > macd_signal else -10
        rsi = technical.get("rsi_14")
        if rsi is not None:
            if 45 <= rsi <= 65:
                score += 15
            elif 30 <= rsi < 45 or 65 < rsi <= 75:
                score += 5
            elif rsi > 75:
                score -= 15
        ret = technical.get("period_return_pct")
        if ret is not None:
            score += _clip(ret, -20, 20) / 20 * 10
        return round(_clip(score), 1)

    def _sector_score(self, sector_peers: dict):
        if not sector_peers.get("ok") or sector_peers.get("peer_avg_percent_change") is None:
            return None
        avg = sector_peers["peer_avg_percent_change"]
        return round(_clip(50 + avg * 5), 1)

    def _market_score(self, breadth: dict):
        if not breadth.get("ok"):
            return None
        score = 50 + (breadth.get("avg_percent_change") or 0) * 5
        total = breadth.get("advancers", 0) + breadth.get("decliners", 0)
        if total:
            score += ((breadth.get("advancers", 0) - breadth.get("decliners", 0)) / total) * 20
        return round(_clip(score), 1)

    def _sentiment_score(self, sentiment: dict):
        if sentiment is None or sentiment.get("score") is None:
            return None
        return round(_clip((sentiment["score"] + 1) / 2 * 100), 1)

    def _macro_score(self, macro: dict):
        if not macro.get("ok"):
            return None
        indicators = macro.get("indicators", {})
        score = 50
        gdp = indicators.get("gdp_growth", {}).get("value")
        inflation = indicators.get("inflation", {}).get("value")
        if gdp is not None:
            score += _clip(gdp * 3, -15, 15)
        if inflation is not None:
            score -= _clip((inflation - 6) * 2, -10, 15)
        return round(_clip(score), 1)

    def _risk_score(self, risk: dict):
        if not risk.get("ok"):
            return None
        metrics = risk.get("metrics", {})
        vol = metrics.get("volatility_annualized_pct")
        drawdown = metrics.get("max_drawdown_pct")
        score = 100
        if vol is not None:
            score -= vol
        if drawdown is not None:
            score -= abs(drawdown) * 0.5
        return round(_clip(score), 1)

    def score(self, technical, sector_peers, breadth, sentiment, macro, risk, verification) -> dict:
        raw = {
            "technical": self._technical_score(technical),
            "sector": self._sector_score(sector_peers),
            "market": self._market_score(breadth),
            "sentiment": self._sentiment_score(sentiment),
            "macro": self._macro_score(macro),
            "risk": self._risk_score(risk),
        }

        available_weight = sum(WEIGHTS[k] for k, v in raw.items() if v is not None)
        if available_weight == 0:
            return {
                "ok": False,
                "error": "No scoreable data available",
                "category_scores": raw,
                "final_score": None,
                "recommendation": "INSUFFICIENT DATA",
                "confidence": 0,
            }

        weighted_sum = sum(raw[k] * WEIGHTS[k] for k, v in raw.items() if v is not None)
        final_score = round(weighted_sum / available_weight, 1)

        recommendation = "INSUFFICIENT DATA"
        for lo, hi, label in RECOMMENDATION_BANDS:
            if lo <= final_score < hi or (hi == 100 and final_score == 100):
                recommendation = label
                break

        capped = False
        if verification.get("low_liquidity") and recommendation in ("STRONG BUY", "BUY", "ACCUMULATE"):
            recommendation = "HOLD / WATCH"
            capped = True

        if not verification.get("checks", {}).get("price_history"):
            recommendation = "INSUFFICIENT DATA"

        confidence = round(
            (available_weight / 100) * 100 * (verification.get("data_quality_score", 100) / 100), 1
        )

        return {
            "ok": True,
            "category_scores": raw,
            "weights_used": {k: WEIGHTS[k] for k in raw if raw[k] is not None},
            "final_score": final_score,
            "recommendation": recommendation,
            "recommendation_capped_by_liquidity": capped,
            "confidence": confidence,
        }
