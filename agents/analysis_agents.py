"""
Analysis Agents: turn raw NEPSE data (price series, headlines, sector
peers) into signals. Technical/Correlation are pure numeric (pandas/
numpy). Sector/Sentiment lean on the Groq LLM for qualitative judgement.
"""
import numpy as np
import pandas as pd
from utils.groq_client import chat, chat_json
import json
import logging

logger = logging.getLogger(__name__)


class TechnicalAgent:
    """Classic technical indicators computed from a close-price series."""

    def analyze(self, closes: list) -> dict:
        if not closes or len(closes) < 5:
            return {"ok": False, "error": "Not enough price history for technical analysis"}

        s = pd.Series(closes, dtype=float)

        sma_20 = s.rolling(20).mean()
        sma_50 = s.rolling(50).mean()
        ema_12 = s.ewm(span=12, adjust=False).mean()
        ema_26 = s.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()

        delta = s.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        returns = s.pct_change().dropna()
        volatility_annualized = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else None

        last = float(s.iloc[-1])
        trend = "uptrend" if not np.isnan(sma_20.iloc[-1]) and last > sma_20.iloc[-1] else "downtrend/neutral"

        return {
            "ok": True,
            "last_price": round(last, 2),
            "sma_20": round(float(sma_20.iloc[-1]), 2) if not np.isnan(sma_20.iloc[-1]) else None,
            "sma_50": round(float(sma_50.iloc[-1]), 2) if not np.isnan(sma_50.iloc[-1]) else None,
            "rsi_14": round(float(rsi.iloc[-1]), 2) if not np.isnan(rsi.iloc[-1]) else None,
            "macd": round(float(macd.iloc[-1]), 4),
            "macd_signal": round(float(macd_signal.iloc[-1]), 4),
            "volatility_annualized_pct": round(volatility_annualized * 100, 2) if volatility_annualized else None,
            "trend": trend,
            "period_return_pct": round(((last - float(s.iloc[0])) / float(s.iloc[0])) * 100, 2),
        }


class CorrelationAgent:
    """Correlates a stock's daily returns against its sector-peer average
    return - NEPSE has no published index series in the free dataset this
    app uses, so the peer basket stands in for a benchmark."""

    def analyze(self, stock_closes: list, peer_closes_by_symbol: dict) -> dict:
        if not stock_closes or len(stock_closes) < 5:
            return {"ok": False, "error": "Not enough data for correlation analysis"}

        stock_returns = pd.Series(stock_closes).pct_change().dropna()
        correlations = {}
        for symbol, series in (peer_closes_by_symbol or {}).items():
            if not series or len(series) < 5:
                continue
            peer_returns = pd.Series(series).pct_change().dropna()
            n = min(len(stock_returns), len(peer_returns))
            if n < 5:
                continue
            try:
                corr = float(np.corrcoef(stock_returns.tail(n), peer_returns.tail(n))[0, 1])
                if not np.isnan(corr):
                    correlations[symbol] = round(corr, 3)
            except Exception:
                continue

        avg_corr = round(sum(correlations.values()) / len(correlations), 3) if correlations else None
        return {"ok": True, "peer_correlations": correlations, "avg_sector_correlation": avg_corr}


class SectorAgent:
    """Qualitative sector-context read, via LLM, grounded in today's
    sector-peer performance on NEPSE."""

    def analyze(self, sector: str, company_name: str, peer_avg_change: float, peers: list) -> dict:
        if not sector:
            return {"ok": False, "error": "No sector information available"}

        peer_summary = [
            {"symbol": p.get("symbol"), "percent_change": p.get("percent_change")}
            for p in (peers or [])[:8]
        ]
        system = (
            "You are a NEPSE (Nepal Stock Exchange) sector analyst. Given a company's "
            "inferred sector and how its peers are trading today, give a brief, concrete "
            "read on sector positioning. Be specific and avoid generic filler."
        )
        user = (
            f"Company: {company_name}\nSector: {sector}\n"
            f"Sector peers' % change today: {json.dumps(peer_summary)}\n"
            f"Sector average % change today: {peer_avg_change}\n\n"
            "In 3-4 sentences: how does this sector look today, and what should an "
            "investor watch for?"
        )
        text = chat(system, user, temperature=0.4, max_tokens=300)
        return {"ok": True, "sector": sector, "commentary": text}


class SentimentAgent:
    """LLM-based sentiment scoring over a batch of headlines."""

    def analyze(self, target: str, headlines: list) -> dict:
        if not headlines:
            return {"ok": True, "score": 0, "label": "neutral", "rationale": "No recent headlines found."}

        titles = [h.get("title") for h in headlines if h.get("title")]
        system = (
            "You are a financial news sentiment classifier. Score overall sentiment "
            "toward the given target from -1 (very negative) to 1 (very positive)."
        )
        user = (
            f"Target: {target}\nHeadlines:\n- " + "\n- ".join(titles[:15]) +
            "\n\nReturn JSON with keys: score (float -1..1), label (negative|neutral|positive), "
            "rationale (1-2 sentences)."
        )
        raw = chat_json(system, user, temperature=0.1, max_tokens=250)
        try:
            parsed = json.loads(raw)
            return {"ok": True, **parsed}
        except Exception:
            logger.warning("SentimentAgent could not parse JSON: %s", raw)
            return {"ok": True, "score": 0, "label": "neutral", "rationale": raw}
