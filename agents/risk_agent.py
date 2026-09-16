"""Risk Agent: quantitative risk metrics + LLM narrative."""
import numpy as np
import pandas as pd
from utils.groq_client import chat


class RiskAgent:
    def analyze(self, closes: list, forecast: dict, sentiment: dict) -> dict:
        if not closes or len(closes) < 10:
            return {"ok": False, "error": "Not enough data for risk analysis"}

        s = pd.Series(closes, dtype=float)
        returns = s.pct_change().dropna()

        volatility_annualized = float(returns.std() * np.sqrt(252))
        mean_return_annualized = float(returns.mean() * 252)
        sharpe = round(mean_return_annualized / volatility_annualized, 2) if volatility_annualized else None

        # Historical 1-day 95% VaR (loss, positive number = % of position)
        var_95 = float(-np.percentile(returns, 5)) if len(returns) > 20 else None

        # Max drawdown
        cummax = s.cummax()
        drawdown = (s - cummax) / cummax
        max_drawdown = float(drawdown.min())

        metrics = {
            "volatility_annualized_pct": round(volatility_annualized * 100, 2),
            "sharpe_ratio_est": sharpe,
            "value_at_risk_95_1day_pct": round(var_95 * 100, 2) if var_95 is not None else None,
            "max_drawdown_pct": round(max_drawdown * 100, 2),
        }

        risk_level = "low"
        if metrics["volatility_annualized_pct"] > 40 or (metrics["max_drawdown_pct"] and metrics["max_drawdown_pct"] < -25):
            risk_level = "high"
        elif metrics["volatility_annualized_pct"] > 20:
            risk_level = "medium"

        system = (
            "You are a risk analyst. Be concise, concrete, and avoid hedging every "
            "sentence. State the risk level and the 1-2 biggest drivers of it."
        )
        user = (
            f"Risk metrics: {metrics}\n"
            f"Forecast confidence: {forecast.get('confidence')}\n"
            f"News sentiment: {sentiment.get('label')} ({sentiment.get('score')})\n\n"
            "In 3-4 sentences, explain the risk profile and what could invalidate it."
        )
        narrative = chat(system, user, temperature=0.3, max_tokens=250)

        return {"ok": True, "risk_level": risk_level, "metrics": metrics, "narrative": narrative}
