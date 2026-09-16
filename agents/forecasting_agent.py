"""
Forecasting Agent.

Default model: XGBoost regression on lag/rolling features, trained
on-the-fly from the fetched price history. This is deliberately the
default because it trains in well under a second on a few hundred rows
and needs no GPU - a good fit for Render's free web-service tier.

LSTM / Transformer paths are stubbed behind the same interface
(`BaseForecaster.predict_next`) so you can swap in a real deep-learning
model later (e.g. via torch) without touching the orchestrator. Training
a real LSTM/Transformer per-request is NOT recommended on Render's free
tier (no GPU, 512MB RAM, requests will time out) - use them only if you
pre-train offline and load saved weights.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


def _make_features(closes: list, n_lags: int = 5) -> pd.DataFrame:
    # Build every column in one dict first, then construct the DataFrame in
    # a single shot. Avoids incremental df["col"] = ... assignment, which
    # trips pandas' Copy-on-Write "chained assignment" warning on some
    # pandas/Python builds even though the code itself isn't chained.
    s = pd.Series(closes, dtype=float)
    columns = {"close": s}
    for lag in range(1, n_lags + 1):
        columns[f"lag_{lag}"] = s.shift(lag)
    columns["sma_5"] = s.rolling(5).mean()
    columns["sma_10"] = s.rolling(10).mean()
    columns["momentum_5"] = s - s.shift(5)
    columns["target_next"] = s.shift(-1)
    df = pd.DataFrame(columns)
    return df.dropna()


class BaseForecaster:
    name = "base"

    def predict_next(self, closes: list, horizon: int = 5) -> dict:
        raise NotImplementedError


class NaiveForecaster(BaseForecaster):
    """Fallback: simple drift model. Used if XGBoost isn't installed or
    there isn't enough history to train anything smarter."""
    name = "naive-drift"

    def predict_next(self, closes: list, horizon: int = 5) -> dict:
        s = pd.Series(closes, dtype=float)
        avg_daily_change = s.diff().tail(10).mean() if len(s) > 10 else s.diff().mean()
        avg_daily_change = 0.0 if pd.isna(avg_daily_change) else float(avg_daily_change)
        last = float(s.iloc[-1])
        preds = [round(last + avg_daily_change * i, 2) for i in range(1, horizon + 1)]
        return {
            "ok": True,
            "model": self.name,
            "last_price": round(last, 2),
            "predicted_path": preds,
            "predicted_next": preds[0],
            "confidence": "low",
        }


class XGBoostForecaster(BaseForecaster):
    name = "xgboost"

    def predict_next(self, closes: list, horizon: int = 5) -> dict:
        df = _make_features(closes)
        if len(df) < 20:
            return NaiveForecaster().predict_next(closes, horizon)

        feature_cols = [c for c in df.columns if c not in ("target_next",)]
        X, y = df[feature_cols], df["target_next"]

        model = xgb.XGBRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror",
            n_jobs=2,
        )
        model.fit(X, y)

        # Recursive multi-step forecast: predict one step, append, refeat.
        history = list(closes)
        preds = []
        for _ in range(horizon):
            feat_df = _make_features(history)
            if feat_df.empty:
                break
            last_row = feat_df[feature_cols].iloc[[-1]]
            next_val = float(model.predict(last_row)[0])
            preds.append(round(next_val, 2))
            history.append(next_val)

        if not preds:
            return NaiveForecaster().predict_next(closes, horizon)

        # crude confidence heuristic from in-sample residual spread
        in_sample_pred = model.predict(X)
        resid_std = float(np.std(y.values - in_sample_pred))
        last_price = float(closes[-1])
        confidence = "high" if resid_std < 0.01 * last_price else (
            "medium" if resid_std < 0.03 * last_price else "low"
        )

        return {
            "ok": True,
            "model": self.name,
            "last_price": round(last_price, 2),
            "predicted_path": preds,
            "predicted_next": preds[0],
            "residual_std": round(resid_std, 3),
            "confidence": confidence,
        }


class ForecastingAgent:
    def __init__(self):
        self.forecaster = XGBoostForecaster() if _HAS_XGB else NaiveForecaster()
        if not _HAS_XGB:
            logger.warning("xgboost not installed - falling back to naive drift forecaster")

    def forecast(self, closes: list, horizon: int = 5) -> dict:
        if not closes or len(closes) < 5:
            return {"ok": False, "error": "Not enough price history to forecast"}
        try:
            return self.forecaster.predict_next(closes, horizon=horizon)
        except Exception as e:
            logger.exception("Forecasting failed, falling back to naive")
            try:
                return NaiveForecaster().predict_next(closes, horizon=horizon)
            except Exception as e2:
                return {"ok": False, "error": str(e2)}
