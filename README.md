# NEPSE Multi-Agent Analyst

A Flask implementation of the multi-agent pipeline, scoped entirely to
the Nepal Stock Exchange (NEPSE) - no global-market data involved:

```
Data Agents (NEPSE security/market breadth/sector peers, Nepal Macro)
Event Agents (News/Political/Economic/Company)
        -> Analysis Agents (Technical/Sector/Sentiment/Correlation)
        -> Verification Agent (data quality + liquidity checks)
        -> Deterministic Scoring Engine (BUY/HOLD/SELL - plain Python, no LLM numbers)
        -> Forecasting Agent (XGBoost)
        -> Risk Agent
        -> Decision / Final Agent (Groq LLM explains the score, doesn't compute it)
```

Plus: a stock **comparison** view, a today's-movers **screener**, and
inline **charts** (price + moving averages, score breakdown) on the
results page.

## What's implemented vs. the full spec

If you're comparing this against a fuller "NEPSE AI Analyzer" spec
(fundamental ratios like EPS/ROE/NPL per sector, P/E-based valuation,
NRB macro scraping, LangGraph, a 10+ table Postgres schema, a full test
suite): those pieces are **not** in this build, on purpose, because they
either need data this project can't currently source for free, or are a
much larger build than fits here:

- **Fundamentals & valuation (EPS, ROE, P/E, P/B, NPL, etc.)** — the
  free NEPSE dataset this app reads (`nepse_data.json` + LTP history)
  has no balance-sheet/income-statement feed. There's no confirmed free
  source for NEPSE per-company financials at the time of writing. Rather
  than fabricate these numbers, the scoring engine's weights are
  redistributed across the 6 categories it *can* back with real data
  (see below). If you find/verify a free financials source, add a
  `FundamentalAgent` alongside `agents/data_agents.py` and re-add its
  weight to `agents/scoring_agent.py::WEIGHTS`.
- **Nepal Rastra Bank (NRB) macro data** — NRB publishes policy
  rates/inflation/liquidity as PDF reports, not a JSON API. Nepal-level
  macro here instead comes from the World Bank's free API (GDP growth,
  inflation, unemployment) — directionally useful, but coarser than
  NRB's own bulletins and updated far less often.
- **LangGraph** — the orchestration is a plain Python
  `agents/orchestrator.py` instead. It implements the same
  fan-out/fan-in flow (data+event agents concurrently, then
  analysis → verification → scoring → forecast → risk → decision) with
  fewer moving parts to deploy on a free tier.
- **Screener** — implemented as a fast "today's movers" view straight
  from the live snapshot (no per-symbol historical fetch, so it stays
  quick across 250+ listed securities). A full RSI/SMA-based technical
  screener across every listed stock would need historical data fetched
  and cached for all of them, which is a background-job job, not a
  per-request one — a good next step if you add a scheduled worker.

## What it uses

| Role | Tool | Cost |
|---|---|---|
| LLM (sentiment, sector commentary, final report) | Groq API (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`) | Free tier, no card |
| NEPSE data | Community static JSON dataset (see below) | Free, unofficial |
| News | Google News RSS | Free, no key |
| Nepal macro data | World Bank API | Free, no key |
| Optional upgrade | NewsAPI.org, GNews.io | Free tier, need a key |
| DB | SQLite via SQLAlchemy | Free |
| Charts | Chart.js (CDN) | Free |

## About the NEPSE data source

NEPSE has no official public API. This app reads a free,
**community-maintained static JSON dataset**, published on GitHub Pages
and refreshed periodically via GitHub Actions (not real-time):

```
{NEPSE_DATA_BASE}/market/status.json        # {is_open, last_checked}
{NEPSE_DATA_BASE}/nepse_data.json           # current snapshot of EVERY listed
                                             # security (ltp, change %, volume,
                                             # turnover, market cap, ...)
{NEPSE_DATA_BASE}/ltp/manifest.json         # which months/days of history exist
{NEPSE_DATA_BASE}/ltp/monthly/YYYY-MM.json  # per-symbol daily LTP series for that month
```

`agents/data_agents.py::NepseAgent` wraps these:
- `get_security(symbol)` — current snapshot + an inferred sector label
  (NEPSE company names are matched against sector keywords like
  "Hydropower", "Laghubitta", "Life Insurance" — there's no sector field
  in the raw data, so this is a heuristic, not authoritative).
- `get_market_breadth()` — advancers/decliners/average % change across
  every listed security, used as a stand-in for a NEPSE index (the free
  dataset doesn't publish an aggregate index value).
- `get_sector_peers(symbol)` — other securities in the same inferred
  sector, used for sector commentary and as the correlation benchmark.
- `get_price_history(symbol)` — stitches together the last
  `NEPSE_HISTORY_MONTHS` monthly shards into one chronological series for
  technical analysis and forecasting.

**If this dataset goes stale or the GitHub Pages site changes:** point
`NEPSE_DATA_BASE` at a fork/mirror, or swap `NepseAgent`'s internals for
another unofficial source (PyPI: `nepse-sdk`, `nepseman-api`,
`NepseUnofficialApi`). Every agent method returns
`{"ok": False, "error": ...}` on failure instead of raising, so the rest
of the pipeline keeps working even if NEPSE data is temporarily
unavailable.

## Local setup

```bash
git clone <your-repo-url>
cd nepse-multiagent
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your free Groq key from https://console.groq.com
python app.py
```

Visit `http://localhost:5000` and enter a NEPSE symbol like `NABIL`,
`NHPC`, or `HIDCL`.

## Getting a free Groq key

1. Go to https://console.groq.com, sign in (no credit card).
2. Create an API key.
3. Set it as `GROQ_API_KEY` in `.env` (local) or in Render's environment
   variables (deployed).
4. Groq's free-tier model catalog and rate limits change fairly often —
   check https://console.groq.com/docs/models if `GROQ_MODEL_HEAVY` /
   `GROQ_MODEL_FAST` in `.env` stop working, and swap in whatever's
   currently free.

## Deploying to Render.com

### Option A — Blueprint (`render.yaml`), recommended
1. Push this project to a GitHub repo.
2. In Render: **New > Blueprint**, connect the repo. Render reads `render.yaml`
   and creates the web service automatically.
3. Fill in the secret env vars it prompts for (`GROQ_API_KEY` at minimum).
4. Deploy.

### Option B — Manual web service
1. Push to GitHub.
2. Render: **New > Web Service**, connect the repo.
3. Environment: `Python 3`. Build command: `pip install -r requirements.txt`.
   Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120`.
4. Add environment variables (see `.env.example`) in the Render dashboard.
5. Deploy.

### Notes for Render's free tier
- Free web services spin down after inactivity — the first request after
  idling will be slow (cold start + first LLM/data calls).
- SQLite (`data.db`) lives on the container's ephemeral disk — it resets
  on redeploy. Fine for caching/demo; for persistent history use Render's
  managed Postgres (free tier available) and change `DATABASE_URL`.
- `get_price_history` fetches a few monthly JSON files per request (each
  a few hundred KB, covering every NEPSE-listed security for that
  month) — this is cached, but the first request for a new month will be
  slower than subsequent ones.

## Extending it

- **Add a data/event agent**: subclass the pattern in `agents/data_agents.py`
  or `agents/event_agents.py` — return `{"ok": True/False, ...}`, wrap the
  fetch in `_cached(...)` for free caching.
- **Swap the forecaster**: `agents/forecasting_agent.py` exposes
  `BaseForecaster.predict_next()`. XGBoost is the default because it trains
  in-request in well under a second; an LSTM/Transformer needs pre-trained
  weights loaded from disk if you want to use it on Render's free tier
  (no GPU, limited RAM/CPU — training per-request will time out).
- **Better sector data**: the sector classifier is a keyword heuristic on
  company names. If you find a source with authoritative NEPSE sector
  mappings, swap `infer_sector()` in `data_agents.py` for a lookup.
- **Background refresh**: for a NEPSE market-breadth cache that updates on
  a schedule instead of per-request, add APScheduler and call
  `NepseAgent().get_market_breadth()` on a timer.
- **Tune the scoring engine**: category weights, RSI/trend/MACD scoring
  rules, recommendation bands, and the liquidity gate all live in
  `agents/scoring_agent.py` as plain constants/formulas — no retraining
  needed to adjust them.
- **Full technical screener**: `NepseAgent.get_price_history()` already
  fetches per-symbol history; a scheduled job that runs it for every
  listed symbol once and caches the result would let `/screener` add
  RSI/SMA-based filters without slowing down the request path.

## Project layout

```
app.py                     Flask routes
config.py                  Env-driven config
agents/
  orchestrator.py          Wires all agents together per request
  data_agents.py           NEPSE (security/breadth/sector peers/history) + Nepal Macro
  event_agents.py          News / Political / Economic / Company
  analysis_agents.py       Technical / Sector / Sentiment / Correlation
  scoring_agent.py         Verification Agent + deterministic Scoring Engine (BUY/HOLD/SELL)
  forecasting_agent.py     XGBoost forecaster (+ naive fallback)
  risk_agent.py            Volatility / VaR / drawdown / Sharpe + narrative
  final_agent.py           Decision Agent - Groq LLM explains the score, doesn't compute it
utils/
  groq_client.py           Groq chat wrapper
  db.py                    SQLite models + cache helpers
templates/                 Server-rendered UI (index, result w/ charts, compare, screener)
render.yaml, Procfile      Render deployment
```
