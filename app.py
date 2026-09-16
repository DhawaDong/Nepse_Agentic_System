import logging
from flask import Flask, render_template, request, jsonify

from config import Config
from utils.db import init_db, save_run, get_recent_runs
from agents.orchestrator import Orchestrator
from agents.data_agents import NepseAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = Flask(__name__)
app.config.from_object(Config)

init_db()
orchestrator = Orchestrator()
nepse_agent = NepseAgent()


@app.route("/")
def index():
    recent = get_recent_runs(10)
    return render_template("index.html", recent=recent)


@app.route("/analyze", methods=["POST"])
def analyze():
    target = request.form.get("target", "").strip().upper()
    if not target:
        return render_template("index.html", error="Please enter a NEPSE symbol.",
                                recent=get_recent_runs(10))

    result = orchestrator.run(target=target)
    run_id = save_run(target=target, market="NEPSE", result=result)
    return render_template("result.html", result=result, run_id=run_id)


@app.route("/compare", methods=["GET", "POST"])
def compare():
    if request.method == "GET":
        return render_template("compare.html", results=None)

    raw = request.form.get("symbols", "")
    symbols = [s.strip().upper() for s in raw.split(",") if s.strip()][:3]
    if len(symbols) < 2:
        return render_template("compare.html", results=None,
                                error="Enter 2-3 NEPSE symbols separated by commas, e.g. NABIL,EBL,SCB.")

    results = [orchestrator.run(target=s, explain=False) for s in symbols]
    return render_template("compare.html", results=results, error=None)


@app.route("/screener")
def screener():
    """Simple, fast screeners built directly from today's NEPSE snapshot -
    no per-symbol historical fetch, so this stays quick even with ~250+
    listed securities."""
    movers = nepse_agent.get_top_movers(n=10)
    return render_template("screener.html", movers=movers)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """JSON API: POST {"target": "NABIL"}"""
    payload = request.get_json(silent=True) or {}
    target = (payload.get("target") or "").strip().upper()
    if not target:
        return jsonify({"ok": False, "error": "target is required"}), 400

    result = orchestrator.run(target=target)
    run_id = save_run(target=target, market="NEPSE", result=result)
    return jsonify({"ok": True, "run_id": run_id, "result": result})


@app.route("/api/compare", methods=["POST"])
def api_compare():
    payload = request.get_json(silent=True) or {}
    symbols = [s.strip().upper() for s in (payload.get("symbols") or []) if s.strip()][:5]
    if len(symbols) < 2:
        return jsonify({"ok": False, "error": "provide at least 2 symbols"}), 400
    results = [orchestrator.run(target=s, explain=False) for s in symbols]
    return jsonify({"ok": True, "results": results})


@app.route("/api/screener")
def api_screener():
    n = request.args.get("n", 10, type=int)
    return jsonify(nepse_agent.get_top_movers(n=n))


@app.route("/history")
def history():
    return jsonify(get_recent_runs(50))


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
