"""Decision / Final Agent: explains the deterministic score - it does not
compute or override it. The Groq LLM's job here is purely interpretation:
identify the strongest positive/negative factors, resolve conflicts
between signals, and write the report in plain English."""
import json
from config import Config
from utils.groq_client import chat


class FinalAgent:
    def explain(self, context: dict) -> dict:
        system = (
            "You are the Decision Agent in a NEPSE stock analysis system. You are given "
            "a deterministic score, a recommendation, and structured evidence from "
            "specialized agents (technical, sector, sentiment, market breadth, macro, "
            "risk). Your job is ONLY to interpret and explain this evidence - you must "
            "NOT invent, recalculate, or override any number given to you. If a field is "
            "null or missing, say the data was unavailable rather than guessing a value. "
            "Be direct about uncertainty and conflicting signals. This is a NEPSE "
            "decision-support tool, not financial advice - say that once, briefly, at "
            "the end."
        )
        user = (
            "Structured evidence and deterministic score (JSON):\n\n"
            f"{json.dumps(context, default=str, indent=2)[:6500]}\n\n"
            "Write the report with these sections: Summary, Key Positive Factors, "
            "Key Risk Factors, Technical Read, Sentiment & News, Sector Context, "
            "Data Limitations, Bottom Line. Reference the given score/recommendation "
            "exactly as provided - do not state a different number."
        )
        text = chat(system, user, model=Config.GROQ_MODEL_HEAVY, temperature=0.35, max_tokens=1300)
        return {"ok": True, "report": text}
