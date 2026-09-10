"""
Structured JSONL logger. Every fast-layer signal, agent decision, risk
governor verdict, and order event gets a line here. This is what the
demo dashboard reads to show the agent's reasoning trail — the thing
hackathon judges actually want to see, not just a PnL number.
"""
import json
import os
from contextvars import ContextVar
from datetime import datetime, timezone

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "events.jsonl")

# Set once per run_cycle() invocation (see agent_layer/autonomous_agent.py)
# so every event logged during that cycle — including from deep inside
# tools.py's tool dispatch — carries the same cycle_id, without having to
# thread it through every single call site by hand. Exists specifically
# to make it possible to tell apart two genuinely concurrent cycles from
# one cycle's events merely being interleaved in time with another
# process's (e.g. the separate hackathon bot) — a real, unexplained
# anomaly observed once and never root-caused; this is the tool to
# diagnose it if it happens again, not a fix for a confirmed cause.
_current_cycle_id: ContextVar[str] = ContextVar("current_cycle_id", default="")


def set_cycle_id(cycle_id: str) -> None:
    _current_cycle_id.set(cycle_id)


def log_event(event_type: str, payload: dict):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "cycle_id": _current_cycle_id.get(),
        "payload": payload,
    }
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_events(limit: int = 200) -> list:
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r") as f:
        lines = f.readlines()[-limit:]
    return [json.loads(line) for line in lines]

