"""
Persistent per-position thesis + peak-value tracking.

The problem this solves: a position's original entry rationale, and
how far its value has fallen from its best point, currently only
exist as free text buried somewhere in an ever-growing, append-only
event log -- retrievable only if a future cycle happens to go
searching for it, which it usually doesn't (that's exactly what let
three NVDA spreads round-trip from 60-78% profit back to losses
without ever being explicitly re-evaluated against why they were
opened).

This module makes two specific facts structurally impossible to lose
for as long as a position is open: why it was entered, and its peak
mark-to-market value. It does NOT impose any exit rule, threshold, or
trigger -- see autonomous_agent.py's _build_position_thesis_brief,
which surfaces these facts as context for the agent's own judgment,
never as an instruction to act.

Keyed by the sorted pair of (buy_symbol, sell_symbol) -- stable for a
given spread across cycles, distinct from any other spread even on
the same underlying.
"""
import json
import os
from datetime import datetime, timezone

STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "state", "position_theses.json"
)


def _load() -> dict:
    if not os.path.exists(STORE_PATH):
        return {}
    try:
        with open(STORE_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict) -> None:
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
    tmp_path = STORE_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, STORE_PATH)  # atomic — no risk of a half-written file on crash


def _key(buy_symbol: str, sell_symbol: str) -> str:
    return "|".join(sorted([buy_symbol, sell_symbol]))


def record_entry(buy_symbol: str, sell_symbol: str, underlying: str, rationale: str, entry_net_value: float) -> None:
    """Called right when an 'open' order is submitted (agent_layer/tools.py).
    entry_net_value is the limit_price actually submitted -- for a debit
    spread this is what was paid, so peak tracking and current value use
    the same sign convention throughout."""
    data = _load()
    now = datetime.now(timezone.utc).isoformat()
    data[_key(buy_symbol, sell_symbol)] = {
        "underlying": underlying,
        "buy_symbol": buy_symbol,
        "sell_symbol": sell_symbol,
        "thesis": rationale,
        "entry_time": now,
        "entry_net_value": entry_net_value,
        "peak_net_value": entry_net_value,
        "peak_time": now,
    }
    _save(data)


def clear_entry(buy_symbol: str, sell_symbol: str) -> None:
    """Called when a 'close' order is submitted (agent_layer/tools.py).
    Also self-heals via prune_closed() below if a close happens through
    some path this doesn't see, or an 'open' order never actually fills."""
    data = _load()
    data.pop(_key(buy_symbol, sell_symbol), None)
    _save(data)


def all_records() -> dict:
    return _load()


def set_peak_if_higher(buy_symbol: str, sell_symbol: str, current_net_value: float) -> None:
    data = _load()
    key = _key(buy_symbol, sell_symbol)
    if key in data and current_net_value > data[key].get("peak_net_value", float("-inf")):
        data[key]["peak_net_value"] = current_net_value
        data[key]["peak_time"] = datetime.now(timezone.utc).isoformat()
        _save(data)


def prune_closed(live_keys: set) -> None:
    """Drops any stored thesis whose position is no longer fully open —
    e.g. closed via expiration, or an 'open' order that never filled —
    so stale entries don't accumulate indefinitely."""
    data = _load()
    stale = [k for k in data if k not in live_keys]
    for k in stale:
        data.pop(k)
    if stale:
        _save(data)
