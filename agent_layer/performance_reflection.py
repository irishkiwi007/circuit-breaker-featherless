"""
Closes the gap between "the agent logs real outcomes" and "the agent's
own future reasoning is actually informed by those outcomes."

Architectural note (why this exists instead of wiring up
agent_layer/rules_review_agent.py): that module was built for the
older, now-unused main.py path, which had a fixed mechanical pre-filter
with tunable numeric parameters (vix_entry_threshold, min_iv_rank,
etc.). The live autonomous loop (agent_layer/autonomous_agent.py) has
no such parameters -- Claude originates every decision fresh each
cycle, bounded only by the two hard backstops in agent_layer/tools.py.
There is nothing for a parameter-tuning agent to tune in that
architecture, so wiring up rules_review_agent.py as-is would be a
second dead end, not a fix.

What this does instead: periodically computes realized P&L on actually
closed trades (matching each close's agent_order_submit event back to
its opening one by underlying + leg symbols -- the same matching logic
already proven in streamlit_app.py's reasoning display), and asks
Claude for a short, honest reflection on whatever pattern is actually
visible in that data. That reflection is written to
PERFORMANCE_REFLECTION, which agent_layer/autonomous_agent.py picks up
and injects into the next cycle's opening message -- the same channel
already built for manual operator notes, just fed by real data instead
of a human typing it in.

This changes nothing about position sizing, risk limits, or the two
hard backstops. It can only ever add a paragraph of text to what
Claude reads before it reasons about the next cycle -- the same kind
of influence a human operator note already has, which was already
judged acceptable. It cannot touch execution or risk code, and every
reflection it generates is logged in full, so "what did it tell itself
and why" is as auditable as every other decision in this system.
"""
import json
import os
from datetime import datetime, timezone

from config import CONFIG
from agent_layer.llm_client import get_client
from execution.trade_logger import log_event, read_events

REFLECTION_NOTE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PERFORMANCE_REFLECTION"
)
REFLECTION_STATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "reflection_state.json"
)

MIN_NEW_CLOSED_TRADES = 3


def _load_state() -> dict:
    if not os.path.exists(REFLECTION_STATE_PATH):
        return {"last_reflected_closed_trade_count": 0}
    with open(REFLECTION_STATE_PATH, "r") as f:
        return json.load(f)


def _save_state(state: dict):
    os.makedirs(os.path.dirname(REFLECTION_STATE_PATH), exist_ok=True)
    with open(REFLECTION_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _extract_closed_trades(limit: int = 1000) -> list:
    """
    Matches each 'close' agent_order_submit event back to the 'open'
    event for the same underlying with overlapping leg symbols -- a
    trade can only be closed by trading the exact legs it was opened
    with, so this is an unambiguous match, not a timestamp guess.

    Realized P&L is approximated as (close credit - open debit) *
    contracts * 100, using each event's own limit_price. This is the
    price the agent intended, not necessarily the exact fill price
    (fills aren't separately logged here) -- close enough to spot a
    real pattern, not precise enough to be quoted as exact accounting.
    That caveat is passed into the reflection prompt itself so Claude
    doesn't overstate precision it doesn't have.
    """
    events = read_events(limit=limit)
    order_events = [
        e["payload"] for e in events
        if e.get("event_type") == "agent_order_submit"
    ]

    opens = [e for e in order_events if e.get("action") == "open"]
    closes = [e for e in order_events if e.get("action") == "close"]

    closed_trades = []
    used_open_indices = set()
    for close in closes:
        close_symbols = {close.get("buy_symbol"), close.get("sell_symbol")}
        for i, open_ in enumerate(opens):
            if i in used_open_indices:
                continue
            if open_.get("underlying") != close.get("underlying"):
                continue
            open_symbols = {open_.get("buy_symbol"), open_.get("sell_symbol")}
            if open_symbols & close_symbols:
                used_open_indices.add(i)
                contracts = open_.get("contracts") or 0
                open_price = open_.get("limit_price") or 0
                close_price = close.get("limit_price") or 0
                pnl = (close_price - open_price) * contracts * 100
                closed_trades.append({
                    "underlying": open_.get("underlying"),
                    "contracts": contracts,
                    "open_price": open_price,
                    "close_price": close_price,
                    "approx_pnl": round(pnl, 2),
                    "open_rationale": (open_.get("rationale") or "")[:200],
                    "close_rationale": (close.get("rationale") or "")[:200],
                })
                break

    return closed_trades


class PerformanceReflectionAgent:
    def __init__(self, config=CONFIG):
        self.config = config
        self._client = get_client(config)

    def maybe_generate_reflection(self) -> dict:
        """
        Checks whether enough new closed-trade data exists since the
        last reflection; if so, generates one and writes it for the
        next cycle to pick up. Always logs what it decided, including
        'skipped, not enough new data' -- same transparency standard
        as every other decision in this system. Never raises -- a
        failure here must never be able to affect the trading loop
        calling it.
        """
        try:
            return self._run()
        except Exception as e:
            record = {"generated": False, "reason": f"reflection generation failed: {e}"}
            log_event("performance_reflection_error", record)
            return record

    def _run(self) -> dict:
        if not self.config.claude.api_key:
            record = {"generated": False, "reason": "ANTHROPIC_API_KEY not configured"}
            log_event("performance_reflection_skipped", record)
            return record

        closed_trades = _extract_closed_trades()
        state = _load_state()
        already_reflected = state.get("last_reflected_closed_trade_count", 0)
        new_trades = closed_trades[already_reflected:]

        if len(new_trades) < MIN_NEW_CLOSED_TRADES:
            record = {
                "generated": False,
                "reason": f"only {len(new_trades)} new closed trade(s) since last reflection "
                          f"(need {MIN_NEW_CLOSED_TRADES}); skipping to avoid drawing conclusions "
                          f"from too small a sample",
            }
            log_event("performance_reflection_skipped", record)
            return record

        total_pnl = sum(t["approx_pnl"] for t in new_trades)
        wins = sum(1 for t in new_trades if t["approx_pnl"] > 0)
        losses = sum(1 for t in new_trades if t["approx_pnl"] <= 0)

        trades_text = "\n".join(
            f"- {t['underlying']}: opened at ${t['open_price']:.2f}, closed at ${t['close_price']:.2f}, "
            f"~${t['approx_pnl']:.2f} P&L ({t['contracts']} contracts). "
            f"Opened because: {t['open_rationale']} "
            f"Closed because: {t['close_rationale']}"
            for t in new_trades
        )

        prompt = f"""You are reviewing your own recent real trading history -- not a backtest, actual paper trades with real outcomes.

New closed trades since your last self-review ({len(new_trades)} trades, {wins} win(s), {losses} loss(es), approximate total P&L ${total_pnl:.2f}):

{trades_text}

Note: P&L figures are approximate, computed from your own logged limit prices at open and close, not separately-recorded fill prices -- treat them as directionally accurate, not exact accounting.

Write a short, honest reflection (3-5 sentences) on whatever pattern is actually visible in this specific data -- not generic trading advice. If you see something worth being more careful about next cycle (a setup type that's underperforming, a rationale pattern that didn't hold up, timing that's been off), say so plainly. If the sample is too mixed or too small to support a real conclusion, say that too rather than inventing a pattern. This reflection will be shown to you at the start of your next decision cycle as context, not as an instruction -- you're free to weigh it however your judgment says to."""

        response = self._client.messages.create(
            model=self.config.claude.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        reflection_text = "".join(b.text for b in response.content if hasattr(b, "text")).strip()

        record = {
            "generated": True,
            "new_trade_count": len(new_trades),
            "wins": wins,
            "losses": losses,
            "approx_total_pnl": round(total_pnl, 2),
            "reflection": reflection_text,
        }
        log_event("performance_reflection_generated", record)

        with open(REFLECTION_NOTE_PATH, "w") as f:
            f.write(reflection_text)

        _save_state({"last_reflected_closed_trade_count": len(closed_trades)})

        return record
