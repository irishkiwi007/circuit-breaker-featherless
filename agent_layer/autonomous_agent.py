"""
Runs one autonomous decision cycle: Claude is given the tool set from
agent_layer.tools and drives its own tool-calling conversation — check
data, decide, trade, assess, adjust — until it stops requesting tools
and writes a final summary including its chosen interval until the
next cycle.

This differs fundamentally from agent_layer/claude_agent.py (which
only reviews a pre-generated candidate) and agent_layer/
rules_review_agent.py (which only adjusts whitelisted config values).
Here, Claude originates everything: what to look at, what to trade,
when, and how much, bounded only by the two hard backstops enforced
inside agent_layer/tools.py itself.
"""
import re
import os
from datetime import datetime, timezone

from config import CONFIG
from agent_layer.llm_client import get_client
from agent_layer.tools import TOOL_SCHEMAS, ToolDispatcher
from agent_layer.autonomous_prompts import AUTONOMOUS_AGENT_SYSTEM_PROMPT
from agent_layer.remote_feedback import fetch_and_consume_remote_notes
from agent_layer import position_memory
from execution.alpaca_client import AlpacaExecutionClient
from execution.trade_logger import log_event, set_cycle_id

DEFAULT_NEXT_CHECK_MINUTES = 15
MAX_TOOL_ROUNDS_PER_CYCLE = 15  # lowered from 25 -- credit/cost control: every round both resends the whole growing conversation AND costs a separate billed request
OPERATOR_NOTE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OPERATOR_NOTE")
PERFORMANCE_REFLECTION_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PERFORMANCE_REFLECTION")


def _consume_operator_note() -> str:
    """
    Combines the local OPERATOR_NOTE file (see deploy/DEPLOY.md — write
    it directly when SSHed into the VM) with any remote notes submitted
    via the dashboard's passcode-gated feedback box (see
    agent_layer/remote_feedback.py). Both are one-shot: the local file
    is deleted after reading, and remote notes arrive as GitHub issues
    that get closed immediately after being read, so neither channel
    re-injects the same note on a later cycle.
    """
    parts = []

    if os.path.exists(OPERATOR_NOTE_PATH):
        with open(OPERATOR_NOTE_PATH, "r") as f:
            local_note = f.read().strip()
        os.remove(OPERATOR_NOTE_PATH)
        if local_note:
            parts.append(local_note)

    remote_note = fetch_and_consume_remote_notes()
    if remote_note:
        parts.append(remote_note)

    return "\n\n".join(parts)


def _read_performance_reflection() -> str:
    """
    Reads the most recent auto-generated performance reflection (see
    agent_layer/performance_reflection.py), if one exists. Unlike the
    operator note, this is NOT deleted after reading — a reflection on
    real trading outcomes is meant to inform judgment across many
    cycles until a newer one supersedes it, not just the next one.
    """
    if not os.path.exists(PERFORMANCE_REFLECTION_PATH):
        return ""
    with open(PERFORMANCE_REFLECTION_PATH, "r") as f:
        return f.read().strip()


async def _build_position_thesis_brief(config) -> str:
    """
    Thin safety wrapper — see _build_position_thesis_brief_inner for the
    actual logic. Nothing in this new-tonight code path may ever be able
    to crash a cycle; a prior version only protected the first API call,
    which let an unhandled exception elsewhere in this function silently
    abort an entire cycle (no autonomous_cycle_end logged at all) during
    a live Alpaca data outage — confirmed via cycle_ids bf89ad42 and
    0649c051 on 2026-09-11.
    """
    try:
        return await _build_position_thesis_brief_inner(config)
    except Exception as exc:
        log_event("position_thesis_brief_failed", {"error": str(exc)})
        return ""


async def _build_position_thesis_brief_inner(config) -> str:
    """
    Structurally resurfaces, every cycle, the two facts a genuine
    re-evaluation of an open position needs and that free-text log
    history reliably loses track of: why it was entered, and how far
    its value has fallen from its best point. This is deliberately
    NOT an exit rule or a threshold of any kind — nothing here tells
    the agent what to decide, only what's true. The decision, sizing,
    and timing remain entirely the agent's own judgment call.

    Computed directly from live Alpaca data each cycle, not from the
    agent's own memory of past cycles, which is exactly the channel
    that was losing this information before.
    """
    live_positions = await AlpacaExecutionClient(config).open_positions()

    if not live_positions:
        return ""

    live_symbols = {p.get("symbol") for p in live_positions}
    by_symbol = {p.get("symbol"): p for p in live_positions}
    theses = position_memory.all_records()

    live_keys = {
        key for key, r in theses.items()
        if r["buy_symbol"] in live_symbols and r["sell_symbol"] in live_symbols
    }
    position_memory.prune_closed(live_keys)
    if not live_keys:
        return ""

    lines = [
        "Your open positions' original theses (peak value vs. current — this is context for your own "
        "re-evaluation, not an instruction; you decide whether each thesis still holds):"
    ]
    for key in live_keys:
        try:
            record = theses[key]
            long_pos = by_symbol.get(record["buy_symbol"])
            short_pos = by_symbol.get(record["sell_symbol"])
            try:
                current_value = float(long_pos.get("current_price", 0) or 0) - float(short_pos.get("current_price", 0) or 0)
            except (TypeError, ValueError, AttributeError):
                current_value = None

            if current_value is not None:
                position_memory.set_peak_if_higher(record["buy_symbol"], record["sell_symbol"], current_value)

            # Re-read after the possible peak update above, so the line
            # reflects the true current peak rather than a stale one.
            peak_value = position_memory.all_records().get(key, record)["peak_net_value"]
            entry_value = record["entry_net_value"]

            line = (
                f"- {record['underlying']} ({record['buy_symbol']}/{record['sell_symbol']}): "
                f"entered at ${entry_value:.2f}/spread because \"{record['thesis']}\". "
                f"Peak value since entry: ${peak_value:.2f}/spread"
                + (f", currently ${current_value:.2f}/spread" if current_value is not None else " (current value unavailable this cycle)")
            )
            if current_value is not None and peak_value > entry_value and current_value < peak_value:
                given_back = peak_value - current_value
                line += f". Has given back ${given_back:.2f}/spread from its peak."
            lines.append(line)
        except Exception as exc:
            # One position's data being malformed (e.g. mid-outage,
            # partial/unexpected API response) must never be able to take
            # down the whole cycle -- this new code had exactly that gap
            # until now: only the initial fetch was protected, nothing
            # in this per-position processing loop was.
            log_event("position_thesis_brief_line_failed", {"key": key, "error": str(exc)})
            continue

    return "\n".join(lines)


KEEP_FULL_TOOL_RESULT_ROUNDS = 6  # only the most recent N rounds' raw tool results stay in
# full context; older ones get compacted. Real cost driver on a long cycle: every round
# resends the ENTIRE conversation so far, so full tool-result JSON from round 1 (option
# chains, position dumps, etc.) was still being resent at full size in round 20, growing
# every single round for the rest of the cycle — the dominant contributor to per-cycle
# token/credit cost, not any individual call.


def _compact_old_round_if_needed(messages: list, current_round_num: int) -> None:
    """
    Once a round falls more than KEEP_FULL_TOOL_RESULT_ROUNDS behind the current one,
    truncate its tool_result content in place. Leaves the assistant's own reasoning text
    for that round untouched (cheap, and carries the actual decision narrative forward) —
    only the bulky raw tool-call JSON gets compacted, and only once each round ever needs it.
    """
    round_to_compact = current_round_num - KEEP_FULL_TOOL_RESULT_ROUNDS
    if round_to_compact < 0:
        return
    msg_index = 2 * round_to_compact + 2  # position of that round's tool-result (user) message
    if msg_index >= len(messages):
        return
    msg = messages[msg_index]
    if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
        return
    for block in msg["content"]:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            original = block.get("content", "")
            if isinstance(original, str) and len(original) > 300:
                block["content"] = (
                    original[:300]
                    + f"... [truncated — {len(original)} chars total; an older round's raw tool "
                      f"result, compacted to bound this cycle's context size. Your own reasoning "
                      f"conclusion from that round, just above, is unaffected.]"
                )


class AutonomousTradingAgent:
    def __init__(self, config=CONFIG):
        self.config = config
        self._client = get_client(config)
        self._dispatcher = ToolDispatcher(config)

    async def run_cycle(self) -> int:
        """
        Runs one full decision cycle. Returns the number of minutes
        until the next cycle should run, as chosen by Claude (or a
        default if parsing fails or the key is missing).
        """
        import uuid
        cycle_id = uuid.uuid4().hex[:8]
        set_cycle_id(cycle_id)

        provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
        key_configured = (
            bool(os.getenv("FEATHERLESS_API_KEY")) if provider == "featherless"
            else bool(self.config.claude.api_key)
        )
        if not key_configured:
            log_event("autonomous_cycle_skipped", {"reason": f"{provider} API key not configured"})
            return DEFAULT_NEXT_CHECK_MINUTES

        log_event("autonomous_cycle_start", {"cycle_id": cycle_id})

        operator_note = _consume_operator_note()
        performance_reflection = _read_performance_reflection()
        position_thesis_brief = await _build_position_thesis_brief(self.config)
        opening_text = (
            "Begin this decision cycle. Check whatever account, position, and market information "
            "you need, decide whether to act, and act if warranted within your limits. End with "
            "your summary and the NEXT_CHECK_MINUTES line."
        )
        if position_thesis_brief:
            log_event("position_thesis_brief_injected", {"brief": position_thesis_brief})
            opening_text = f"{position_thesis_brief}\n\n{opening_text}"
        if performance_reflection:
            log_event("performance_reflection_injected", {"reflection": performance_reflection})
            opening_text = (
                f"YOUR OWN RECENT PERFORMANCE (a self-generated reflection on real, closed trades "
                f"and their actual outcomes — context to weigh as you judge fit, not an instruction): "
                f"{performance_reflection}\n\n{opening_text}"
            )
        if operator_note:
            log_event("operator_note_injected", {"note": operator_note})
            opening_text = (
                f"OPERATOR NOTE (read this first, it may correct something you previously assumed): "
                f"{operator_note}\n\n{opening_text}"
            )

        messages = [{"role": "user", "content": opening_text}]

        final_text = ""
        for round_num in range(MAX_TOOL_ROUNDS_PER_CYCLE):
            response = self._client.messages.create(
                model=self.config.claude.model,
                max_tokens=4096,
                system=AUTONOMOUS_AGENT_SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )

            text_blocks = [b.text for b in response.content if b.type == "text"]
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            if text_blocks:
                final_text = "\n".join(text_blocks)
                log_event("agent_reasoning", {"round": round_num, "text": final_text})

            if not tool_use_blocks:
                # Claude is done for this cycle — no more tools requested.
                break

            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in tool_use_blocks:
                log_event("agent_tool_call", {"round": round_num, "tool": block.name, "input": block.input})
                result_text = await self._dispatcher.dispatch(block.name, block.input)
                log_event("agent_tool_result", {"round": round_num, "tool": block.name, "result": result_text[:500]})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })
            messages.append({"role": "user", "content": tool_results})
            _compact_old_round_if_needed(messages, round_num)

        else:
            log_event("autonomous_cycle_max_rounds_hit", {"max_rounds": MAX_TOOL_ROUNDS_PER_CYCLE})

        next_check = self._parse_next_check_minutes(final_text)
        next_check = await self._respect_market_hours(next_check)
        log_event("autonomous_cycle_end", {"summary": final_text[-2000:], "next_check_minutes": next_check})
        return next_check

    @staticmethod
    def _parse_next_check_minutes(text: str) -> int:
        match = re.search(r"NEXT_CHECK_MINUTES:\s*(\d+)", text)
        if match:
            minutes = int(match.group(1))
            return max(1, min(minutes, 240))  # sane outer bounds: 1 min to 4 hours
        return DEFAULT_NEXT_CHECK_MINUTES

    async def _respect_market_hours(self, llm_next_check: int) -> int:
        """
        The LLM's own NEXT_CHECK_MINUTES guess is capped at 4 hours
        (see _parse_next_check_minutes), which is fine for an active
        session but means a closed market (evenings, weekends,
        holidays) gets polled every 4 hours for nothing. If the market
        is closed, override with the actual time until next_open
        (from Alpaca's clock, which correctly accounts for holidays)
        instead of trusting the LLM to reason about calendars.
        Falls back to the LLM's own figure if the clock call fails,
        rather than blocking the cycle on it.
        """
        try:
            clock = await AlpacaExecutionClient(self.config).market_clock()
        except Exception as exc:
            log_event("market_clock_check_failed", {"error": str(exc)})
            return llm_next_check

        if clock["is_open"] or not clock.get("next_open"):
            return llm_next_check

        try:
            next_open = datetime.fromisoformat(clock["next_open"])
            now = datetime.now(timezone.utc)
            minutes_until_open = int((next_open - now).total_seconds() / 60)
        except (ValueError, TypeError) as exc:
            log_event("market_clock_parse_failed", {"error": str(exc), "next_open": clock.get("next_open")})
            return llm_next_check

        # Wake 5 minutes before the bell rather than exactly at it, and
        # never schedule something nonsensical (negative, or absurdly
        # long — outer bound of 3 days covers even a long weekend).
        minutes_until_open = max(1, minutes_until_open - 5)
        minutes_until_open = min(minutes_until_open, 3 * 24 * 60)

        log_event("market_closed_next_check_overridden", {
            "llm_suggested_minutes": llm_next_check,
            "market_next_open": clock["next_open"],
            "minutes_until_open": minutes_until_open,
        })
        return minutes_until_open
