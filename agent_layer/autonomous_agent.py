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
from execution.alpaca_client import AlpacaExecutionClient
from execution.trade_logger import log_event

DEFAULT_NEXT_CHECK_MINUTES = 15
MAX_TOOL_ROUNDS_PER_CYCLE = 25  # safety valve against a runaway tool-call loop within one cycle
OPERATOR_NOTE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OPERATOR_NOTE")
PERFORMANCE_REFLECTION_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PERFORMANCE_REFLECTION")


def _consume_operator_note() -> str:
    """
    If a note has been left (see deploy/DEPLOY.md), read it, delete the
    file so it's only injected once, and return its text. Lets the
    operator correct a factual error or flag something without needing
    to stop and restart the whole process — the note becomes part of
    the very next cycle's opening message.
    """
    if not os.path.exists(OPERATOR_NOTE_PATH):
        return ""
    with open(OPERATOR_NOTE_PATH, "r") as f:
        note = f.read().strip()
    os.remove(OPERATOR_NOTE_PATH)
    return note


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
        provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
        key_configured = (
            bool(os.getenv("FEATHERLESS_API_KEY")) if provider == "featherless"
            else bool(self.config.claude.api_key)
        )
        if not key_configured:
            log_event("autonomous_cycle_skipped", {"reason": f"{provider} API key not configured"})
            return DEFAULT_NEXT_CHECK_MINUTES

        log_event("autonomous_cycle_start", {})

        operator_note = _consume_operator_note()
        performance_reflection = _read_performance_reflection()
        opening_text = (
            "Begin this decision cycle. Check whatever account, position, and market information "
            "you need, decide whether to act, and act if warranted within your limits. End with "
            "your summary and the NEXT_CHECK_MINUTES line."
        )
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
