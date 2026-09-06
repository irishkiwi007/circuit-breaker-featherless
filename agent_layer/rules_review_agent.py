"""
Periodic (not per-tick) review process: Claude looks at recent logged
activity — how often the fast layer found nothing, why, and what the
market regime looked like — and decides whether a whitelisted,
bounded entry-sensitivity parameter should change.

This is deliberately a *separate* agent from TradeReviewAgent
(agent_layer/claude_agent.py), which reviews individual trade
candidates. Mixing "should this one trade happen" judgment with
"should the rules themselves change" judgment in one call would make
both harder to audit. Keeping them separate means each decision type
has its own clean log trail — see docs/SUBMISSION.md for why this
separation is the point, not incidental.
"""
import json
from collections import Counter
from dataclasses import asdict, fields
from typing import Optional

from config import CONFIG
from config.dynamic_overrides import ADJUSTABLE_BOUNDS, apply_override, load_overrides
from execution.trade_logger import read_events, log_event
from agent_layer.llm_client import get_client
from agent_layer.rules_review_prompts import RULES_REVIEW_SYSTEM_PROMPT, build_review_prompt
from agent_layer.claude_agent import _strip_code_fences


def _summarize_recent_activity(limit: int = 200) -> dict:
    events = read_events(limit=limit)
    counts = Counter(e["event_type"] for e in events)

    no_candidate_reasons = []
    for e in events:
        if e["event_type"] == "no_candidate":
            vix = e.get("payload", {}).get("vix", {})
            no_candidate_reasons.append(vix)

    agent_decisions = Counter(
        e["payload"].get("decision") for e in events if e["event_type"] == "agent_decision"
    )

    return {
        "total_events": len(events),
        "no_candidate_count": counts.get("no_candidate", 0),
        "candidate_generated_count": counts.get("agent_decision", 0),
        "agent_decisions_breakdown": dict(agent_decisions),
        "sample_vix_readings_when_no_candidate": no_candidate_reasons[-10:],
        "market_data_fetch_failures": counts.get("market_data_fetch_failed", 0),
    }


class RulesReviewAgent:
    def __init__(self, config=CONFIG):
        self.config = config
        self._client = get_client(config)

    def review(self) -> dict:
        """
        Runs one review pass. Returns a record of what was decided,
        including if no change was made. Always logs, regardless of
        outcome, so 'the agent looked and decided not to change
        anything' is as visible in the audit trail as an actual change.
        """
        if not self.config.claude.api_key:
            record = {
                "change_recommended": False,
                "field": None,
                "new_value": None,
                "reasoning": "ANTHROPIC_API_KEY not configured; skipping review, base config used unchanged.",
                "applied": False,
            }
            log_event("rules_review", record)
            return record

        current_values = {
            field: getattr(self.config.strategy, field) for field in ADJUSTABLE_BOUNDS
        }
        activity_summary = _summarize_recent_activity()

        prompt = build_review_prompt(current_values, activity_summary)
        response = self._client.messages.create(
            model=self.config.claude.model,
            max_tokens=1000,
            system=RULES_REVIEW_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if hasattr(block, "text"))

        try:
            parsed = json.loads(_strip_code_fences(text))
        except (json.JSONDecodeError, TypeError):
            record = {
                "change_recommended": False,
                "field": None,
                "new_value": None,
                "reasoning": f"Agent response could not be parsed; no change applied. Raw: {text[:200]}",
                "applied": False,
            }
            log_event("rules_review", record)
            return record

        record = {
            "change_recommended": parsed.get("change_recommended", False),
            "field": parsed.get("field"),
            "new_value": parsed.get("new_value"),
            "reasoning": parsed.get("reasoning", ""),
            "applied": False,
        }

        if record["change_recommended"] and record["field"] and record["new_value"] is not None:
            try:
                applied = apply_override(record["field"], float(record["new_value"]), record["reasoning"])
                record["applied"] = True
                record["applied_record"] = applied
            except ValueError as e:
                # Field wasn't in the whitelist — should be impossible given the
                # system prompt's constraints, but fail closed and log it rather
                # than silently ignoring a malformed agent response.
                record["applied"] = False
                record["rejection_reason"] = str(e)

        log_event("rules_review", record)
        return record
