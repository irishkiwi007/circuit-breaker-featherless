"""
A second, remotely-reachable channel for operator notes, alongside the
local OPERATOR_NOTE file (see autonomous_agent.py / deploy/DEPLOY.md).

The dashboard (streamlit_app.py, hosted separately on Streamlit Cloud)
has no filesystem access to this VM, so it can't write OPERATOR_NOTE
directly. Instead, a passcode-gated box on the dashboard creates a
GitHub issue labeled "operator-note" in this repo. This module polls
for open issues with that label, concatenates them into the same kind
of note text the local file produces, and closes each issue so it's
only consumed once — mirroring the local file's delete-after-read
behavior.

Trust note: an operator note (from either channel) is injected as
plain context into the *same* tool-calling conversation the live
trading agent uses to place real orders — it is not sandboxed the way
the dashboard's separate "Ask the Agent" Q&A is. The passcode gate on
the dashboard's submission form exists specifically because this
channel is not isolated: anything that reaches an open issue here
will be read by an agent that can actually trade.
"""
import os

import requests

GITHUB_API_BASE = "https://api.github.com"
NOTE_LABEL = "operator-note"


def _repo() -> str:
    return os.getenv("GITHUB_REPO", "irishkiwi007/circuit-breaker-featherless")


def _token() -> str:
    return os.getenv("GITHUB_TOKEN", "")


def fetch_and_consume_remote_notes() -> str:
    """
    Fetches open issues labeled operator-note, closes each one (so it
    isn't read again next cycle), and returns their combined text.
    Returns "" on any failure or if no token is configured — this must
    never block or crash a trading cycle over a broken feedback channel.
    """
    token = _token()
    if not token:
        return ""

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    try:
        resp = requests.get(
            f"{GITHUB_API_BASE}/repos/{_repo()}/issues",
            headers=headers,
            params={"labels": NOTE_LABEL, "state": "open", "sort": "created", "direction": "asc"},
            timeout=10,
        )
        resp.raise_for_status()
        issues = resp.json()
    except Exception:
        return ""

    if not issues:
        return ""

    notes = []
    for issue in issues:
        title = issue.get("title", "").strip()
        body = (issue.get("body") or "").strip()
        created = issue.get("created_at", "")
        notes.append(f"[{created}] {title}: {body}" if title else f"[{created}] {body}")

        # Close it immediately after reading so a failure later in this
        # function can't cause it to be silently dropped without ever
        # having been surfaced to the agent.
        number = issue.get("number")
        if number is not None:
            try:
                requests.patch(
                    f"{GITHUB_API_BASE}/repos/{_repo()}/issues/{number}",
                    headers=headers,
                    json={"state": "closed"},
                    timeout=10,
                )
            except Exception:
                pass  # Already surfaced to the agent; a failed close just risks a repeat next cycle.

    return "\n\n".join(notes)
