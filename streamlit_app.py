"""
Streamlit demo dashboard for the hackathon's "Demo Application Platform"
requirement. Queries Alpaca's REST API directly — doesn't depend on the
VM running main_autonomous.py being up, so it works as a standalone,
always-available demo of the live paper account's real state.

Reads credentials from Streamlit secrets (st.secrets), never hardcoded.

Note: this app's dependencies live in the root requirements.txt
(lightweight — streamlit + requests + plotly) so Streamlit Cloud's
default auto-detection picks them up with zero configuration. The
trading agent's own, much heavier dependencies live in
requirements-agent.txt instead — install that one on the VM, not this one.

This dashboard has no live connection to the VM — it only sees what
Alpaca's API exposes directly (orders, positions, fills, bars). The
agent's own written rationale for each trade lives in logs/events.jsonl
on the VM, so it's bridged in indirectly: deploy/sync_reasoning.sh runs
on the VM (via cron), derives a small logs/reasoning_export.json from
the local event log, and pushes it to this repo. This app then fetches
that file over raw.githubusercontent.com and matches it to each trade
by underlying + leg symbols (see fetch_reasoning_export /
match_reasoning below). If the VM's sync cron isn't running, or a
trade is very recent, the detail page degrades gracefully to "no
reasoning synced yet" rather than erroring.
"""
import streamlit as st
import re
import requests
from openai import OpenAI
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from collections import defaultdict
import plotly.graph_objects as go

# Trade-record reconstruction (parse_underlying, group_key,
# root_symbol_from_group, parse_occ_symbol, build_trade_records) now
# lives in execution/trade_records.py, shared with the live agent's
# own get_setup_performance tool — see that module for the logic.
from execution.trade_records import (
    parse_underlying,
    group_key,
    root_symbol_from_group,
    parse_occ_symbol,
    build_trade_records,
)

st.set_page_config(page_title="Circuit Breaker (Featherless) — Live Demo", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0A0E17; }
    h1, h2, h3, p, span, div, label { color: #F9FAFB; }
    .stDataFrame { background-color: #1F2937; }

    /* Streamlit's default metric font is oversized for mobile — was
    causing excessive scrolling on phones. */
    [data-testid="stMetricValue"] { font-size: 1.3rem; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem; }
    [data-testid="stMetricDelta"] { font-size: 0.75rem; }

    h1 { font-size: 1.6rem !important; }
    h2 { font-size: 1.3rem !important; }
    h3 { font-size: 1.1rem !important; }

    .stDataFrame { font-size: 0.85rem; }

    @media (max-width: 600px) {
        [data-testid="stMetricValue"] { font-size: 1.05rem; }
        [data-testid="stMetricLabel"] { font-size: 0.7rem; }
        h1 { font-size: 1.35rem !important; }
        h2 { font-size: 1.1rem !important; }
        h3 { font-size: 0.95rem !important; }
        .stDataFrame { font-size: 0.75rem; }
        p, span, div, label { font-size: 0.85rem; }
    }
</style>
""", unsafe_allow_html=True)

API_KEY = st.secrets.get("ALPACA_API_KEY", "")
SECRET_KEY = st.secrets.get("ALPACA_SECRET_KEY", "")
BASE_URL = st.secrets.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
DATA_URL = st.secrets.get("ALPACA_DATA_URL", "https://data.alpaca.markets")

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY,
}
FEATHERLESS_API_KEY = st.secrets.get("FEATHERLESS_API_KEY", "")
FEATHERLESS_BASE_URL = st.secrets.get("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1")
LLM_MODEL = st.secrets.get("LLM_MODEL", "deepseek-ai/DeepSeek-V4-Pro")

NYC_TZ = ZoneInfo("America/New_York")


def fetch(base: str, path: str, params: dict = None):
    try:
        resp = requests.get(f"{base}{path}", headers=HEADERS, params=params or {}, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


REASONING_EXPORT_URL = (
    "https://raw.githubusercontent.com/irishkiwi007/circuit-breaker-featherless"
    "/main/logs/reasoning_export.json"
)


@st.cache_data(ttl=300)
def fetch_reasoning_export():
    """
    Pulls the agent's own trade rationale, synced from the VM's local
    events.jsonl via deploy/sync_reasoning.sh. Cached 5 minutes.
    Returns [] on any failure — callers treat that as "no reasoning
    available yet", not an error to surface.
    """
    try:
        resp = requests.get(REASONING_EXPORT_URL, timeout=10)
        if resp.status_code != 200:
            return []
        return resp.json().get("records", [])
    except Exception:
        return []


def render_rationale_text(text: str) -> str:
    """
    Two fixes for the agent's rationale text, which is written by the
    LLM as dense run-on prose:
    1. Escapes literal "$" before markdown rendering — Streamlit's
       markdown treats a pair of "$" as inline LaTeX/KaTeX math
       delimiters by default, and this text is full of dollar amounts,
       so without this, arbitrary spans between dollar signs render as
       garbled math notation in a different font.
    2. Splits on sentence boundaries (period + whitespace) and puts
       each sentence on its own line, dropping the period. Safe
       against decimals like "$2.01" since a decimal point is never
       followed by whitespace.
    """
    if not text:
        return "—"
    text = text.replace("$", "\\$")
    sentences = re.split(r"\.\s+", text.strip())
    sentences = [s.rstrip(".").strip() for s in sentences if s.strip()]
    return "  \n".join(sentences)


def match_reasoning(trade: dict, records: list) -> dict:
    """
    Matches a built trade record to its open/close reasoning entries.
    Keyed on underlying + actual leg symbols (not timestamp proximity)
    since two trades on the same underlying can be open concurrently.
    """
    open_symbols = {e["symbol"] for e in trade.get("initial_open_events", [])}
    close_symbols = {e["symbol"] for e in trade.get("close_events", [])}

    open_reasoning, close_reasoning = None, None
    for r in records:
        if r.get("underlying") != trade["underlying"]:
            continue
        r_symbols = {r.get("buy_symbol"), r.get("sell_symbol")}
        if r.get("action") == "open" and open_symbols and r_symbols & open_symbols:
            open_reasoning = r
        elif r.get("action") == "close" and close_symbols and r_symbols & close_symbols:
            close_reasoning = r

    return {"open": open_reasoning, "close": close_reasoning}


def trade_expiry(trade: dict) -> str:
    """Every leg of a trade shares the same expiration date (that's what
    groups them into one trade in the first place) — pull it from
    whichever leg event is available, opening or closing."""
    all_events = trade["initial_open_events"] + trade["modification_events"] + trade["close_events"]
    if not all_events:
        return "—"
    return parse_occ_symbol(all_events[0]["symbol"])["expiry"]


def compute_leg_breakdown(trade: dict, live_positions: list) -> dict:
    """
    Per-leg purchase/current-or-exit/profit breakdown, per contract and
    per-leg totals, plus a grand total row across all legs of the trade.

    Purchase price per leg = qty-weighted average of that leg's opening
    fill prices (handles a leg built across multiple opening orders).
    Current/exit price per leg = live current_price (open trades, from
    the live position) or qty-weighted average of closing fill prices
    (closed trades, including any synthesized $0 expiration events).
    Profit per contract respects long vs short: a long leg profits when
    price rises, a short leg profits when price falls (bought back
    cheaper than the credit received).

    Per-leg totals are qty × price (no ×100 multiplier) to match a
    plain "cost of N contracts at $X each" reading. The grand total row
    additionally nets long cost against short credit (the same signed
    convention already used for entry/current spread value elsewhere in
    the app), since summing both legs' raw totals unsigned would double
    count rather than net them.
    """
    all_events = trade["initial_open_events"] + trade["modification_events"] + trade["close_events"]
    symbols = sorted(set(e["symbol"] for e in all_events))

    legs = []
    for symbol in symbols:
        opens = [e for e in (trade["initial_open_events"] + trade["modification_events"]) if e["symbol"] == symbol]
        closes = [e for e in trade["close_events"] if e["symbol"] == symbol]
        if not opens:
            continue

        is_long = opens[0]["intent"] == "buy_to_open"
        qty = sum(e["qty"] for e in opens)
        purchase_price = sum(e["qty"] * e["price"] for e in opens) / qty if qty else 0.0

        current_or_exit_price = None
        if closes:
            close_qty = sum(e["qty"] for e in closes)
            current_or_exit_price = sum(e["qty"] * e["price"] for e in closes) / close_qty if close_qty else None
        elif trade["status"] == "open":
            live = next((p for p in live_positions if p.get("symbol") == symbol), None)
            if live is not None:
                current_or_exit_price = float(live.get("current_price", 0) or 0)

        if current_or_exit_price is None:
            profit_per_contract = None
        else:
            profit_per_contract = (current_or_exit_price - purchase_price) if is_long else (purchase_price - current_or_exit_price)

        occ = parse_occ_symbol(symbol)
        legs.append({
            "symbol": symbol,
            "type": occ["type"],
            "strike": occ["strike"],
            "expiry": occ["expiry"],
            "side": "Long" if is_long else "Short",
            "qty": qty,
            "purchase_price": purchase_price,
            "current_or_exit_price": current_or_exit_price,
            "profit_per_contract": profit_per_contract,
            "leg_total_purchase": qty * purchase_price,
            "leg_total_current": (qty * current_or_exit_price) if current_or_exit_price is not None else None,
            "leg_total_profit": (qty * profit_per_contract) if profit_per_contract is not None else None,
        })

    # Grand total: net long cost against short credit, matching the
    # signed convention already used for spread entry/current value
    # elsewhere in the app — NOT a naive unsigned sum of both legs.
    grand_purchase = sum(
        (l["leg_total_purchase"] if l["side"] == "Long" else -l["leg_total_purchase"]) for l in legs
    )
    have_all_current = all(l["leg_total_current"] is not None for l in legs) and legs
    grand_current = sum(
        (l["leg_total_current"] if l["side"] == "Long" else -l["leg_total_current"]) for l in legs
    ) if have_all_current else None
    grand_profit = sum(l["leg_total_profit"] for l in legs if l["leg_total_profit"] is not None) if legs else 0.0

    return {
        "legs": legs,
        "grand_purchase": grand_purchase,
        "grand_current": grand_current,
        "grand_profit": grand_profit,
    }


def format_nyc(iso_ts: str) -> str:
    """Converts any ISO timestamp (assumed UTC if no offset given) to
    'dd mm yyyy HH:MM' in America/New_York time."""
    if not iso_ts:
        return "—"
    try:
        ts = iso_ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        nyc = dt.astimezone(NYC_TZ)
        return nyc.strftime("%d %m %Y %H:%M")
    except Exception:
        return iso_ts




if not API_KEY or not SECRET_KEY:
    st.error(
        "Alpaca API credentials not configured. Add ALPACA_API_KEY and ALPACA_SECRET_KEY "
        "in this app's Streamlit Cloud secrets to connect to the live account."
    )
    st.stop()

# ---------------------------------------------------------------
# Session state / simple router: dashboard vs. trade detail
# ---------------------------------------------------------------
if "view" not in st.session_state:
    st.session_state.view = "dashboard"
if "selected_trade_idx" not in st.session_state:
    st.session_state.selected_trade_idx = None


def go_to_detail(idx: int):
    st.session_state.view = "detail"
    st.session_state.selected_trade_idx = idx
    st.rerun()


def go_to_dashboard():
    st.session_state.view = "dashboard"
    st.session_state.selected_trade_idx = None
    st.rerun()


def render_leg_table(events, empty_msg):
    if not events:
        st.caption(empty_msg)
        return
    rows = []
    for e in events:
        occ = parse_occ_symbol(e["symbol"])
        rows.append({
            "Time (NYC)": format_nyc(e["ts"]),
            "Type": occ["type"],
            "Strike": occ["strike"],
            "Expiry": occ["expiry"],
            "Side": e.get("side") or "—",
            "Intent": e["intent"],
            "Qty": e["qty"],
            "Fill Price": f"${e['price']:.2f}",
            "Source": "Expired worthless" if e.get("source") == "expiration" else "Order fill",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


# =================================================================
# Fetch everything up front — both views need it
# =================================================================
account = fetch(BASE_URL, "/v2/account")
if "error" in account:
    st.warning(f"Could not reach Alpaca right now: {account['error']}")
    st.stop()

positions = fetch(BASE_URL, "/v2/positions")
if not isinstance(positions, list):
    positions = []

all_orders = fetch(BASE_URL, "/v2/orders", {"status": "all", "limit": 200, "direction": "desc"})
if not isinstance(all_orders, list):
    all_orders = []

# OPEXP = Alpaca's non-trade activity for an option expiring — never
# shows up as an order at all, so without this, an expired position
# would silently stay stuck as "open" forever with no outcome.
expiry_activities = fetch(BASE_URL, "/v2/account/activities/OPEXP")
if not isinstance(expiry_activities, list):
    expiry_activities = []

trades = build_trade_records(all_orders, positions, expiry_activities)


def ask_agent_isolated(question: str, account: dict, positions: list, reasoning_records: list) -> str:
    """
    Answers a question about the agent's real recent activity, using
    only data this dashboard already has. No tools are passed to this
    API call -- there is nothing here that can place, close, or modify
    a trade, by construction, not by instruction alone.
    """
    recent_reasoning = reasoning_records[-15:] if reasoning_records else []
    reasoning_text = "\n".join(
        f"[{r.get('timestamp')}] {r.get('action')} {r.get('underlying')}: {r.get('rationale', '')[:300]}"
        for r in recent_reasoning
    ) or "No recent reasoning synced yet."

    context = (
        f"Current account equity: ${account.get('equity', 'unknown')}\n"
        f"Current open positions: {len(positions)} position(s)\n\n"
        f"Recent trade reasoning (most recent {len(recent_reasoning)} entries):\n{reasoning_text}"
    )

    system_prompt = (
        "You are answering a question from someone viewing your public trading dashboard, "
        "about your own real recent trading activity. This conversation has no tools to "
        "place, close, or modify any order or position -- you are architecturally incapable "
        "of trading right now, so answer honestly and reflectively, not as if you're deciding "
        "anything. Nothing you say here will be shown to your trading-cycle self or affect "
        "what you do next cycle. Base your answer only on the real context provided; if you "
        "don't have enough information to answer confidently, say so rather than guessing. "
        "Keep it to a few sentences -- this is a dashboard, not a report.\n\n"
        "For context only (you still can't call these here): your live trading-cycle self "
        "has get_setup_performance (win rate/P&L by setup type), get_portfolio_greeks (net "
        "delta/theta/vega across open positions), get_order_fill_status (real fill price "
        "vs. assumed), and get_market_context (VIX/SPX regime). If asked whether you have "
        "access to things like historical win-rate breakdowns, portfolio Greeks, real fill "
        "prices, or VIX context, say that your trading-cycle self has these tools now, not "
        "that they don't exist -- but note you can't run them from this Q&A box."
    )

    client = OpenAI(api_key=FEATHERLESS_API_KEY, base_url=FEATHERLESS_BASE_URL)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=500,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context on your recent activity:\n\n{context}\n\nQuestion: {question}"},
        ],
    )
    return (response.choices[0].message.content or "").strip()


# =================================================================
# DETAIL VIEW
# =================================================================
if st.session_state.view == "detail" and st.session_state.selected_trade_idx is not None:
    idx = st.session_state.selected_trade_idx
    if idx >= len(trades):
        st.error("That trade is no longer available.")
        if st.button("← Back to dashboard"):
            go_to_dashboard()
        st.stop()

    trade = trades[idx]

    if st.button("← Back to dashboard"):
        go_to_dashboard()

    st.title(f"Trade Detail — {trade['underlying']}")
    st.caption(f"Group: {trade['group_key']} · Status: {trade['status']}")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Class", trade["class"])
    with c2:
        st.metric("Status", trade["status"])
    with c3:
        st.metric("Qty", trade["qty"])
    with c4:
        st.metric("Outcome", f"${trade['outcome']:,.2f}")
    with c5:
        result_label = trade["profit_loss"].upper() if trade["profit_loss"] else ("OPEN" if trade["status"] == "open" else "—")
        st.metric("Result", result_label)

    if trade["current_value_per_contract"] is not None:
        value_line = f"Current value: ${trade['current_value_per_contract']:.2f}/contract"
        if trade.get("purchase_price_per_contract") is not None:
            value_line = f"Entry: ${trade['purchase_price_per_contract']:.2f}/contract → " + value_line
        st.caption(value_line)

    st.caption(f"Opened: {format_nyc(trade['time_opened'])} NYC" + (f" · Closed: {format_nyc(trade['time_closed'])} NYC" if trade["time_closed"] else " · Still open"))

    reasoning_records = fetch_reasoning_export()
    matched = match_reasoning(trade, reasoning_records)

    st.subheader("Agent's Reasoning")
    if not matched["open"] and not matched["close"]:
        st.info(
            "ℹ️ No reasoning synced for this trade yet. The agent's rationale is written on "
            "the VM and synced to this dashboard periodically — if this trade was opened "
            "very recently, it may not have synced yet."
        )
    else:
        if matched["open"]:
            r = matched["open"]
            with st.container(border=True):
                st.markdown(f"**Why it opened this trade** · {format_nyc(r['timestamp'])} NYC")
                st.markdown(render_rationale_text(r.get("rationale")))
                st.caption(
                    f"{r.get('contracts')} contract(s) · limit ${r.get('limit_price'):.2f} · "
                    f"max loss/contract ${r.get('max_loss_per_contract'):.2f}"
                )
        if matched["close"]:
            r = matched["close"]
            with st.container(border=True):
                st.markdown(f"**Why it closed this trade** · {format_nyc(r['timestamp'])} NYC")
                st.markdown(render_rationale_text(r.get("rationale")))
        if matched["open"] and not matched["close"] and trade["status"] == "closed":
            st.caption(
                "No closing rationale synced — this trade may have been closed by expiration "
                "or the drawdown backstop rather than an agent decision, neither of which log "
                "a rationale field."
            )

    # ---- Price chart around entry ----
    st.subheader(f"Market Context — {trade['underlying']}")
    try:
        entry_dt = datetime.fromisoformat(trade["time_opened"].replace("Z", "+00:00"))
    except Exception:
        entry_dt = datetime.now(timezone.utc)

    exit_dt = None
    if trade["time_closed"]:
        try:
            exit_dt = datetime.fromisoformat(trade["time_closed"].replace("Z", "+00:00"))
        except Exception:
            exit_dt = None

    span_end = exit_dt or datetime.now(timezone.utc)
    span = span_end - entry_dt

    # Padding scales with how long the trade was actually open, so a quick
    # same-day trade gets a tight, readable window and a multi-day trade
    # doesn't get compressed into an unreadable sliver.
    padding = max(timedelta(minutes=30), span * 0.15)
    padding = min(padding, timedelta(hours=6))

    if span <= timedelta(hours=6):
        bar_timeframe = "5Min"
    elif span <= timedelta(days=2):
        bar_timeframe = "15Min"
    else:
        bar_timeframe = "1Hour"

    start = (entry_dt - padding).isoformat()
    # Never request an end-time in the future — the bars API has no
    # data there and returns nothing for the *entire* request rather
    # than just the missing tail, which is why open trades previously
    # showed no chart at all.
    requested_end = min(span_end + padding, datetime.now(timezone.utc))
    end = requested_end.isoformat()
    bars_resp = fetch(DATA_URL, f"/v2/stocks/{trade['underlying']}/bars", {
        "timeframe": bar_timeframe, "start": start, "end": end, "limit": 300,
        "feed": "iex",  # SIP (the default) isn't authorized for recent/real-time
                        # data on paper/free accounts and returns nothing for the
                        # whole request rather than just the recent tail; IEX covers
                        # the current session without needing a real-time subscription.
    })
    bars = bars_resp.get("bars", []) if isinstance(bars_resp, dict) else []

    if bars:
        def to_nyc_naive(dt_or_iso):
            if isinstance(dt_or_iso, str):
                dt = datetime.fromisoformat(dt_or_iso.replace("Z", "+00:00"))
            else:
                dt = dt_or_iso
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(NYC_TZ).replace(tzinfo=None).isoformat()

        fig = go.Figure(data=[go.Candlestick(
            x=[to_nyc_naive(b["t"]) for b in bars],
            open=[b["o"] for b in bars],
            high=[b["h"] for b in bars],
            low=[b["l"] for b in bars],
            close=[b["c"] for b in bars],
            increasing_line_color="#22D3A8", decreasing_line_color="#EF4444",
        )])
        fig.add_vline(x=to_nyc_naive(entry_dt), line_dash="dash", line_color="#F59E0B",
                       annotation_text="Entry", annotation_font_color="#F59E0B")
        if exit_dt:
            fig.add_vline(x=to_nyc_naive(exit_dt), line_dash="dash", line_color="#0EA5E9",
                           annotation_text="Exit", annotation_font_color="#0EA5E9")
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#151B2B",  # deliberately lighter than the page background (#0A0E17)
            plot_bgcolor="#151B2B",   # so the chart's own boundary is visually obvious, not blending in
            height=450, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_rangeslider_visible=False,
            dragmode=False,  # a scroll gesture starting on the chart was being captured as a
                              # zoom-select drag instead of scrolling the page — this was the
                              # actual cause of "touching the chart resizes it" while scrolling.
        )
        # Force the modebar (autoscale/reset button included) to always be visible rather than
        # only appearing on hover, which doesn't work on a touch screen at all.
        chart_config = {
            "scrollZoom": False,      # pinch/wheel no longer zooms the chart unexpectedly
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        }
        with st.container(border=True):  # gives the chart a real, visible boundary on the page
            st.plotly_chart(fig, use_container_width=True, config=chart_config)
    else:
        st.info(f"No bar data available for {trade['underlying']} in this window.")

    st.divider()

    # ---- Trade Summary: entry vs current/exit, per-contract and total ----
    st.subheader("Trade Summary")
    summary_breakdown = compute_leg_breakdown(trade, positions)
    qty = trade["qty"] or 1
    entry_per_ctr = summary_breakdown["grand_purchase"] / qty
    exit_or_current_per_ctr = (summary_breakdown["grand_current"] / qty) if summary_breakdown["grand_current"] is not None else None
    price_label = "Exit" if trade["status"] == "closed" else "Current/Last"

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("Entry $/Ctr", f"${entry_per_ctr:.2f}")
    with sc2:
        st.metric(f"{price_label} $/Ctr", f"${exit_or_current_per_ctr:.2f}" if exit_or_current_per_ctr is not None else "—")
    with sc3:
        st.metric("Contracts", trade["qty"])

    sc4, sc5 = st.columns(2)
    with sc4:
        st.metric("Entry Total", f"${summary_breakdown['grand_purchase'] * 100:,.2f}")
    with sc5:
        st.metric(f"{price_label} Total", f"${summary_breakdown['grand_current'] * 100:,.2f}" if summary_breakdown["grand_current"] is not None else "—")

    st.divider()

    # ---- Per-leg purchase / current-or-exit / profit breakdown ----
    st.subheader("Per-Leg Breakdown")
    breakdown = compute_leg_breakdown(trade, positions)
    if breakdown["legs"]:
        leg_rows = [{
            "Type": l["type"],
            "Strike": l["strike"],
            "Expiry": l["expiry"],
            "Side": l["side"],
            "Contracts": l["qty"],
            "Purchase $": f"${l['purchase_price']:.2f}",
            ("Exit $" if trade["status"] == "closed" else "Current $"): (f"${l['current_or_exit_price']:.2f}" if l["current_or_exit_price"] is not None else "—"),
            "P/L per Ctr": f"${l['profit_per_contract']:.2f}" if l["profit_per_contract"] is not None else "—",
            "Leg Total P/L": f"${l['leg_total_profit'] * 100:,.2f}" if l["leg_total_profit"] is not None else "—",
        } for l in breakdown["legs"]]
        st.dataframe(leg_rows, use_container_width=True, hide_index=True)

        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            st.metric("Total Purchase", f"${breakdown['grand_purchase'] * 100:,.2f}")
        with gc2:
            st.metric("Total Current/Exit", f"${breakdown['grand_current'] * 100:,.2f}" if breakdown["grand_current"] is not None else "—")
        with gc3:
            st.metric("Total P/L (all legs)", f"${breakdown['grand_profit'] * 100:,.2f}")
    else:
        st.caption("No leg data available for this trade.")

    st.divider()

    st.subheader("Opening trade details")
    render_leg_table(trade["initial_open_events"], "No opening legs recorded.")

    if trade["modification_events"]:
        st.subheader("Modified trade details")
        st.caption("Additional opening orders placed on this same position after the initial entry.")
        render_leg_table(trade["modification_events"], "")

    if trade["close_events"]:
        st.subheader("Closing trade details")
        render_leg_table(trade["close_events"], "No closing legs recorded.")
    else:
        st.subheader("Closing trade details")
        st.caption("Still open")

    st.stop()

# =================================================================
# DASHBOARD VIEW
# =================================================================
st.title("⚡ Circuit Breaker")
st.caption("An autonomous AI trading agent, built on Alpaca — live paper account status")

funds_committed = abs(sum(float(p.get("cost_basis", 0) or 0) for p in positions))
current_equity = float(account.get("equity", 0))
cash_available = current_equity - funds_committed
open_unrealized = sum(float(p.get("unrealized_pl", 0) or 0) for p in positions)

# Mobile-friendly: 4 + 3 instead of 7 across one row, which was forcing
# heavy horizontal squeeze/scroll on phone screens.
row1 = st.columns(4)
with row1[0]:
    st.metric("Equity", f"${current_equity:,.2f}", help="Total account value, including the current market value of open positions.")
with row1[1]:
    st.metric("Cash Available", f"${cash_available:,.2f}", help="Equity minus capital currently committed to open positions.")
with row1[2]:
    st.metric("Buying Power", f"${float(account.get('buying_power', 0)):,.2f}")
with row1[3]:
    st.metric("Funds Committed", f"${funds_committed:,.2f}", help="How much was actually paid into currently open positions (net cost basis).")

row2 = st.columns(3)
with row2[0]:
    st.metric("Unrealized P&L", f"${open_unrealized:,.2f}", help="Live floating gain/loss on currently open positions — changes constantly while a position is open.")
with row2[1]:
    st.metric("Options Level", account.get("options_trading_level", "—"))
with row2[2]:
    st.metric("Status", account.get("status", "—"))

st.divider()

# ---- Trades summary (open/closed at the TRADE level, not raw order status) ----
st.subheader("Trades Summary")
open_trade_count = len([t for t in trades if t["status"] == "open"])
closed_trade_count = len([t for t in trades if t["status"] == "closed"])
cancelled_order_count = sum(1 for o in all_orders if o.get("status") in ("canceled", "expired"))
rejected_order_count = sum(1 for o in all_orders if o.get("status") == "rejected")

tcols = st.columns(4)
with tcols[0]:
    st.metric("Open Trades", open_trade_count)
with tcols[1]:
    st.metric("Closed Trades", closed_trade_count)
with tcols[2]:
    st.metric("Cancelled Orders", cancelled_order_count)
with tcols[3]:
    st.metric("Rejected Orders", rejected_order_count)

st.divider()

# ---- Win / Loss ----
# Realized P&L uses actual account equity, not reconstructed fill
# prices, so it's fee-inclusive to the cent. This is correctly stable
# by construction: subtracting the currently-open positions' own live
# unrealized P&L exactly cancels out their fluctuation, leaving only
# the fixed, already-settled result of closed trades. Any tiny jitter
# seen is just sub-second timing between two separate API calls
# (positions vs account), not a real drift in the number itself.
st.subheader("Win / Loss")
closed_trades = [t for t in trades if t["status"] == "closed"]
wins = [t for t in closed_trades if t["profit_loss"] == "win"]
losses = [t for t in closed_trades if t["profit_loss"] == "loss"]

STARTING_EQUITY = 10000.0  # this account's actual starting balance — NOT the hackathon's $100k
total_realized = (current_equity - STARTING_EQUITY) - open_unrealized

wl1, wl2, wl3, wl4 = st.columns(4)
with wl1:
    st.metric("Realized P&L", f"${total_realized:,.2f}", help="From actual account equity — includes exchange/regulatory fees. Mathematically excludes any currently-open position's fluctuation.")
with wl2:
    st.metric("Wins", len(wins))
with wl3:
    st.metric("Losses", len(losses))
with wl4:
    win_rate = (len(wins) / len(closed_trades) * 100) if closed_trades else 0
    st.metric("Win Rate", f"{win_rate:.0f}%" if closed_trades else "—")

# Alpaca charges no commission on options, but does pass through small
# real regulatory/exchange fees (SEC, FINRA TAF, CAT, ORF, OCC clearing)
# on trading activity. Most of these are billed as a single DAILY
# aggregate across multiple trades (e.g. "TAF fee for 4 trades on
# 2026-08-31"), not itemized per trade — so this is shown as one
# honest total, not attributed to individual trades, since Alpaca
# itself doesn't calculate it that way.
fee_activities = fetch(BASE_URL, "/v2/account/activities/FEE")
if not isinstance(fee_activities, list):
    fee_activities = []
total_fees = sum(abs(float(f.get("net_amount", 0) or 0)) for f in fee_activities)

st.metric("Total Fees Paid", f"${total_fees:,.2f}", help="Real exchange/regulatory fees (SEC, FINRA TAF, CAT, ORF, OCC clearing) — already included in Realized P&L above. Shown separately because most fee types are billed as a single daily total across multiple trades, not itemized per trade, so they can't be honestly split across individual Trade History rows.")

if not closed_trades:
    st.info("No closed trades yet — win/loss populates once a position is opened and closed.")

st.divider()

# ---- Open positions ----
st.subheader("Open Positions")
open_trades_indexed = [(i, t) for i, t in enumerate(trades) if t["status"] == "open"]

if open_trades_indexed:
    rows = []
    for _, t in open_trades_indexed:
        # Same compute_leg_breakdown() used for Trade History, not the
        # separate purchase_price_per_contract field — keeps both tables
        # using one consistent calculation method, not two that happen
        # to agree most of the time.
        breakdown = compute_leg_breakdown(t, positions)
        qty = t["qty"] or 1
        entry_per_ctr = breakdown["grand_purchase"] / qty
        current_per_ctr = (breakdown["grand_current"] / qty) if breakdown["grand_current"] is not None else None
        rows.append({
            "Underlying": t["underlying"],
            "Time Opened (NYC)": format_nyc(t["time_opened"]),
            "Class": t["class"],
            "Expiry": trade_expiry(t),
            "Contracts": t["qty"],
            "Entry $/Ctr": f"${entry_per_ctr:.2f}",
            "Current $/Ctr": f"${current_per_ctr:.2f}" if current_per_ctr is not None else "—",
            "Entry Total": f"${breakdown['grand_purchase'] * 100:,.2f}",
            "Current Total": f"${breakdown['grand_current'] * 100:,.2f}" if breakdown["grand_current"] is not None else "—",
            "Status": t["status"],
            "Outcome": f"${t['outcome']:,.2f}",
        })

    event = st.dataframe(
        rows, use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row", key="open_positions_table",
        column_config={
            "Entry $/Ctr": st.column_config.TextColumn(width="small"),
            "Current $/Ctr": st.column_config.TextColumn(width="small"),
        },
    )
    selected_rows = event.selection.rows if hasattr(event, "selection") else []
    if selected_rows:
        position_in_filtered_list = selected_rows[0]
        original_idx, selected_trade = open_trades_indexed[position_in_filtered_list]
        st.write(f"Selected: **{selected_trade['underlying']}** — opened {rows[position_in_filtered_list]['Time Opened (NYC)']}")
        if st.button("🔍 View trade detail & chart", type="primary", key="open_position_detail_btn"):
            go_to_detail(original_idx)
else:
    st.info("No open positions right now.")

st.divider()

# ---- Trade History — CLOSED trades only, one row per TRADE ----
# Open positions already have their own section above (Open Positions);
# including them here too was redundant and confusing — found via user
# report: a single completed trade appeared to show as "two lines"
# because a still-open, unrelated position was also being listed here
# with status='open' the moment it was placed, before it had any
# actual outcome yet.
st.subheader("Trade History")

closed_trades_indexed = [(i, t) for i, t in enumerate(trades) if t["status"] == "closed"]

if closed_trades_indexed:
    trade_rows = []
    for _, t in closed_trades_indexed:
        breakdown = compute_leg_breakdown(t, [])
        qty = t["qty"] or 1
        entry_per_ctr = breakdown["grand_purchase"] / qty
        exit_per_ctr = (breakdown["grand_current"] / qty) if breakdown["grand_current"] is not None else None
        trade_rows.append({
            "Underlying": t["underlying"],
            "Time Opened (NYC)": format_nyc(t["time_opened"]),
            "Class": t["class"],
            "Expiry": trade_expiry(t),
            "Contracts": t["qty"],
            "Entry $/Ctr": f"${entry_per_ctr:.2f}",
            "Exit $/Ctr": f"${exit_per_ctr:.2f}" if exit_per_ctr is not None else "—",
            "Entry Total": f"${breakdown['grand_purchase'] * 100:,.2f}",
            "Exit Total": f"${breakdown['grand_current'] * 100:,.2f}" if breakdown["grand_current"] is not None else "—",
            "Status": t["status"],
            "Outcome": f"${t['outcome']:,.2f}",
            "Time Closed (NYC)": format_nyc(t["time_closed"]) if t["time_closed"] else "—",
            "Profit/Loss": t["profit_loss"] if t["profit_loss"] else "—",
        })

    event = st.dataframe(
        trade_rows, use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row", key="trade_table",
    )

    selected_rows = event.selection.rows if hasattr(event, "selection") else []
    if selected_rows:
        position_in_filtered_list = selected_rows[0]
        original_idx, selected_trade = closed_trades_indexed[position_in_filtered_list]
        st.write(f"Selected: **{selected_trade['underlying']}** — {selected_trade['status']} — {trade_rows[position_in_filtered_list]['Time Opened (NYC)']}")
        if st.button("🔍 View trade detail & chart", type="primary"):
            go_to_detail(original_idx)
else:
    st.info("No completed trades yet. Check Open Positions above if something is currently active.")

st.divider()

st.subheader("Ask the Agent")
st.write("Feel free to ask my AI for information on its trading (note this does not influence its decision making)")

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if not FEATHERLESS_API_KEY:
    st.info("Q&A isn't configured on this dashboard yet.")
else:
    question = st.text_input("Your question", key="qa_question", label_visibility="collapsed",
                              placeholder="e.g. Why did you hold NVDA overnight instead of taking profit?")
    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Thinking..."):
            try:
                reasoning_records = fetch_reasoning_export()
                answer = ask_agent_isolated(question.strip(), account, positions, reasoning_records)
                st.session_state.qa_history.insert(0, {"q": question.strip(), "a": answer})
            except Exception as e:
                st.error(f"Couldn't get an answer right now: {e}")

    for pair in st.session_state.qa_history:
        with st.container(border=True):
            st.markdown(f"**Q: {pair['q']}**")
            st.write(pair["a"])

st.divider()

st.subheader("What makes this different")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**🎯 Genuinely autonomous**")
    st.write("An open-weight model (DeepSeek V4-Pro, via Featherless) originates every trade decision directly via Alpaca's MCP server — no rules engine pre-filtering candidates.")
with c2:
    st.markdown("**🛡️ Three hard backstops**")
    st.write("Defined-risk-only, spread-economics-sane, and a 15% per-trade sizing cap — enforced in code, not by prompt.")
with c3:
    st.markdown("**🔄 Self-assessing**")
    st.write("Reviews its own recent activity each cycle and adjusts its own approach — not a fixed script.")

now_nyc = datetime.now(timezone.utc).astimezone(NYC_TZ)
st.caption(f"Last refreshed: {now_nyc.strftime('%d %m %Y %H:%M')} NYC · [View source on GitHub](https://github.com/irishkiwi007/circuit-breaker-featherless)")

if st.button("🔄 Refresh"):
    st.rerun()
