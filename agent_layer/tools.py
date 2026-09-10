"""
Defines the tools Claude can call during an autonomous decision cycle,
and dispatches each tool call to the underlying Alpaca MCP client.

The critical design point: place_spread_order — the only tool that can
actually risk money — runs three hard backstops (risk/hard_backstops.py)
BEFORE calling Alpaca. If any fail, the tool returns a clear
rejection to Claude instead of executing, and nothing is sent to
Alpaca. Every other tool is read-only or account-management and
carries no backstop because it can't place risk.
"""
import json
from datetime import date

from execution.mcp_client import AlpacaMCPClient, unwrap_data
from execution.trade_logger import log_event
from risk.hard_backstops import check_defined_risk, check_position_sizing, check_spread_economics
from agent_layer import position_memory
from config import CONFIG

TOOL_SCHEMAS = [
    {
        "name": "get_account_info",
        "description": "Get current account equity, buying power, and options buying power.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_positions",
        "description": "Get all currently open positions.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_stock_quote",
        "description": "Get the latest quote for a stock/ETF symbol.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "get_stock_bars",
        "description": "Get recent intraday bars for a symbol, for trend/momentum context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "minutes": {"type": "integer", "description": "How many minutes of history to fetch"},
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_option_chain",
        "description": "Get the option chain (strikes, deltas, bid/ask) for an underlying, optionally filtered to a specific expiration date (YYYY-MM-DD). Use today's date for 0DTE.",
        "input_schema": {
            "type": "object",
            "properties": {
                "underlying": {"type": "string"},
                "expiration": {"type": "string", "description": "ISO date, e.g. 2026-08-29. Omit for all expirations."},
            },
            "required": ["underlying"],
        },
    },
    {
        "name": "place_spread_order",
        "description": (
            "Open OR close a two-leg options spread order — you must specify which via the "
            "required 'action' field. This is the ONLY way to place a multi-leg options order — "
            "naked single-leg orders are not available as a tool, by design. Both legs are required, "
            "on opposite sides (one buy, one sell), which structurally caps the position's maximum loss. "
            "IMPORTANT: to close an existing spread, use this same tool with action='close' and the "
            "correct buy/sell symbols for what you're doing in THIS order (e.g. if you originally "
            "bought the 716 call and sold the 720 call to open, closing means buy_symbol=the 720 call "
            "you're now buying back, sell_symbol=the 716 call you're now selling off). Do NOT call this "
            "with action='open' when your intent is to reduce or close a position — that will incorrectly "
            "open an ADDITIONAL position instead of closing the existing one. This exact mistake has "
            "happened before and doubled a position's size unintentionally. "
            "The order will be rejected with an explanation if it fails any hard backstop: "
            "not being a genuine defined-risk spread, the price paid/received not being economically "
            "sane relative to the spread's width, or risking more than 15% of account equity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["open", "close"], "description": "Whether this order opens a new position or closes/reduces an existing one. Always required, never inferred."},
                "underlying": {"type": "string"},
                "buy_symbol": {"type": "string", "description": "OCC option symbol for the leg to buy in THIS order"},
                "sell_symbol": {"type": "string", "description": "OCC option symbol for the leg to sell in THIS order"},
                "contracts": {"type": "integer"},
                "limit_price": {"type": "number", "description": "Net limit price for the spread; negative for net credit, positive for net debit"},
                "max_loss_per_contract": {"type": "number", "description": "Your calculated worst-case loss for ONE contract of this spread, in dollars (e.g. spread width minus credit, times 100). For a close order, use the remaining max loss on the position being closed."},
                "rationale": {"type": "string", "description": "Your reasoning for this specific order, including why action is open vs close"},
                "setup_type": {
                    "type": "string",
                    "description": (
                        "REQUIRED when action='open' (ignored/optional for action='close'). Your own "
                        "short label for what kind of setup this is, e.g. 'momentum_breakout', "
                        "'mean_reversion', 'earnings_iv_crush', 'macro_hedge'. This is not a fixed "
                        "enum — use a consistent label for genuinely similar setups so your own "
                        "historical performance by setup type (get_setup_performance) is meaningful, "
                        "but don't force a label that doesn't fit just to reuse one. This is how you "
                        "build a real track record instead of only your in-the-moment confidence."
                    ),
                },
            },
            "required": ["action", "underlying", "buy_symbol", "sell_symbol", "contracts", "limit_price", "max_loss_per_contract", "rationale"],
        },
    },
    {
        "name": "close_position",
        "description": "Close a specific open position by its symbol.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "close_all_positions",
        "description": "Close ALL open positions immediately. Use for emergency de-risking, not routine exits.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_most_active_stocks",
        "description": (
            "Screens the whole market for the most actively traded stocks right now, by volume or "
            "trade count. Use this to discover candidates beyond SPY/QQQ — this is the actual "
            "mechanism for looking at the broader liquid large-cap universe, not just a suggestion "
            "to do so. A stock showing up here is, by definition, currently liquid enough to trade well."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "by": {"type": "string", "enum": ["volume", "trades"], "description": "Ranking metric, default volume"},
                "top": {"type": "integer", "description": "How many to return, 1-100, default 10"},
            },
        },
    },
    {
        "name": "get_market_movers",
        "description": (
            "Returns today's top gainers and losers by percentage move, real-time. Use this alongside "
            "get_most_active_stocks to discover candidates with an actual catalyst or directional move "
            "happening right now, not just habitually checking the same one or two symbols every cycle."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "top": {"type": "integer", "description": "How many gainers and losers each to return, 1-50, default 10"},
            },
        },
    },
    {
        "name": "get_sp500_batch",
        "description": (
            "Returns a batch of real S&P 500 tickers to check, rotating through the full index over "
            "successive calls so you get systematic coverage over time — not just whatever happens to "
            "show up in volume/movers screeners, which skew heavily toward penny stocks and rarely "
            "surface genuine large-caps at all. Each call returns the NEXT batch in rotation, continuing "
            "from wherever the last call left off (state persists across cycles). Use this periodically "
            "to genuinely broaden what you're evaluating, not only SPY/QQQ or whatever movers surfaced."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "batch_size": {"type": "integer", "description": "How many tickers to return, default 15"},
            },
        },
    },
    {
        "name": "get_recent_activity_log",
        "description": "Read recent logged events from this agent's own history — past trades, reasoning, and outcomes — to inform self-assessment and strategy adjustment.",
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "Number of recent events to retrieve, default 100"}},
        },
    },
    {
        "name": "get_setup_performance",
        "description": (
            "Your actual historical win rate and P&L, grouped by the setup_type label you gave each "
            "trade at entry (via place_spread_order). Use this BEFORE sizing or entering a trade whose "
            "setup resembles one you've traded before — pattern-matching on 'this looks like it's "
            "working' in the moment is not the same as knowing whether that setup type has actually "
            "made or lost money historically, and this is the only way to tell the difference. Reconstructs "
            "real round-trip trades from actual fills and expirations (not just your own stated rationale), "
            "so this reflects what really happened, including trades you may not remember clearly. Trades "
            "placed before setup_type tagging existed, or where the tag is missing, are reported separately "
            "as 'untagged' rather than silently dropped or guessed at."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "min_trades": {"type": "integer", "description": "Only include setup types with at least this many CLOSED trades in the summary (still-thin setups are noisy). Default 1 (show everything)."},
            },
        },
    },
    {
        "name": "get_portfolio_greeks",
        "description": (
            "Your net delta, theta, vega, and gamma exposure aggregated ACROSS ALL currently open "
            "positions — both by underlying and as one portfolio-wide total. You can reason correctly "
            "about a single spread's own risk and still be blind to concentration building up across "
            "several trades on the same or correlated underlyings, because nothing else shows you the "
            "combined picture — this is that picture. Call this before opening a new position in an "
            "underlying you already hold, and periodically whenever you have multiple open positions, "
            "not just when something has already gone wrong. Delta/theta/vega/gamma are computed "
            "per-position as contracts x 100 x the per-share Greek from live option snapshots, signed "
            "for long vs short, then summed — a genuine portfolio aggregate, not a per-trade estimate. "
            "Flags any single underlying responsible for an outsized share of total portfolio delta so "
            "concentration is visible without you having to compute it yourself."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "concentration_threshold": {"type": "number", "description": "Flag an underlying if its share of total absolute portfolio delta exceeds this fraction (0-1). Default 0.4 (40%)."},
            },
        },
    },
    {
        "name": "get_order_fill_status",
        "description": (
            "Check the ACTUAL current status and fill price of one or more orders you've placed, by "
            "order id (returned in place_spread_order's response). You reason about mid-prices and "
            "limit prices at the moment you submit an order, but you don't automatically learn what "
            "actually happened after that — whether it filled, at what real price, or whether it's "
            "still sitting open. Use this later in the same cycle or in a future cycle to check on any "
            "order you're uncertain about, rather than assuming a limit order filled at your limit "
            "price just because you didn't hear otherwise. Also useful for reconciling your own "
            "assumed P&L against what actually got filled."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_ids": {"type": "array", "items": {"type": "string"}, "description": "One or more Alpaca order ids to check, e.g. from a previous place_spread_order response."},
            },
            "required": ["order_ids"],
        },
    },
    {
        "name": "get_market_context",
        "description": (
            "Current VIX and S&P 500 index levels, with a plain-language volatility-regime label "
            "(low/normal/elevated/high) so you don't have to interpret the raw VIX number yourself "
            "every time. Use this to sanity-check whether the broad market backdrop supports the kind "
            "of setup you're considering — e.g. a low-VIX regime behaves very differently from an "
            "elevated one for the same nominal strategy. This does NOT cover scheduled macro events "
            "(Fed meetings, CPI/jobs releases) or news headlines — it's index levels only, not an "
            "economic calendar or news feed; don't infer the presence or absence of a scheduled event "
            "from this tool."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "report_tooling_issue",
        "description": (
            "Report a suspected bug or unexpected behavior in one of YOUR OWN tools — not a market "
            "observation, a code/infrastructure problem. You cannot fix your own tools; you have no "
            "access to your own source code and no ability to modify it. What you CAN do is report the "
            "problem clearly and immediately, the moment you notice it, so a human can review and fix "
            "it quickly rather than only discovering it later by reading through trade history. Use this "
            "whenever a tool behaves in a way that doesn't match its description, produces a result you "
            "didn't expect given your input, or you find yourself having to work around something rather "
            "than use a tool as intended (e.g., using close_position leg-by-leg instead of a tool that "
            "should have handled it directly). Report as soon as you notice the issue, in the same cycle "
            "— don't wait."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "severity": {"type": "string", "enum": ["low", "medium", "high"], "description": "high = caused or risked an unintended trade/position change; medium = caused confusion or a workaround but no unintended action; low = cosmetic or informational"},
                "tool_name": {"type": "string", "description": "The tool you believe has the problem"},
                "what_you_tried": {"type": "string", "description": "What you called the tool to do, and with what inputs"},
                "what_happened": {"type": "string", "description": "What actually happened, including any resulting order/position changes"},
                "suspected_cause": {"type": "string", "description": "Your best guess at the root cause, if you have one — this is a hypothesis for the human to check, not a diagnosis you can verify yourself"},
            },
            "required": ["severity", "tool_name", "what_you_tried", "what_happened", "suspected_cause"],
        },
    },
]


class ToolDispatcher:
    def __init__(self, config=CONFIG):
        self.config = config

    async def dispatch(self, tool_name: str, tool_input: dict) -> str:
        """Returns a JSON string result to feed back to Claude as tool_result content."""
        try:
            if tool_name == "get_account_info":
                return await self._get_account_info()
            elif tool_name == "get_positions":
                return await self._get_positions()
            elif tool_name == "get_stock_quote":
                return await self._get_stock_quote(tool_input["symbol"])
            elif tool_name == "get_stock_bars":
                return await self._get_stock_bars(tool_input["symbol"], tool_input.get("minutes", 60))
            elif tool_name == "get_option_chain":
                return await self._get_option_chain(tool_input["underlying"], tool_input.get("expiration"))
            elif tool_name == "get_most_active_stocks":
                return await self._get_most_active_stocks(tool_input.get("by", "volume"), tool_input.get("top", 10))
            elif tool_name == "get_market_movers":
                return await self._get_market_movers(tool_input.get("top", 10))
            elif tool_name == "get_sp500_batch":
                return self._get_sp500_batch(tool_input.get("batch_size", 15))
            elif tool_name == "place_spread_order":
                return await self._place_spread_order(tool_input)
            elif tool_name == "close_position":
                return await self._close_position(tool_input["symbol"])
            elif tool_name == "close_all_positions":
                return await self._close_all_positions()
            elif tool_name == "get_recent_activity_log":
                return self._get_recent_activity_log(tool_input.get("limit", 100))
            elif tool_name == "get_setup_performance":
                return await self._get_setup_performance(tool_input.get("min_trades", 1))
            elif tool_name == "get_portfolio_greeks":
                return await self._get_portfolio_greeks(tool_input.get("concentration_threshold", 0.4))
            elif tool_name == "get_order_fill_status":
                return await self._get_order_fill_status(tool_input["order_ids"])
            elif tool_name == "get_market_context":
                return await self._get_market_context()
            elif tool_name == "report_tooling_issue":
                return self._report_tooling_issue(tool_input)
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as e:
            log_event("tool_dispatch_error", {"tool": tool_name, "input": tool_input, "error": str(e)})
            return json.dumps({"error": f"Tool '{tool_name}' failed: {str(e)}"})

    async def _get_account_info(self) -> str:
        async with AlpacaMCPClient(self.config) as mcp:
            result = await mcp.call_tool("get_account_info", {})
        return json.dumps(result)

    async def _get_positions(self) -> str:
        async with AlpacaMCPClient(self.config) as mcp:
            result = await mcp.call_tool("get_all_positions", {})
        return json.dumps(result)

    async def _get_stock_quote(self, symbol: str) -> str:
        async with AlpacaMCPClient(self.config) as mcp:
            result = await mcp.call_tool("get_stock_latest_quote", {"symbols": symbol})
        return json.dumps(result)

    async def _get_stock_bars(self, symbol: str, minutes: int) -> str:
        from datetime import datetime, timedelta, timezone
        start = (datetime.now(timezone.utc) - timedelta(minutes=minutes * 2)).isoformat()
        async with AlpacaMCPClient(self.config) as mcp:
            result = await mcp.call_tool("get_stock_bars", {"symbols": symbol, "timeframe": "1Min", "start": start})
        return json.dumps(result)

    async def _get_option_chain(self, underlying: str, expiration: str = None) -> str:
        params = {"underlying_symbol": underlying}
        if expiration:
            params["expiration_date"] = expiration
        async with AlpacaMCPClient(self.config) as mcp:
            result = await mcp.call_tool("get_option_chain", params)
        return json.dumps(result)

    def _looks_like_non_common_stock(self, symbol: str) -> bool:
        """
        Heuristic filter: on Nasdaq-style tickers, a 5th character of W/U/R
        typically denotes a warrant, unit, or rights offering — not the
        underlying common stock, and these essentially never have liquid
        (or any) listed options. Only applied to 5-character symbols
        specifically to avoid false-positives on legitimate short tickers
        that happen to end in those letters (e.g. 3-4 letter common tickers).
        Found necessary via live testing: get_market_movers surfaced
        "BRLSW" and "PASW" as top gainers, both warrants, not tradeable
        via options at all.
        """
        return len(symbol) == 5 and symbol[-1] in ("W", "U", "R")

    async def _get_most_active_stocks(self, by: str, top: int) -> str:
        async with AlpacaMCPClient(self.config) as mcp:
            result = await mcp.call_tool("get_most_active_stocks", {"by": by, "top": top})
        data = unwrap_data(result)
        actives = data.get("most_actives", []) if isinstance(data, dict) else []
        filtered = [a for a in actives if not self._looks_like_non_common_stock(a.get("symbol", ""))]
        removed = len(actives) - len(filtered)
        return json.dumps({
            "most_actives": filtered,
            "note": (
                f"Filtered out {removed} likely warrant/unit/rights symbols (not options-tradeable). "
                "Remaining results are still raw 'most active by share volume' data, which skews toward "
                "low-priced, high-turnover names — many won't have liquid options either. Cross-check "
                "against get_option_chain before treating any of these as a real candidate."
            ) if removed else "Note: raw 'most active by volume' data skews toward low-priced, high-turnover names — cross-check against get_option_chain before treating any as a real candidate.",
        })

    async def _get_market_movers(self, top: int) -> str:
        async with AlpacaMCPClient(self.config) as mcp:
            result = await mcp.call_tool("get_market_movers", {"market_type": "stocks", "top": top})
        data = unwrap_data(result)
        gainers = data.get("gainers", []) if isinstance(data, dict) else []
        losers = data.get("losers", []) if isinstance(data, dict) else []

        MIN_PRICE = 10.0  # excludes penny-stock noise; real optionable large-caps are essentially never this cheap

        def keep(item):
            symbol = item.get("symbol", "")
            price = float(item.get("price", 0) or 0)
            return price >= MIN_PRICE and not self._looks_like_non_common_stock(symbol)

        filtered_gainers = [g for g in gainers if keep(g)]
        filtered_losers = [l for l in losers if keep(l)]
        removed = (len(gainers) - len(filtered_gainers)) + (len(losers) - len(filtered_losers))

        return json.dumps({
            "gainers": filtered_gainers,
            "losers": filtered_losers,
            "note": (
                f"Filtered out {removed} symbols under ${MIN_PRICE:.0f} or that look like warrants/units/rights "
                "(speculative penny-stock moves that dominate raw % gainers/losers data and essentially never "
                "have liquid options markets)."
            ),
        })

    def _get_sp500_batch(self, batch_size: int) -> str:
        from config.sp500_tickers import SP500_TICKERS
        from config.sp500_rotation import get_next_batch
        batch = get_next_batch(SP500_TICKERS, batch_size)
        return json.dumps({
            "tickers": batch,
            "note": (
                f"Batch of {len(batch)} from the S&P 500, rotating — the next call continues from here "
                "rather than repeating this same batch. These are genuine index constituents, not "
                "screener output, so no further liquidity filtering has been applied here; check "
                "get_option_chain for any name you're seriously considering."
            ),
        })

    async def _place_spread_order(self, tool_input: dict) -> str:
        action = tool_input["action"]
        if action not in ("open", "close"):
            return json.dumps({"rejected": True, "backstop": "invalid_action", "reason": f"action must be 'open' or 'close', got '{action}'."})

        underlying = tool_input["underlying"]
        buy_symbol = tool_input["buy_symbol"]
        sell_symbol = tool_input["sell_symbol"]
        contracts = int(tool_input["contracts"])
        limit_price = float(tool_input["limit_price"])
        max_loss_per_contract = float(tool_input["max_loss_per_contract"])
        rationale = tool_input["rationale"]
        setup_type = (tool_input.get("setup_type") or "").strip()

        # setup_type is required on open (not a hard risk backstop — it's a
        # data-quality gate for get_setup_performance, rejected the same way
        # so an untagged trade never silently breaks the win-rate-by-setup
        # picture rather than failing loudly at the point it would happen.
        if action == "open" and not setup_type:
            log_event("backstop_rejected", {"backstop": "missing_setup_type", "reason": "setup_type is required when action='open'.", "input": tool_input})
            return json.dumps({"rejected": True, "backstop": "missing_setup_type", "reason": "setup_type is required when action='open' — see get_setup_performance for why this matters."})

        # --- Hard backstop 1: defined risk only ---
        risk_check = check_defined_risk(sell_symbol, buy_symbol, short_side="sell", long_side="buy")
        if not risk_check.approved:
            log_event("backstop_rejected", {"backstop": "defined_risk", "reason": risk_check.reason, "input": tool_input})
            return json.dumps({"rejected": True, "backstop": "defined_risk", "reason": risk_check.reason})

        # --- Hard backstop 2: spread economics sanity ---
        economics_check = check_spread_economics(buy_symbol, sell_symbol, limit_price)
        if not economics_check.approved:
            log_event("backstop_rejected", {"backstop": "spread_economics", "reason": economics_check.reason, "input": tool_input})
            return json.dumps({"rejected": True, "backstop": "spread_economics", "reason": economics_check.reason})

        # --- Hard backstop 3: per-trade sizing cap ---
        async with AlpacaMCPClient(self.config) as mcp:
            account = await mcp.call_tool("get_account_info", {})
        equity = float(unwrap_data(account).get("equity", 0))

        sizing_check = check_position_sizing(max_loss_per_contract, contracts, equity)
        if not sizing_check.approved:
            log_event("backstop_rejected", {"backstop": "position_sizing", "reason": sizing_check.reason, "input": tool_input})
            return json.dumps({"rejected": True, "backstop": "position_sizing", "reason": sizing_check.reason})

        # Both backstops passed — place the order. Intent suffix is derived
        # directly and unambiguously from the required 'action' field — this
        # is the fix for a real incident where a hardcoded "_to_open" caused
        # an attempted position reduction to instead open an ADDITIONAL
        # position, doubling size unintentionally. There is no code path
        # left where intent can be inferred wrong: it's always exactly what
        # the agent explicitly declared.
        intent_suffix = "open" if action == "open" else "close"
        legs = [
            {"symbol": sell_symbol, "side": "sell", "ratio_qty": "1", "position_intent": f"sell_to_{intent_suffix}"},
            {"symbol": buy_symbol, "side": "buy", "ratio_qty": "1", "position_intent": f"buy_to_{intent_suffix}"},
        ]

        # NBBO snapshot at the moment of submission — logged AND returned
        # immediately, not something the agent has to ask for separately.
        # Real gap this closes: after a close attempt on META at -$3.75
        # didn't fill, the agent had to guess afterward what the market
        # was actually offering ("I had to guess the market was offering
        # $3.29-3.49"). Recording the actual NBBO at the exact moment the
        # limit price was chosen means that guess is never necessary again
        # — a non-fill can be explained by comparing the limit to what the
        # market genuinely was, not reconstructed after the fact. Best-effort:
        # if the snapshot call itself fails, the order still proceeds rather
        # than being blocked by a data-quality nicety.
        nbbo_at_submission = {}
        try:
            async with AlpacaMCPClient(self.config) as mcp:
                snap_result = await mcp.call_tool("get_option_snapshot", {"symbols": f"{buy_symbol},{sell_symbol}"})
            snaps = unwrap_data(snap_result) or {}
            if isinstance(snaps, dict) and "data" in snaps and set(snaps.keys()) <= {"data", "next_page_token"}:
                snaps = snaps["data"]
            for sym in (buy_symbol, sell_symbol):
                q = (snaps.get(sym) or {}).get("latestQuote") or {}
                if q:
                    nbbo_at_submission[sym] = {"bid": q.get("bp") or q.get("bid_price"), "ask": q.get("ap") or q.get("ask_price")}
        except Exception as e:
            nbbo_at_submission = {"error": f"NBBO snapshot failed, order proceeding anyway: {e}"}

        log_event("agent_order_submit", {
            "action": action,
            "underlying": underlying,
            "sell_symbol": sell_symbol,
            "buy_symbol": buy_symbol,
            "contracts": contracts,
            "limit_price": limit_price,
            "max_loss_per_contract": max_loss_per_contract,
            "rationale": rationale,
            "setup_type": setup_type,
            "nbbo_at_submission": nbbo_at_submission,
        })
        async with AlpacaMCPClient(self.config) as mcp:
            result = await mcp.call_tool("place_option_order", {
                "qty": str(contracts),
                "type": "limit",
                "time_in_force": "day",
                "order_class": "mleg",
                "legs": legs,
                "limit_price": str(limit_price),
            })
        log_event("agent_order_response", {"result": result})
        order_data = unwrap_data(result) or {}
        order_id = order_data.get("id") if isinstance(order_data, dict) else None
        order_status = order_data.get("status") if isinstance(order_data, dict) else None

        # Capture/clear the position's thesis regardless of whether this
        # order has actually filled yet — if it never fills, the live
        # position never appears and position_memory.prune_closed() in
        # autonomous_agent.py's per-cycle brief will remove the stale
        # entry on its own next cycle. Recording at submission (not fill)
        # keeps this simple and avoids needing to correlate an async fill
        # event back to this specific order later.
        if action == "open":
            position_memory.record_entry(buy_symbol, sell_symbol, underlying, rationale, limit_price)
        else:
            position_memory.clear_entry(buy_symbol, sell_symbol)

        return json.dumps({
            "rejected": False,
            "order_result": result,
            "nbbo_at_submission": nbbo_at_submission,
            "note": (
                f"Order id {order_id} submitted with status '{order_status}'. If it's not yet 'filled', "
                "this is a day limit order — it may fill later this session or expire unfilled. Use "
                "get_order_fill_status with this order id later in the cycle or a future cycle to check "
                "what actually happened, rather than assuming it filled at your limit price."
            ) if order_id else "Order submitted.",
        })

    async def _close_position(self, symbol: str) -> str:
        log_event("agent_close_position", {"symbol": symbol})
        async with AlpacaMCPClient(self.config) as mcp:
            result = await mcp.call_tool("close_position", {"symbol_or_asset_id": symbol})
        return json.dumps(result)

    async def _close_all_positions(self) -> str:
        log_event("agent_close_all_positions", {})
        async with AlpacaMCPClient(self.config) as mcp:
            result = await mcp.call_tool("close_all_positions", {"cancel_orders": True})
        return json.dumps(result)

    def _get_recent_activity_log(self, limit: int) -> str:
        from execution.trade_logger import read_events
        events = read_events(limit=limit)
        return json.dumps(events)

    def _setup_type_map(self) -> dict:
        """
        Reads the FULL event log (not just the recent tail used by
        get_recent_activity_log) for every agent_order_submit event with
        action='open', and returns {frozenset({buy_symbol, sell_symbol}):
        setup_type}. Keyed on the exact traded symbol pair rather than
        underlying+time, since that's the only thing guaranteed to match
        build_trade_records' initial_open_events unambiguously — two
        different trades on the same underlying can open minutes apart.
        Trades placed before setup_type existed, or with a blank tag,
        simply won't appear in this map — callers treat that as
        'untagged', not an error.
        """
        import json as _json
        import os as _os
        from execution.trade_logger import LOG_PATH

        mapping = {}
        if not _os.path.exists(LOG_PATH):
            return mapping
        with open(LOG_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = _json.loads(line)
                except Exception:
                    continue
                if entry.get("event_type") != "agent_order_submit":
                    continue
                payload = entry.get("payload", {})
                if payload.get("action") != "open":
                    continue
                setup_type = (payload.get("setup_type") or "").strip()
                if not setup_type:
                    continue
                buy_symbol = payload.get("buy_symbol")
                sell_symbol = payload.get("sell_symbol")
                if not buy_symbol or not sell_symbol:
                    continue
                mapping[frozenset({buy_symbol, sell_symbol})] = setup_type
        return mapping

    async def _get_setup_performance(self, min_trades: int) -> str:
        from collections import defaultdict
        from execution.trade_records import build_trade_records

        async with AlpacaMCPClient(self.config) as mcp:
            orders_result = await mcp.call_tool("get_orders", {"status": "all", "limit": 500})
            positions_result = await mcp.call_tool("get_all_positions", {})
            activities_result = await mcp.call_tool("get_account_activities", {"activity_types": "OPEXP"})

        orders = unwrap_data(orders_result)
        if isinstance(orders, dict):
            orders = orders.get("orders", orders.get("data", []))
        positions = unwrap_data(positions_result)
        if isinstance(positions, dict):
            # Confirmed shape (see execution/alpaca_client.py's
            # open_positions, verified against a live response): the list
            # is nested under "result", not "positions" or "data" — this
            # was silently returning [] even after the tool-name fix above.
            positions = positions.get("result", [])
        if isinstance(expiry_activities, dict):
            expiry_activities = expiry_activities.get("activities", expiry_activities.get("data", []))

        trades = build_trade_records(orders or [], positions or [], expiry_activities or [])
        setup_map = self._setup_type_map()

        stats = defaultdict(lambda: {"closed_trades": 0, "wins": 0, "losses": 0, "flats": 0, "total_pnl": 0.0, "open_trades": 0})
        for trade in trades:
            open_symbols = frozenset(e["symbol"] for e in trade.get("initial_open_events", []))
            setup_type = setup_map.get(open_symbols, "untagged")
            if trade["status"] == "open":
                stats[setup_type]["open_trades"] += 1
                continue
            s = stats[setup_type]
            s["closed_trades"] += 1
            s["total_pnl"] += trade["outcome"]
            if trade["profit_loss"] == "win":
                s["wins"] += 1
            elif trade["profit_loss"] == "loss":
                s["losses"] += 1
            else:
                s["flats"] += 1

        summary = []
        for setup_type, s in stats.items():
            if s["closed_trades"] < min_trades:
                continue
            win_rate = (s["wins"] / s["closed_trades"]) if s["closed_trades"] else None
            avg_pnl = (s["total_pnl"] / s["closed_trades"]) if s["closed_trades"] else None
            summary.append({
                "setup_type": setup_type,
                "closed_trades": s["closed_trades"],
                "open_trades": s["open_trades"],
                "wins": s["wins"],
                "losses": s["losses"],
                "flats": s["flats"],
                "win_rate": round(win_rate, 3) if win_rate is not None else None,
                "total_pnl": round(s["total_pnl"], 2),
                "avg_pnl_per_trade": round(avg_pnl, 2) if avg_pnl is not None else None,
            })
        summary.sort(key=lambda x: x["closed_trades"], reverse=True)

        omitted_below_min_trades = [
            {"setup_type": st, "closed_trades": s["closed_trades"]}
            for st, s in stats.items() if s["closed_trades"] < min_trades and s["closed_trades"] > 0
        ]

        return json.dumps({
            "by_setup_type": summary,
            "omitted_below_min_trades": omitted_below_min_trades,
            "note": (
                "'untagged' covers trades placed before setup_type existed or with a missing tag — "
                "it is not itself a real setup type, so don't draw conclusions about a specific "
                "strategy from its numbers. Every new open now requires a setup_type, so this bucket "
                "should stop growing going forward."
            ),
        })

    async def _get_portfolio_greeks(self, concentration_threshold: float) -> str:
        from collections import defaultdict
        from execution.trade_records import parse_occ_symbol, parse_underlying

        async with AlpacaMCPClient(self.config) as mcp:
            positions_result = await mcp.call_tool("get_all_positions", {})

        positions = unwrap_data(positions_result)
        if isinstance(positions, dict):
            # Same confirmed shape as above — nested under "result".
            positions = positions.get("result", [])
        positions = positions or []

        # Only OCC option symbols carry Greeks — a bare stock/ETF position
        # (length < 16 for an OCC symbol) has none and is skipped here
        # rather than guessed at.
        option_positions = [p for p in positions if p.get("symbol") and len(p["symbol"]) >= 16]

        if not option_positions:
            return json.dumps({
                "positions_included": 0,
                "by_underlying": [],
                "portfolio_totals": {"delta": 0.0, "theta": 0.0, "vega": 0.0, "gamma": 0.0},
                "concentration_warnings": [],
                "note": "No open option positions to aggregate.",
            })

        symbols = sorted({p["symbol"] for p in option_positions})
        async with AlpacaMCPClient(self.config) as mcp:
            snapshot_result = await mcp.call_tool("get_option_snapshot", {"symbols": ",".join(symbols)})
        snapshots = unwrap_data(snapshot_result)
        if isinstance(snapshots, dict) and "data" in snapshots and set(snapshots.keys()) <= {"data", "next_page_token"}:
            snapshots = snapshots["data"]
        snapshots = snapshots or {}

        MULTIPLIER = 100
        by_underlying = defaultdict(lambda: {"delta": 0.0, "theta": 0.0, "vega": 0.0, "gamma": 0.0, "positions": 0})
        totals = {"delta": 0.0, "theta": 0.0, "vega": 0.0, "gamma": 0.0}
        missing_greeks = []

        for p in option_positions:
            symbol = p["symbol"]
            snap = snapshots.get(symbol) or {}
            greeks = snap.get("greeks") or {}
            if not greeks:
                missing_greeks.append(symbol)
                continue

            try:
                qty = abs(float(p.get("qty", 0) or 0))
            except (TypeError, ValueError):
                qty = 0.0
            side = (p.get("side") or "").lower()
            # Alpaca positions report qty as non-negative with a separate
            # side field ("long"/"short"); fall back to a negative qty
            # convention if a caller/mock ever reports it that way instead.
            is_short = side == "short" or float(p.get("qty", 0) or 0) < 0
            signed_qty = -qty if is_short else qty

            underlying = parse_underlying(symbol)
            for greek in ("delta", "theta", "vega", "gamma"):
                raw = greeks.get(greek)
                if raw is None:
                    continue
                exposure = signed_qty * float(raw) * MULTIPLIER
                by_underlying[underlying][greek] += exposure
                totals[greek] += exposure
            by_underlying[underlying]["positions"] += 1

        total_abs_delta = sum(abs(v["delta"]) for v in by_underlying.values())
        concentration_warnings = []
        breakdown = []
        for underlying, g in by_underlying.items():
            share_of_delta = (abs(g["delta"]) / total_abs_delta) if total_abs_delta else 0.0
            breakdown.append({
                "underlying": underlying,
                "positions": g["positions"],
                "delta": round(g["delta"], 2),
                "theta": round(g["theta"], 2),
                "vega": round(g["vega"], 2),
                "gamma": round(g["gamma"], 4),
                "share_of_portfolio_abs_delta": round(share_of_delta, 3),
            })
            if share_of_delta >= concentration_threshold:
                concentration_warnings.append({
                    "underlying": underlying,
                    "share_of_portfolio_abs_delta": round(share_of_delta, 3),
                    "positions": g["positions"],
                })
        breakdown.sort(key=lambda x: abs(x["delta"]), reverse=True)
        concentration_warnings.sort(key=lambda x: x["share_of_portfolio_abs_delta"], reverse=True)

        return json.dumps({
            "positions_included": len(option_positions) - len(missing_greeks),
            "positions_missing_greeks": missing_greeks,
            "by_underlying": breakdown,
            "portfolio_totals": {k: round(v, 2) for k, v in totals.items()},
            "concentration_warnings": concentration_warnings,
            "note": (
                "delta/theta/vega are position-signed (long positive, short negative) and summed as "
                "contracts x 100 x per-share Greek from live snapshots. gamma is reported per underlying "
                "but not used for concentration flagging. A symbol in positions_missing_greeks had no "
                "Greeks in the snapshot response (e.g. illiquid contract, feed gap) and was excluded "
                "from totals rather than estimated."
            ),
        })

    async def _get_order_fill_status(self, order_ids: list) -> str:
        results = []
        async with AlpacaMCPClient(self.config) as mcp:
            for order_id in order_ids:
                try:
                    raw = await mcp.call_tool("get_order_by_id", {"order_id": order_id, "nested": True})
                    order = unwrap_data(raw) or {}
                    legs = order.get("legs") or [order]
                    results.append({
                        "order_id": order_id,
                        "status": order.get("status"),
                        "filled_qty": order.get("filled_qty"),
                        "filled_avg_price": order.get("filled_avg_price"),
                        "submitted_at": order.get("submitted_at"),
                        "filled_at": order.get("filled_at"),
                        "legs": [
                            {
                                "symbol": leg.get("symbol"),
                                "side": leg.get("side"),
                                "position_intent": leg.get("position_intent"),
                                "status": leg.get("status"),
                                "filled_qty": leg.get("filled_qty"),
                                "filled_avg_price": leg.get("filled_avg_price"),
                            }
                            for leg in legs
                        ],
                    })
                except Exception as e:
                    results.append({"order_id": order_id, "error": str(e)})

        return json.dumps({
            "orders": results,
            "note": (
                "filled_avg_price here is the REAL fill price from Alpaca, not your assumed mid-price "
                "or limit price at submission — use it to reconcile actual P&L against what you expected. "
                "A status other than 'filled' (e.g. 'new', 'accepted', 'canceled', 'expired') means the "
                "order did not execute as you may be assuming."
            ),
        })

    async def _get_market_context(self) -> str:
        # get_index_latest_values is not a real registered MCP tool — this
        # was the root cause of it failing every single cycle. The working
        # pattern (confirmed via fast_layer/market_data.py's vix_snapshot,
        # and via every successful get_stock_quote call tonight) is
        # get_stock_latest_quote. SPX itself also isn't a standard
        # Alpaca-quotable symbol; SPY is the real, liquid, always-quotable
        # proxy for "S&P 500 level" and is what's actually used for signal
        # generation elsewhere in this codebase.
        async with AlpacaMCPClient(self.config) as mcp:
            raw = await mcp.call_tool("get_stock_latest_quote", {"symbols": "VIX,SPY"})
        data = unwrap_data(raw) or {}
        if isinstance(data, dict) and "data" in data and set(data.keys()) <= {"data", "next_page_token"}:
            data = data["data"]
        quotes = data.get("quotes", {}) if isinstance(data, dict) else {}

        def _extract_value(entry):
            if not isinstance(entry, dict):
                return None
            for key in ("bidPrice", "bp", "askPrice", "ap"):
                if key in entry and entry[key] is not None:
                    return float(entry[key])
            return None

        vix = _extract_value(quotes.get("VIX"))
        spy = _extract_value(quotes.get("SPY"))

        if vix is None:
            regime = "unknown"
        elif vix < 15:
            regime = "low"
        elif vix < 20:
            regime = "normal"
        elif vix < 30:
            regime = "elevated"
        else:
            regime = "high"

        return json.dumps({
            "vix": vix,
            "spy": spy,
            "vix_regime": regime,
            "note": (
                "Index levels only — this does NOT include scheduled macro events (Fed meetings, "
                "CPI/jobs releases) or news headlines. spy is the S&P 500 ETF proxy (SPX itself "
                "isn't a standard quotable symbol on this feed). A missing vix/spy value means the "
                "underlying quote call returned no data for that symbol right now, not that the "
                "market is closed."
            ),
        })

    def _report_tooling_issue(self, tool_input: dict) -> str:
        """
        Writes to two places: the structured event log (so it's part of
        the same audit trail as everything else) AND a dedicated,
        human-readable TOOLING_ISSUES.md file (so a human doesn't have
        to dig through the full log to notice it). This is the agent's
        only channel for surfacing a suspected bug in its own tools —
        it cannot fix the tool itself, only report it clearly and
        immediately so a human can.
        """
        import os
        from datetime import datetime, timezone

        severity = tool_input.get("severity", "medium")
        tool_name = tool_input.get("tool_name", "unknown")
        what_tried = tool_input.get("what_you_tried", "")
        what_happened = tool_input.get("what_happened", "")
        suspected_cause = tool_input.get("suspected_cause", "")
        timestamp = datetime.now(timezone.utc).isoformat()

        report = {
            "timestamp": timestamp,
            "severity": severity,
            "tool_name": tool_name,
            "what_you_tried": what_tried,
            "what_happened": what_happened,
            "suspected_cause": suspected_cause,
        }
        log_event("tooling_issue_reported", report)

        issues_path = os.path.join(os.path.dirname(__file__), "..", "TOOLING_ISSUES.md")
        severity_marker = {"high": "🔴", "medium": "🟡", "low": "⚪"}.get(severity, "🟡")
        entry = (
            f"\n## {severity_marker} [{severity.upper()}] {tool_name} — {timestamp}\n\n"
            f"**What I tried:** {what_tried}\n\n"
            f"**What happened:** {what_happened}\n\n"
            f"**Suspected cause:** {suspected_cause}\n\n"
            f"---\n"
        )
        file_exists = os.path.exists(issues_path)
        with open(issues_path, "a") as f:
            if not file_exists:
                f.write("# Tooling Issues Reported by the Autonomous Agent\n\n"
                        "Each entry below was written by the agent itself the moment it noticed a "
                        "tool behaving unexpectedly. It cannot fix these — only report them for human "
                        "review. Check this file for anything unreviewed.\n")
            f.write(entry)

        return json.dumps({
            "acknowledged": True,
            "message": "Issue reported. A human will review TOOLING_ISSUES.md. You cannot fix this yourself — if possible, avoid the problematic tool/pattern for the rest of this session.",
        })
