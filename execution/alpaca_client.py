"""
Order placement and account access, routed through Alpaca's official
MCP server (execution/mcp_client.py) rather than a direct alpaca-py
client. This is the piece that satisfies the hackathon's "use Alpaca's
Trading API via its MCP server or CLI" requirement.

Kept separate from fast_layer.market_data (read-only, also MCP-backed)
so no code path which merely observes the market can accidentally
place an order — the async context manager here is only entered by
code that intends to act.
"""
from config import CONFIG
from execution.mcp_client import AlpacaMCPClient, unwrap_data
from execution.trade_logger import log_event


class AlpacaExecutionClient:
    """
    Thin async wrapper around AlpacaMCPClient with domain-specific
    methods and logging. Every method opens its own MCP session
    (stdio subprocess) and closes it cleanly — simplest correct
    behavior for a bot that runs on an interval rather than holding
    a long-lived connection.
    """

    def __init__(self, config=CONFIG):
        assert config.alpaca.paper is True, "This repo is hard-locked to paper trading."
        self.config = config

    async def account_snapshot(self) -> dict:
        async with AlpacaMCPClient(self.config) as mcp:
            raw = await mcp.call_tool("get_account_info", {})
            acct = unwrap_data(raw)
            return {
                "equity": float(acct.get("equity", 0)),
                "buying_power": float(acct.get("buying_power", 0)),
                "options_buying_power": float(acct.get("options_buying_power", 0) or 0),
                "raw": raw,
            }

    async def open_positions(self) -> list:
        async with AlpacaMCPClient(self.config) as mcp:
            raw = await mcp.call_tool("get_all_positions", {})
            data = unwrap_data(raw)
            # get_all_positions nests the list one level deeper as
            # {"result": [...]}, not directly as a top-level list —
            # verified against a live response, not assumed.
            if isinstance(data, dict):
                return data.get("result", [])
            return data if isinstance(data, list) else []

    async def market_clock(self) -> dict:
        """
        Alpaca's clock endpoint, via the MCP server's get_clock tool —
        used in preference to hand-rolled weekday/hour logic because it
        correctly accounts for market holidays and early closes, not
        just "is it Mon-Fri 9:30-4."
        Returns {"is_open": bool, "next_open": iso str, "next_close": iso str}.
        """
        async with AlpacaMCPClient(self.config) as mcp:
            raw = await mcp.call_tool("get_clock", {})
            data = unwrap_data(raw)
            return {
                "is_open": bool(data.get("is_open", False)),
                "next_open": data.get("next_open"),
                "next_close": data.get("next_close"),
            }

    async def submit_vertical_spread(
        self,
        short_symbol: str,
        long_symbol: str,
        contracts: int,
        limit_credit: float,
        client_order_id: str = None,
    ) -> dict:
        """
        Submit a credit vertical spread as a multi-leg limit order via
        the MCP place_option_order tool. limit_price is negative for
        multi-leg credit (proceeds), per the tool's documented convention.
        """
        legs = [
            {"symbol": short_symbol, "side": "sell", "ratio_qty": "1", "position_intent": "sell_to_open"},
            {"symbol": long_symbol, "side": "buy", "ratio_qty": "1", "position_intent": "buy_to_open"},
        ]
        log_event("order_submit", {
            "short_symbol": short_symbol,
            "long_symbol": long_symbol,
            "contracts": contracts,
            "limit_credit": limit_credit,
        })
        async with AlpacaMCPClient(self.config) as mcp:
            result = await mcp.call_tool("place_option_order", {
                "qty": str(contracts),
                "type": "limit",
                "time_in_force": "day",
                "order_class": "mleg",
                "legs": legs,
                "limit_price": str(-abs(limit_credit)),  # negative = credit/proceeds
                "client_order_id": client_order_id,
            })
        log_event("order_response", {"result": result})
        return result

    async def close_position(self, symbol: str) -> dict:
        log_event("position_close_request", {"symbol": symbol})
        async with AlpacaMCPClient(self.config) as mcp:
            result = await mcp.call_tool("close_position", {"symbol_or_asset_id": symbol})
        log_event("position_close_response", {"symbol": symbol, "result": result})
        return result

    async def close_all_positions(self, cancel_orders: bool = True) -> dict:
        log_event("flatten_all", {})
        async with AlpacaMCPClient(self.config) as mcp:
            result = await mcp.call_tool("close_all_positions", {"cancel_orders": cancel_orders})
        return result
