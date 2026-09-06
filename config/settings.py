"""
Central configuration for the Alpaca Options Agent.

All tunable parameters live here so strategy behavior can be audited
and adjusted without hunting through module code.
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)


@dataclass(frozen=True)
class AlpacaConfig:
    api_key: str = os.getenv("ALPACA_API_KEY", "")
    secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")
    base_url: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    data_url: str = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
    paper: bool = True  # hard-locked True; this repo never trades live


@dataclass(frozen=True)
class ClaudeConfig:
    # LLM_PROVIDER selects which backend agent_layer/llm_client.py routes to:
    # "anthropic" (default) or "featherless". Model string must match the
    # chosen provider's naming (e.g. "claude-sonnet-4-6" vs
    # "deepseek-ai/DeepSeek-V4-Pro").
    api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model: str = os.getenv(
        "LLM_MODEL",
        os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
    )
    max_tokens: int = 1500


@dataclass(frozen=True)
class StrategyConfig:
    # Underlyings the fast layer scans for 0DTE / short-dated credit spread setups.
    # SPY + QQQ: both among the most liquid 0DTE options that exist, and both
    # have VIX-family vol signals that are actually meaningful for them
    # (VIX for SPY/SPX, VXN conceptually for QQQ/NDX — though this repo uses
    # VIX as the shared regime signal for both given data availability).
    universe: tuple = ("SPY", "QQQ")

    # VIX regime gates. NOTE: as of this writing (Aug 2026) VIX has been
    # sitting near 2026 lows (~14-15) for weeks — a threshold of 20, borrowed
    # from a slower positional strategy, would leave this agent idle for the
    # entire hackathon window. Lowered to reflect current reality; IV rank
    # (relative to the underlying's own recent range) is the primary vol-
    # timing signal, with this as a much looser sanity floor rather than the
    # main gate. This value is also in the rules-review agent's adjustable
    # whitelist — see config/dynamic_overrides.py — so it can be tuned
    # further based on observed conditions rather than only by hand.
    vix_entry_threshold: float = 12.0
    vix_backwardation_block: bool = True  # skip new entries if term structure inverted

    # IV rank filter — only consider entries when IV rank (0-100) exceeds this.
    # Lowered from an initial 40 for the same reason as vix_entry_threshold
    # above: calibrated against current low-vol conditions rather than a
    # generically "safe" level. Also in the adjustable whitelist.
    min_iv_rank: float = 20.0

    # Target short-leg delta for credit spreads (absolute value)
    target_short_delta: float = 0.16
    delta_tolerance: float = 0.08  # widened from 0.05 to catch more real strikes

    # Spread width in strikes (dollars) for vertical spreads
    spread_width: float = 5.0

    # Fast-layer scan cadence (minutes) — deterministic layer, no LLM call here
    fast_layer_interval_minutes: int = 5

    # Agent layer is invoked only at these decision points, not every fast-layer tick
    agent_max_calls_per_session: int = 12

    # Exit rules (rule-based, no LLM in the loop for speed)
    profit_take_pct: float = 0.50   # close at 50% of max credit captured
    stop_loss_multiple: float = 2.0  # close if loss reaches 2x credit received
    hard_close_minutes_before_expiry: int = 15  # force-close 0DTE positions before close


@dataclass(frozen=True)
class RiskConfig:
    # Portfolio governor limits — checked before ANY new position, across the whole account
    max_concurrent_positions: int = 4
    max_contracts_per_trade: int = 5
    max_notional_pct_of_equity: float = 0.10  # per trade, of account equity
    max_portfolio_delta: float = 50.0          # net delta exposure ceiling (shares-equivalent)
    max_daily_loss_pct: float = 0.03           # kill-switch: halt new entries for the session
    catastrophic_drawdown_pct: float = 0.15    # kill-switch: flatten everything


@dataclass(frozen=True)
class Config:
    alpaca: AlpacaConfig = field(default_factory=AlpacaConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)


CONFIG = Config()
