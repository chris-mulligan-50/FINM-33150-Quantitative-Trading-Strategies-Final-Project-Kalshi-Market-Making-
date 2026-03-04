"""
ExecutionEngine.py

Central gateway for your Kalshi market-making simulation.

Inputs (per your architecture diagram)
-------------------------------------
From DataIngestor (macro data @ 1Hz):
  - SPX price
  - VIX value
  - SPY price

From Simulator (Kalshi market activity @ 1Hz):
  - taker prices (take_bid / take_ask) and optional best bid/ask snapshots
  - (fill intents are produced by Simulator using your queue assumption)

From MarketMaker:
  - quotes (bid/ask + sizes) computed from (fair value, VIX, positions)

From DeltaHedger:
  - hedge trades (SPY orders) computed from (SPY price, positions)

Outputs
-------
To Pricer:
  - SPX & VIX each tick, requesting fair value for each contract_id

To MarketMaker:
  - VIX each tick
  - fair value per contract_id each tick
  - position updates (per contract_id)

To PositionManager:
  - executed Kalshi fills (delayed by 1 second)
  - executed SPY hedge trades (delayed by 1 second)

To DeltaHedger:
  - SPY price each tick
  - aggregate inventory/positions

To Simulator:
  - current resting quotes by contract_id
  - hedge order(s) (optional), which the simulator treats as our bot orders

Key behavioral assumptions
-------------------------
- Execution delay of 1 second for SPY hedge trades.
- Kalshi fills are applied immediately when received from the Simulator.
- Queue assumption / fill detection lives in the Simulator; the engine receives fills as "intents".

This engine imports Pricer, DeltaHedger, and PositionManager from their own modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo

from DeltaHedger import DeltaHedger, HedgeOrder
from PositionManager import PositionManager
from Pricer import Pricer


# ============================================================
# Market hours (ET) for out-of-market spread widening
# ============================================================

_ET = ZoneInfo("America/New_York")


def _is_regular_trading_hours_et(ts: Any) -> bool:
    """
    True if ts falls within US regular trading hours 9:30–16:00 ET (inclusive 9:30, exclusive 16:00).
    ts can be datetime (naive treated as UTC) or numeric Unix timestamp.
    """
    if isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc).astimezone(_ET)
    elif isinstance(ts, datetime):
        dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(_ET)
    else:
        return True  # unknown type: assume regular hours
    return (dt.hour, dt.minute) >= (9, 30) and dt.hour < 16


# ============================================================
# Engine data structures
# ============================================================

@dataclass(frozen=True)
class FillIntent:
    """
    What the Simulator sends to the ExecutionEngine when it determines we'd get filled.
    Kalshi fills are applied immediately.
    """
    ts: Any
    contract_id: str
    side: str      # "buy" means our bid got hit; "sell" means our ask got lifted
    price: float
    size: int


@dataclass
class PendingTrade:
    """
    A trade that will be applied in the future (execution delay).
    Currently used for SPY hedge trades.
    """
    execute_ts: Any
    kind: str                    # "kalshi" or "spy"
    side: str                    # "buy" or "sell"
    qty: int
    price: float
    contract_id: Optional[str] = None


# ============================================================
# ExecutionEngine
# ============================================================

class ExecutionEngine:
    """
    Central gateway coordinating:
      - pricing (via Pricer)
      - quoting (via MarketMaker)
      - hedging (via DeltaHedger)
      - positions (via PositionManager)
      - trade execution with delay

    Expected usage (called by Simulator each second):
      out = engine.on_tick(ts=..., spx=..., vix=..., spy=..., contract_ids=[...])
      engine.on_fills([FillIntent(...), ...])   # optional, if any fills triggered
    """

    def __init__(
        self,
        *,
        market_maker,
        pricer: Optional[Pricer] = None,
        delta_hedger: Optional[DeltaHedger] = None,
        position_manager: Optional[PositionManager] = None,
        execution_delay_seconds: int = 1,
        default_quote_size: Optional[int] = None,
        quote_contracts: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Parameters
        ----------
        market_maker:
            Instance of your MarketMaker class (MarketMaker.py). Must expose:
              update_vix(vix), update_market_hours(in_regular_hours), update_fair_value(contract_id, fv),
              update_position(contract_id,...), make_quote(contract_id)
        pricer:
            Pricing module. Must expose: price(contract_id, spx, vix) -> fair value
        delta_hedger:
            Hedging module. Must expose:
              hedge(ts, spy_price, spx_price, vix, kalshi_positions, current_spy_position) -> HedgeOrder|None
        position_manager:
            Position tracking module.
        execution_delay_seconds:
            Delay applied to all trades.
        default_quote_size:
            If provided, overwrites MM's default sizes (optional convenience).
        quote_contracts:
            Optional fixed set of contracts to quote. If None, you should pass contract_ids each tick.
        """
        self.mm = market_maker
        self.pricer = pricer or Pricer()
        self.dh = delta_hedger or DeltaHedger()
        self.pm = position_manager or PositionManager()

        self.delay = int(execution_delay_seconds)
        self._pending: List[PendingTrade] = []
        self._last_ts: Any = None

        self._quote_contracts = list(quote_contracts) if quote_contracts is not None else None
        self._default_quote_size = default_quote_size

        # Latest outputs
        self._quotes_by_contract: Dict[str, Dict[str, Any]] = {}
        self._last_hedge_order: Optional[HedgeOrder] = None

    # --------------------------------------------------------
    # Main tick loop (called by Simulator)
    # --------------------------------------------------------

    def on_tick(
        self,
        *,
        ts: Any,
        spx: float,
        vix: float,
        spy: float,
        contract_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Called once per second by the Simulator.

        Responsibilities:
          1) apply any delayed trades due at ts (position updates)
          2) send SPX/VIX to Pricer -> fair values
          3) send VIX + fair values + positions to MarketMaker -> quotes
          4) send SPY price + positions to DeltaHedger -> hedge order (optional)
             and schedule it with execution delay

        Returns dict with:
          - quotes_by_contract: {contract_id: {bid, ask, bid_size, ask_size, fair_value}}
          - hedge_order: HedgeOrder|None
        """
        self._last_ts = ts

        # 1) apply delayed trades due now
        self._apply_pending(ts=ts, spy_price=spy)

        # 1b) settle expired Kalshi contracts (expiry settles at next-day start)
        self.pm.settle_expired_contracts(ts=ts, settlement_spx=float(spx))

        # 2) update MarketMaker's VIX and market-hours state (for out-of-market spread widening)
        self.mm.update_vix(vix)
        self.mm.update_market_hours(_is_regular_trading_hours_et(ts))

        # 3) determine contracts to quote
        if self._quote_contracts is not None:
            cids = self._quote_contracts
        elif contract_ids is not None:
            cids = list(contract_ids)
        else:
            raise ValueError("ExecutionEngine.on_tick needs contract_ids unless quote_contracts was set in __init__.")

        # 4) price + quote each contract
        quotes: Dict[str, Dict[str, Any]] = {}

        for cid in cids:
            try:
                fv = float(self.pricer.price(contract_id=cid, spx=float(spx), vix=float(vix), ts=ts))
            except TypeError:
                # Backward compatibility for custom Pricer implementations
                # that have not yet added the optional ts argument.
                fv = float(self.pricer.price(contract_id=cid, spx=float(spx), vix=float(vix)))
            # Pricer may return NaN for invalid/expired contracts at this timestamp.
            # Skip quoting those contracts instead of crashing inside MarketMaker.
            if not math.isfinite(fv):
                continue
            self.mm.update_fair_value(cid, fv)

            inv = self.pm.get_kalshi_position(cid)
            self.mm.update_position(cid, inventory=inv, cash=self.pm.get_cash(), initial_cash=self.pm.get_initial_cash())

            q = self.mm.make_quote(cid)

            bid_size = int(q.bid_size)
            ask_size = int(q.ask_size)

            # Optional override of sizes (nice for quick experiments)
            if self._default_quote_size is not None:
                bid_size = int(self._default_quote_size)
                ask_size = int(self._default_quote_size)

            quotes[cid] = {
                "contract_id": cid,
                "fair_value": fv,
                "bid": float(q.bid),
                "ask": float(q.ask),
                "bid_size": bid_size,
                "ask_size": ask_size,
            }

        self._quotes_by_contract = quotes

        # 5) hedge decision (SPY)
        spy_pos = self.pm.get_spy_position()
        hedge_order = self.dh.hedge(
            ts=ts,
            spy_price=float(spy),
            spx_price=float(spx),
            vix=float(vix),
            kalshi_positions=self.pm.get_kalshi_positions(),
            current_spy_position=int(spy_pos),
        )
        self._last_hedge_order = hedge_order

        if hedge_order is not None:
            if hedge_order.side == "buy" and self.pm.get_cash() <= 0:
                pass  # do not schedule SPY buy when we have no cash
            else:
                self._schedule_spy_trade(
                    decision_ts=ts,
                    side=hedge_order.side,
                    qty=int(hedge_order.qty),
                    ref_price=float(hedge_order.ref_price),
                )

        return {"quotes_by_contract": quotes, "hedge_order": hedge_order}

    # --------------------------------------------------------
    # Fill handling (called by Simulator when fills occur)
    # --------------------------------------------------------

    def on_fills(self, fills: Sequence[FillIntent]) -> None:
        """
        Apply Kalshi trades immediately from fill intents.

        The Simulator should compute fills using your queue assumption.
        Buys are not applied when cash <= 0 or when there is insufficient cash.
        Short sells (selling more than you own) require margin: cash >= 50% of notional.
        """
        for f in fills:
            if f.side == "buy":
                cost = int(f.size) * float(f.price) + PositionManager._maker_fee_dollars(
                    price=float(f.price), contracts=int(f.size)
                )
                if self.pm.get_cash() <= 0 or self.pm.get_cash() < cost:
                    continue
            elif f.side == "sell":
                pos = self.pm.get_kalshi_position(f.contract_id)
                if pos - int(f.size) < 0:
                    # short sell: need margin = 50% of notional
                    margin_required = 0.50 * (int(f.size) * float(f.price))
                    if self.pm.get_cash() < margin_required:
                        continue
            self.pm.apply_kalshi_trade(
                contract_id=f.contract_id,
                side=f.side,
                qty=int(f.size),
                price=float(f.price),
            )

    # --------------------------------------------------------
    # Observability helpers (useful for logging)
    # --------------------------------------------------------

    def get_quotes(self) -> Dict[str, Dict[str, Any]]:
        """Latest quotes produced by the engine."""
        return dict(self._quotes_by_contract)

    def get_last_hedge_order(self) -> Optional[HedgeOrder]:
        return self._last_hedge_order

    def snapshot_state(self, contract_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Small snapshot for the simulator to log.
        If contract_id is provided, include that contract's inventory.
        """
        out = {
            "pos_spy": self.pm.get_spy_position(),
            "cash": self.pm.get_cash(),
            "pending_trades": len(self._pending),
            "total_kalshi_inventory": self.pm.get_total_kalshi_inventory(),
        }
        if contract_id is not None:
            out["pos_kalshi"] = self.pm.get_kalshi_position(contract_id)
        return out

    def flush(self) -> None:
        """
        Apply all remaining pending trades in time order.
        Useful at the end of the simulation to settle remaining delayed trades.
        """
        if not self._pending:
            return

        try:
            self._pending.sort(key=lambda x: x.execute_ts)
        except Exception:
            # if timestamps aren't orderable, just process in insertion order
            pass

        while self._pending:
            pt = self._pending.pop(0)
            if pt.kind == "kalshi":
                assert pt.contract_id is not None
                self.pm.apply_kalshi_trade(contract_id=pt.contract_id, side=pt.side, qty=pt.qty, price=pt.price)
            elif pt.kind == "spy":
                self.pm.apply_spy_trade(side=pt.side, qty=pt.qty, price=pt.price)

    # --------------------------------------------------------
    # Internals: pending trades, scheduling, timestamp arithmetic
    # --------------------------------------------------------

    def _apply_pending(self, *, ts: Any, spy_price: float) -> None:
        """Apply all trades whose execute_ts is due at current ts."""
        if not self._pending:
            return

        remaining: List[PendingTrade] = []

        for pt in self._pending:
            if self._is_trade_due(execute_ts=pt.execute_ts, now_ts=ts):
                if pt.kind == "kalshi":
                    assert pt.contract_id is not None
                    self.pm.apply_kalshi_trade(contract_id=pt.contract_id, side=pt.side, qty=pt.qty, price=pt.price)
                elif pt.kind == "spy":
                    # do not execute SPY buy if we have no cash (or insufficient cash); drop the order
                    if pt.side == "buy" and (self.pm.get_cash() <= 0 or self.pm.get_cash() < pt.qty * float(spy_price)):
                        continue
                    # short sell margin: if selling more than we own, need cash >= 50% of notional
                    if pt.side == "sell":
                        spy_pos = self.pm.get_spy_position()
                        if spy_pos - pt.qty < 0:
                            margin_required = 0.50 * (pt.qty * float(spy_price))
                            if self.pm.get_cash() < margin_required:
                                continue
                    # mark hedge execution at the *current* spy_price at execution time
                    self.pm.apply_spy_trade(side=pt.side, qty=pt.qty, price=float(spy_price))
            else:
                remaining.append(pt)

        self._pending = remaining

    @staticmethod
    def _is_trade_due(*, execute_ts: Any, now_ts: Any) -> bool:
        """
        Return True if the trade should execute at now_ts.

        For orderable timestamps we use execute_ts <= now_ts so delayed trades
        still execute at the first available tick even when exact timestamps are sparse.
        If timestamps are not orderable/comparable, fall back to exact equality.
        """
        try:
            return execute_ts <= now_ts
        except Exception:
            return execute_ts == now_ts

    def _schedule_spy_trade(self, *, decision_ts: Any, side: str, qty: int, ref_price: float) -> None:
        execute_ts = self._ts_plus_seconds(decision_ts, self.delay)
        self._pending.append(PendingTrade(
            execute_ts=execute_ts,
            kind="spy",
            side=side,
            qty=int(qty),
            price=float(ref_price),
            contract_id=None,
        ))

    @staticmethod
    def _ts_plus_seconds(ts: Any, seconds: int) -> Any:
        """
        Supports:
          - int epoch seconds
          - python datetime-like (supports + timedelta)
        """
        if seconds == 0:
            return ts

        if isinstance(ts, int):
            return ts + seconds

        try:
            from datetime import timedelta
            return ts + timedelta(seconds=seconds)
        except Exception:
            # If we cannot shift time, fall back to no-delay behavior
            return ts