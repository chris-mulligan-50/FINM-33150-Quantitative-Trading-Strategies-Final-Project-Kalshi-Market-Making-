"""
MarketMaker.py

A simple, extensible Market Maker module for a Kalshi market-making system.

Responsibilities
- Ingest *state updates* from other components:
  1) Fair value updates from Pricer (per contract)
  2) VIX updates from ExecutionEngine (or other vol proxy)
  3) Position updates from PositionManager (inventory, cash, risk limits)
- Produce a bid/ask quote (and optionally sizes) for a given contract.

Notes
- This class is intentionally lightweight and dependency-free.
- Replace/extend the quoting logic with your preferred model:
  - inventory-risk, adverse selection, queue position, latency, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class Quote:
    """A quoting decision for a single contract."""
    contract_id: str
    fair_value: float
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    spread: float
    meta: Dict[str, float] = field(default_factory=dict)


@dataclass
class PositionState:
    """
    Minimal position state. Expand as needed (PnL, greeks, per-leg breakdown, etc.).

    inventory: signed position in the *contract* (positive = long, negative = short)
    cash: optional cash balance (if you track it)
    """
    inventory: int = 0
    cash: float = 0.0


class MarketMaker:
    """
    MarketMaker takes:
      - fair values from Pricer
      - VIX from ExecutionEngine (vol/risk proxy)
      - position updates from PositionManager
    and returns quotes (bid/ask + sizes).

    Typical usage pattern
      mm.update_fair_value(contract_id, fv)
      mm.update_vix(vix)
      mm.update_position(contract_id, inventory=..., cash=...)
      quote = mm.make_quote(contract_id)
    """

    def __init__(
        self,
        *,
        tick_size: float = 0.01,
        base_spread: float = 0.02,
        vix_spread_slope: float = 0.001,
        inventory_spread_slope: float = 0.001,
        inventory_skew_slope: float = 0.0005,
        max_spread: float = 0.50,
        min_spread: float = 0.01,
        default_quote_size: int = 1,
        max_quote_size: int = 25,
        max_abs_inventory_for_size: int = 50,
        clamp_prices_to_unit_interval: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        tick_size:
            Minimum price increment for the contract (e.g., 0.01 for $0.01 ticks).
        base_spread:
            Baseline spread around fair value (in *price units*).
        vix_spread_slope:
            How much spread widens per 1 VIX point.
        inventory_spread_slope:
            How much spread widens per 1 contract of absolute inventory.
        inventory_skew_slope:
            How much we skew mid away from fair value based on inventory (to mean-revert inventory).
        max_spread / min_spread:
            Hard clamps for produced spread.
        default_quote_size:
            Default number of contracts to quote on each side.
        max_quote_size:
            Upper bound on quote size.
        max_abs_inventory_for_size:
            Inventory level at which size is reduced to 0 (softly) by default sizing logic.
        clamp_prices_to_unit_interval:
            Many Kalshi binary contracts trade in [0,1]. If True, clamp bid/ask to [0,1].
        """
        self.tick_size = float(tick_size)
        self.base_spread = float(base_spread)
        self.vix_spread_slope = float(vix_spread_slope)
        self.inventory_spread_slope = float(inventory_spread_slope)
        self.inventory_skew_slope = float(inventory_skew_slope)
        self.max_spread = float(max_spread)
        self.min_spread = float(min_spread)
        self.default_quote_size = int(default_quote_size)
        self.max_quote_size = int(max_quote_size)
        self.max_abs_inventory_for_size = int(max_abs_inventory_for_size)
        self.clamp_prices_to_unit_interval = bool(clamp_prices_to_unit_interval)

        # State
        self._fair_values: Dict[str, float] = {}
        self._positions: Dict[str, PositionState] = {}
        self._vix: Optional[float] = None

    # -------------------------
    # State update methods
    # -------------------------

    def update_fair_value(self, contract_id: str, fair_value: float) -> None:
        """Update the latest fair value for a contract (from Pricer)."""
        self._fair_values[contract_id] = float(fair_value)

    def update_vix(self, vix: float) -> None:
        """Update the latest VIX value (from ExecutionEngine / market data)."""
        self._vix = float(vix)

    def update_position(
        self,
        contract_id: str,
        *,
        inventory: Optional[int] = None,
        cash: Optional[float] = None,
    ) -> None:
        """Update position state for a contract (from PositionManager)."""
        ps = self._positions.get(contract_id, PositionState())
        if inventory is not None:
            ps.inventory = int(inventory)
        if cash is not None:
            ps.cash = float(cash)
        self._positions[contract_id] = ps

    def bulk_update_positions(self, updates: Dict[str, PositionState]) -> None:
        """Replace/merge positions in bulk (useful if PositionManager sends snapshots)."""
        for cid, ps in updates.items():
            self._positions[cid] = ps

    # -------------------------
    # Quote generation
    # -------------------------

    def make_quote(self, contract_id: str) -> Quote:
        """
        Compute bid/ask for a contract based on current state.

        Returns
        -------
        Quote
        """
        if contract_id not in self._fair_values:
            raise KeyError(f"No fair value available for contract_id={contract_id!r}. Call update_fair_value first.")

        fv = self._fair_values[contract_id]
        pos = self._positions.get(contract_id, PositionState())
        vix = self._vix if self._vix is not None else 0.0

        # 1) Spread model: base + vix_widen + inv_widen
        spread = (
            self.base_spread
            + self.vix_spread_slope * max(vix, 0.0)
            + self.inventory_spread_slope * abs(pos.inventory)
        )
        spread = self._clamp(spread, self.min_spread, self.max_spread)

        # 2) Inventory skew: shift mid away from fv so we quote more aggressively to reduce inventory
        #    If long inventory, we skew mid DOWN -> more likely to sell (ask closer, bid further)
        #    If short inventory, skew mid UP -> more likely to buy (bid closer, ask further)
        mid_skew = -self.inventory_skew_slope * pos.inventory
        mid = fv + mid_skew

        # 3) Convert mid +/- half-spread to bid/ask
        half = 0.5 * spread
        raw_bid = mid - half
        raw_ask = mid + half

        bid = self._round_down(raw_bid, self.tick_size)
        ask = self._round_up(raw_ask, self.tick_size)

        # Ensure no crossed market due to rounding
        if bid >= ask:
            # push ask one tick above bid
            ask = bid + self.tick_size

        if self.clamp_prices_to_unit_interval:
            bid = self._clamp(bid, 0.0, 1.0)
            ask = self._clamp(ask, 0.0, 1.0)
            if bid >= ask:
                # if clamping caused crossing, widen minimally
                bid = self._clamp(bid - self.tick_size, 0.0, 1.0)
                ask = self._clamp(bid + self.tick_size, 0.0, 1.0)

        # 4) Size model: reduce size as inventory grows (simple default)
        base_sz = self.default_quote_size
        inv_penalty = abs(pos.inventory) / max(1, self.max_abs_inventory_for_size)
        sz = int(round(base_sz * max(0.0, 1.0 - inv_penalty)))
        sz = self._clamp_int(sz, 0, self.max_quote_size)

        # Optional asymmetric sizing (tilt sizes to reduce inventory faster)
        bid_size, ask_size = self._inventory_tilt_sizes(sz, pos.inventory)

        return Quote(
            contract_id=contract_id,
            fair_value=fv,
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            spread=(ask - bid),
            meta={
                "vix": float(vix),
                "inventory": float(pos.inventory),
                "mid": float(mid),
                "mid_skew": float(mid_skew),
                "model_spread": float(spread),
            },
        )

    # -------------------------
    # Helpers
    # -------------------------

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def _clamp_int(x: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, x))

    @staticmethod
    def _round_down(x: float, tick: float) -> float:
        if tick <= 0:
            return x
        return (int(x / tick)) * tick

    @staticmethod
    def _round_up(x: float, tick: float) -> float:
        if tick <= 0:
            return x
        q = int(x / tick)
        if q * tick < x:
            q += 1
        return q * tick

    @staticmethod
    def _inventory_tilt_sizes(base_size: int, inventory: int) -> Tuple[int, int]:
        """
        Simple asymmetric size tilt:
        - If long, quote bigger ask (sell more), smaller bid.
        - If short, quote bigger bid (buy more), smaller ask.
        """
        if base_size <= 0:
            return 0, 0

        if inventory > 0:
            bid_sz = max(0, base_size - 1)
            ask_sz = base_size + 1
        elif inventory < 0:
            bid_sz = base_size + 1
            ask_sz = max(0, base_size - 1)
        else:
            bid_sz = base_size
            ask_sz = base_size

        return bid_sz, ask_sz


if __name__ == "__main__":
    # Quick sanity check / example
    mm = MarketMaker(tick_size=0.01, base_spread=0.02)
    cid = "KALSHI_CONTRACT_ABC"

    mm.update_fair_value(cid, 0.62)
    mm.update_vix(18.5)
    mm.update_position(cid, inventory=10, cash=1000.0)

    q = mm.make_quote(cid)
    print(q)
