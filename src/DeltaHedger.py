"""
DeltaHedger.py

Minimal hedge policy that maps aggregate Kalshi inventory to SPY orders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class HedgeOrder:
    ts: Any
    symbol: str
    side: str
    qty: int
    ref_price: float


class DeltaHedger:
    """
    Quick baseline hedge logic.

    Target SPY position = -k * (total Kalshi inventory), rounded to int.
    """

    def __init__(self, *, k: float = 1.0, max_qty_per_tick: int = 50, symbol: str = "SPY") -> None:
        self.k = float(k)
        self.max_qty_per_tick = int(max_qty_per_tick)
        self.symbol = str(symbol)

    def hedge(
        self,
        *,
        ts: Any,
        spy_price: float,
        total_kalshi_inventory: int,
        current_spy_position: int,
    ) -> Optional[HedgeOrder]:
        target_spy_position = int(round(-self.k * int(total_kalshi_inventory)))
        diff = target_spy_position - int(current_spy_position)
        if diff == 0:
            return None

        qty = min(abs(diff), self.max_qty_per_tick)
        side = "buy" if diff > 0 else "sell"

        return HedgeOrder(
            ts=ts,
            symbol=self.symbol,
            side=side,
            qty=int(qty),
            ref_price=float(spy_price),
        )
