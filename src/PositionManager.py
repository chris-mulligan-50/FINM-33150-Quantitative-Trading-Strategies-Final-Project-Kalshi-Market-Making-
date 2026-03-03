"""
PositionManager.py

Tracks simple Kalshi + SPY positions and cash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PositionSnapshot:
    kalshi: Dict[str, int] = field(default_factory=dict)
    spy: int = 0
    cash: float = 0.0


class PositionManager:
    def __init__(self) -> None:
        self._kalshi: Dict[str, int] = {}
        self._spy: int = 0
        self._cash: float = 0.0

    def apply_kalshi_trade(self, *, contract_id: str, side: str, qty: int, price: float) -> None:
        qty_i = int(qty)
        px_f = float(price)
        inv = self._kalshi.get(contract_id, 0)

        if side == "buy":
            inv += qty_i
            self._cash -= qty_i * px_f
        elif side == "sell":
            inv -= qty_i
            self._cash += qty_i * px_f
        else:
            raise ValueError(f"Unknown side={side!r}")

        self._kalshi[contract_id] = inv

    def apply_spy_trade(self, *, side: str, qty: int, price: float) -> None:
        qty_i = int(qty)
        px_f = float(price)

        if side == "buy":
            self._spy += qty_i
            self._cash -= qty_i * px_f
        elif side == "sell":
            self._spy -= qty_i
            self._cash += qty_i * px_f
        else:
            raise ValueError(f"Unknown side={side!r}")

    def get_kalshi_position(self, contract_id: str) -> int:
        return int(self._kalshi.get(contract_id, 0))

    def get_total_kalshi_inventory(self) -> int:
        return int(sum(self._kalshi.values()))

    def get_spy_position(self) -> int:
        return int(self._spy)

    def get_cash(self) -> float:
        return float(self._cash)

    def snapshot(self) -> PositionSnapshot:
        return PositionSnapshot(kalshi=dict(self._kalshi), spy=int(self._spy), cash=float(self._cash))
