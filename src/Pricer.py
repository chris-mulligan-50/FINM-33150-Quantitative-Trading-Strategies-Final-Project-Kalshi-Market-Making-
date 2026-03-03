"""
Pricer.py

Minimal fair-value pricer for binary-style contracts.
"""

from __future__ import annotations

import math


class Pricer:
    """
    Quick baseline model:
    (SPX, VIX, contract_id) -> fair value in (0, 1).

    Replace this with your contract-specific pricing logic.
    """

    def __init__(self, *, spx_anchor: float = 5000.0, vix_anchor: float = 20.0) -> None:
        self.spx_anchor = float(spx_anchor)
        self.vix_anchor = float(vix_anchor)

    def price(self, *, contract_id: str, spx: float, vix: float) -> float:
        # Keep contract_id in signature for compatibility with multi-contract pricers.
        _ = contract_id
        z = 0.0001 * (float(spx) - self.spx_anchor) + 0.02 * (float(vix) - self.vix_anchor)
        fair_value = 1.0 / (1.0 + math.exp(-z))
        return float(min(0.999, max(0.001, fair_value)))
