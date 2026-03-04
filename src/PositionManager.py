"""
PositionManager.py

Tracks simple Kalshi + SPY positions and cash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
import re
from typing import Any, Dict, Optional


@dataclass
class PositionSnapshot:
    kalshi: Dict[str, int] = field(default_factory=dict)
    spy: int = 0
    cash: float = 0.0


class PositionManager:
    _THRESHOLD_RE = re.compile(r"-T(?P<k>\d+(?:\.\d+)?)$")
    _MIDPOINT_RE = re.compile(r"-B(?P<k>\d+(?:\.\d+)?)$")
    _EXPIRY_RE = re.compile(r"-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})H(?P<hh>\d{2})(?P<mm>\d{2})")
    _MONTHS = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12,
    }

    def __init__(self, *, initial_cash: float = 10_000.0) -> None:
        self._kalshi: Dict[str, int] = {}
        self._spy: int = 0
        self._cash: float = float(initial_cash)
        self._initial_cash: float = float(initial_cash)

    def apply_kalshi_trade(self, *, contract_id: str, side: str, qty: int, price: float) -> None:
        qty_i = int(qty)
        px_f = float(price)
        inv = self._kalshi.get(contract_id, 0)

        # Maker fee when our resting order is taken: round up(0.0175 * C * P * (1 - P)) to next cent
        fee = self._maker_fee_dollars(price=px_f, contracts=qty_i)

        if side == "buy":
            inv += qty_i
            self._cash -= qty_i * px_f + fee
        elif side == "sell":
            inv -= qty_i
            self._cash += qty_i * px_f - fee
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

    def get_kalshi_positions(self) -> Dict[str, int]:
        return dict(self._kalshi)

    def get_total_kalshi_inventory(self) -> int:
        return int(sum(self._kalshi.values()))

    def get_spy_position(self) -> int:
        return int(self._spy)

    def get_cash(self) -> float:
        return float(self._cash)

    def get_initial_cash(self) -> float:
        return float(self._initial_cash)

    @staticmethod
    def _maker_fee_dollars(*, price: float, contracts: int) -> float:
        """
        Maker fee when our resting order is taken.
        fee = round up(0.0175 * C * P * (1 - P)) to the next cent.
        """
        c = max(0, int(contracts))
        p = float(price)
        raw = 0.0175 * c * p * (1.0 - p)
        return math.ceil(raw * 100.0) / 100.0

    def settle_expired_contracts(self, *, ts: Any, settlement_spx: float) -> Dict[str, float]:
        """
        Settle contracts that expire at the start of the next day after contract date.

        For each settled contract:
          cash += qty * payout, where payout is 0 or 1.
          position is removed from the Kalshi book.

        Returns
        -------
        dict with aggregate settlement diagnostics:
          settled_count, cash_delta
        """
        now_dt = self._to_datetime(ts)
        if now_dt is None:
            return {"settled_count": 0.0, "cash_delta": 0.0}

        cash_delta = 0.0
        settled_count = 0

        for contract_id, qty in list(self._kalshi.items()):
            expiry = self._parse_expiry(contract_id)
            if expiry is None:
                continue

            settle_ts = datetime(expiry.year, expiry.month, expiry.day) + timedelta(days=1)
            if now_dt < settle_ts:
                continue

            payout = self._contract_payout(contract_id=contract_id, spx=float(settlement_spx))
            if payout is None:
                continue

            q = int(qty)
            delta_cash = q * payout
            self._cash += delta_cash
            cash_delta += delta_cash
            settled_count += 1
            del self._kalshi[contract_id]

        return {"settled_count": float(settled_count), "cash_delta": float(cash_delta)}

    def snapshot(self) -> PositionSnapshot:
        return PositionSnapshot(kalshi=dict(self._kalshi), spy=int(self._spy), cash=float(self._cash))

    @classmethod
    def _parse_threshold(cls, contract_id: str) -> Optional[float]:
        m = cls._THRESHOLD_RE.search(contract_id)
        return float(m.group("k")) if m else None

    @classmethod
    def _parse_midpoint(cls, contract_id: str) -> Optional[float]:
        m = cls._MIDPOINT_RE.search(contract_id)
        return float(m.group("k")) if m else None

    @classmethod
    def _parse_expiry(cls, contract_id: str) -> Optional[datetime]:
        m = cls._EXPIRY_RE.search(contract_id)
        if not m:
            return None

        yy = int(m.group("yy"))
        month = cls._MONTHS.get(m.group("mon"))
        dd = int(m.group("dd"))
        hh = int(m.group("hh"))
        mm = int(m.group("mm"))
        if month is None:
            return None

        year = 2000 + yy
        try:
            return datetime(year, month, dd, hh, mm)
        except ValueError:
            return None

    @staticmethod
    def _is_no_contract(contract_id: str) -> bool:
        cid = contract_id.upper()
        return any(token in cid for token in (" NO ", "NO ABOVE", "NO_ABOVE", "NO-ABOVE", ":NO", "-NO"))

    @classmethod
    def _contract_payout(cls, *, contract_id: str, spx: float) -> Optional[float]:
        threshold = cls._parse_threshold(contract_id)
        if threshold is not None:
            yes_outcome = 1.0 if float(spx) > threshold else 0.0
        else:
            midpoint = cls._parse_midpoint(contract_id)
            if midpoint is None:
                return None

            # Matches call-spread interpretation used by the pricer:
            # YES range pays 1 when lower < SPX <= upper.
            midpoint_adj = midpoint + 0.5
            half_width = 12.5
            lower = midpoint_adj - half_width
            upper = midpoint_adj + half_width
            yes_outcome = 1.0 if (float(spx) > lower and float(spx) <= upper) else 0.0

        if cls._is_no_contract(contract_id):
            return 1.0 - yes_outcome
        return yes_outcome

    @staticmethod
    def _to_datetime(ts: Any) -> Optional[datetime]:
        if isinstance(ts, datetime):
            return ts.replace(tzinfo=None)
        if isinstance(ts, (int, float)):
            sec = float(ts)
            if sec > 1e15:
                sec = sec / 1e9  # nanoseconds -> seconds
            elif sec > 1e12:
                sec = sec / 1000.0  # milliseconds -> seconds
            try:
                return datetime.utcfromtimestamp(sec).replace(tzinfo=None)
            except (ValueError, OSError):
                return None
        return None
