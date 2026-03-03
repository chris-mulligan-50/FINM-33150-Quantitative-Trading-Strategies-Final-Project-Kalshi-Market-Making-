"""
DeltaHedger.py

Delta hedge Kalshi directional SPX contracts using SPY.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import re
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class HedgeOrder:
    ts: Any
    symbol: str
    side: str
    qty: int
    ref_price: float


class DeltaHedger:
    """
    Hedge logic from binary option replication:
      Delta_yes_above(K) = phi(d2) / (S * sigma * sqrt(tau))
      Delta_no_above(K)  = -Delta_yes_above(K)

    Aggregate book delta in SPX points:
      Delta_book_spx = sum_i q_i * Delta_i

    Map SPX points to SPY dollars with:
      dSPY/dSPX ~= 0.1  -> shares = -(Delta_book_spx / 0.1) = -10 * Delta_book_spx
    """

    _THRESHOLD_RE = re.compile(r"-T(?P<k>\d+(?:\.\d+)?)$")
    _MIDPOINT_RE = re.compile(r"-B(?P<k>\d+(?:\.\d+)?)$")
    _EXPIRY_RE = re.compile(
        r"-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})H(?P<hh>\d{2})(?P<mm>\d{2})"
    )
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

    def __init__(
        self,
        *,
        spx_to_spy_beta: float = 0.1,
        default_sigma: float = 0.20,
        fallback_tau_years: float = 1.0 / 252.0,
        min_tau_years: float = 1e-6,
        max_qty_per_tick: int = 50,
        symbol: str = "SPY",
    ) -> None:
        self.spx_to_spy_beta = float(spx_to_spy_beta)
        self.default_sigma = float(default_sigma)
        self.fallback_tau_years = float(fallback_tau_years)
        self.min_tau_years = float(min_tau_years)
        self.max_qty_per_tick = int(max_qty_per_tick)
        self.symbol = str(symbol)

    def hedge(
        self,
        *,
        ts: Any,
        spy_price: float,
        spx_price: float,
        vix: Optional[float],
        kalshi_positions: Dict[str, int],
        current_spy_position: int,
    ) -> Optional[HedgeOrder]:
        _ = float(spy_price)  # kept for interface consistency / logging
        if self.spx_to_spy_beta == 0.0:
            return None

        sigma = self._sigma_from_vix(vix)
        delta_book_spx = self._book_delta_spx(
            ts=ts,
            spx_price=float(spx_price),
            sigma=sigma,
            kalshi_positions=kalshi_positions,
        )
        target_spy_position = int(round(-(delta_book_spx / self.spx_to_spy_beta)))
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

    def _book_delta_spx(
        self,
        *,
        ts: Any,
        spx_price: float,
        sigma: float,
        kalshi_positions: Dict[str, int],
    ) -> float:
        total = 0.0
        for contract_id, qty in kalshi_positions.items():
            q = int(qty)
            if q == 0:
                continue
            delta_i = self._contract_delta_spx(
                ts=ts,
                contract_id=contract_id,
                spx_price=spx_price,
                sigma=sigma,
            )
            total += q * delta_i
        return total

    def _contract_delta_spx(
        self,
        *,
        ts: Any,
        contract_id: str,
        spx_price: float,
        sigma: float,
    ) -> float:
        strike = self._parse_strike(contract_id)
        if strike is None or strike <= 0.0 or spx_price <= 0.0 or sigma <= 0.0:
            return 0.0

        tau = self._time_to_expiry_years(ts=ts, contract_id=contract_id)
        tau = max(self.min_tau_years, tau)

        vol_t = sigma * math.sqrt(tau)
        if vol_t <= 0.0:
            return 0.0

        d2 = (math.log(spx_price / strike) - 0.5 * sigma * sigma * tau) / vol_t
        phi_d2 = math.exp(-0.5 * d2 * d2) / math.sqrt(2.0 * math.pi)
        delta_yes_above = phi_d2 / (spx_price * vol_t)

        if self._is_no_above(contract_id):
            return -delta_yes_above
        return delta_yes_above

    def _time_to_expiry_years(self, *, ts: Any, contract_id: str) -> float:
        expiry = self._parse_expiry(contract_id)
        if expiry is None:
            return self.fallback_tau_years

        now_dt = self._to_datetime(ts)
        if now_dt is None:
            return self.fallback_tau_years

        seconds = (expiry - now_dt).total_seconds()
        if seconds <= 0.0:
            return self.min_tau_years
        return seconds / (365.0 * 24.0 * 60.0 * 60.0)

    def _sigma_from_vix(self, vix: Optional[float]) -> float:
        if vix is None:
            return self.default_sigma
        return max(1e-6, float(vix) / 100.0)

    @classmethod
    def _parse_strike(cls, contract_id: str) -> Optional[float]:
        m = cls._THRESHOLD_RE.search(contract_id)
        if m:
            return float(m.group("k"))
        m = cls._MIDPOINT_RE.search(contract_id)
        if m:
            return float(m.group("k"))
        return None

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
    def _to_datetime(ts: Any) -> Optional[datetime]:
        if isinstance(ts, datetime):
            return ts.replace(tzinfo=None)
        return None

    @staticmethod
    def _is_no_above(contract_id: str) -> bool:
        cid = contract_id.upper()
        return any(token in cid for token in ("NO ABOVE", "NO_ABOVE", "NO-ABOVE", ":NO", "-NO"))


class NoHedgeDeltaHedger:
    """
    No-op hedger that never submits SPY hedge orders.
    Use this to run the strategy without any delta hedging (e.g. to compare PnL).
    Same interface as DeltaHedger: hedge(...) -> Optional[HedgeOrder], always returns None.
    """

    def hedge(
        self,
        *,
        ts: Any,
        spy_price: float,
        spx_price: float,
        vix: Optional[float],
        kalshi_positions: Dict[str, int],
        current_spy_position: int,
    ) -> Optional[HedgeOrder]:
        return None
