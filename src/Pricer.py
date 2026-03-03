"""
Pricer.py

Fair-value pricer for Kalshi binary-style SPX contracts.
"""

from __future__ import annotations

from datetime import datetime
import math
import re
from typing import Any, Optional


class Pricer:
    """
    Binary pricing logic using lognormal terminal SPX distribution:
      P(S_T > K) = Phi(d2)
      d2 = (ln(F/K) - 0.5*sigma^2*tau) / (sigma*sqrt(tau))

    Model/implementation notes:
      - F ~= S_t (negligible discount at 0-1 DTE).
      - sigma is taken from VIX/100 (flat vol assumption).
      - KXINXU contracts are binary threshold ("above") contracts.
      - KXINX contracts are binary range contracts priced as call spreads.
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
        default_sigma: float = 0.20,
        fallback_tau_years: float = 1.0 / 365.0,
        min_tau_years: float = 1e-6,
        kxinx_bucket_width: float = 25.0,
        kxinx_midpoint_offset: float = 0.5,
        prob_clip_eps: float = 1e-6,
    ) -> None:
        self.default_sigma = float(default_sigma)
        self.fallback_tau_years = float(fallback_tau_years)
        self.min_tau_years = float(min_tau_years)
        self.kxinx_bucket_width = float(kxinx_bucket_width)
        self.kxinx_midpoint_offset = float(kxinx_midpoint_offset)
        self.prob_clip_eps = float(prob_clip_eps)

    def price(self, *, contract_id: str, spx: float, vix: float, ts: Any = None) -> float:
        spx_f = float(spx)
        if spx_f <= 0.0:
            return 0.5

        sigma = self._sigma_from_vix(vix)
        tau = self._time_to_expiry_years(ts=ts, contract_id=contract_id)
        tau = max(self.min_tau_years, tau)

        yes_prob = self._yes_probability(
            contract_id=contract_id,
            spx=spx_f,
            sigma=sigma,
            tau=tau,
        )
        if yes_prob is None:
            yes_prob = 0.5

        if self._is_no_contract(contract_id):
            fair_value = 1.0 - yes_prob
        else:
            fair_value = yes_prob

        lo = self.prob_clip_eps
        hi = 1.0 - self.prob_clip_eps
        return float(min(hi, max(lo, fair_value)))

    def _yes_probability(self, *, contract_id: str, spx: float, sigma: float, tau: float) -> Optional[float]:
        threshold = self._parse_threshold(contract_id)
        if threshold is not None:
            return self._prob_above(spx=spx, strike=threshold, sigma=sigma, tau=tau)

        midpoint = self._parse_midpoint(contract_id)
        if midpoint is None:
            return None

        lower, upper = self._range_strikes_from_midpoint(midpoint)
        p_above_lower = self._prob_above(spx=spx, strike=lower, sigma=sigma, tau=tau)
        p_above_upper = self._prob_above(spx=spx, strike=upper, sigma=sigma, tau=tau)
        return max(0.0, min(1.0, p_above_lower - p_above_upper))

    def _prob_above(self, *, spx: float, strike: float, sigma: float, tau: float) -> float:
        if strike <= 0.0 or sigma <= 0.0 or tau <= 0.0:
            return 0.5
        vol_t = sigma * math.sqrt(tau)
        if vol_t <= 0.0:
            return 0.5
        d2 = (math.log(spx / strike) - 0.5 * sigma * sigma * tau) / vol_t
        return self._norm_cdf(d2)

    def _range_strikes_from_midpoint(self, midpoint_token: float) -> tuple[float, float]:
        midpoint = float(midpoint_token) + self.kxinx_midpoint_offset
        half_width = 0.5 * self.kxinx_bucket_width
        return midpoint - half_width, midpoint + half_width

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
    def _to_datetime(ts: Any) -> Optional[datetime]:
        if isinstance(ts, datetime):
            return ts.replace(tzinfo=None)
        return None

    @staticmethod
    def _is_no_contract(contract_id: str) -> bool:
        cid = contract_id.upper()
        return any(token in cid for token in (" NO ", "NO ABOVE", "NO_ABOVE", "NO-ABOVE", ":NO", "-NO"))

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
