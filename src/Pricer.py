"""
Pricer.py

Fair-value pricer for Kalshi binary-style SPX contracts.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import math
import re
from typing import Any, Optional
import numpy as np


class Pricer:
    """
    Always returns a value in [0.0, 1.0] representing the fair value of the "YES" side of a given Kalshi SPX binary contract, 
    based on the current SPX price, VIX, and time to expiry.

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
        kxinx_bucket_width: float = 25.0, # observed from KXINX contract IDs, e.g. "KXINX-26FEB26H1600-B6987" corresponds to a bucket of [6975.0, 7000.0) for a bucket width of 25 and midpoint offset of 0.5
        kxinx_midpoint_offset: float = 0.5,
    ) -> None:
        self.kxinx_bucket_width = float(kxinx_bucket_width)
        self.kxinx_midpoint_offset = float(kxinx_midpoint_offset)

    def price(self, *, contract_id: str, spx: float, vix: float, ts: Any = None) -> float:
        """
        Returns a fair value in [0.0, 1.0] for the "YES" side given contract, SPX price, VIX, and current timestamp.
        If the contract_id cannot be parsed, or if any of the parameters are invalid, return np.nan for safety.

        - contract_id: the Kalshi contract ID string, e.g. "KXINX-26FEB26H1600-B6987" or "KXINXU-26JAN02H1600-T6799.9999"
        - spx: current SPX index level, e.g. 4500.0
        - vix: current VIX index level, e.g. 20.0
        - ts: current timestamp, either as a datetime or a UNIX timestamp (seconds since epoch) NOTE: MUST BE IN UTC
        """

        spx_f = float(spx)
        if spx_f <= 0.0:
            return np.nan

        sigma = self._sigma_from_vix(vix)
        if math.isnan(sigma):
            return np.nan
        
        tau = self._time_to_expiry_years(ts=ts, contract_id=contract_id)
        if math.isnan(tau):
            return np.nan
        

        # figure out whether our contract is a above/below or range contract
        contract_type = self._parse_contract_type(contract_id)
        if contract_type is None:
            return np.nan
        
        fair_value = np.nan
        
        if contract_type == "threshold":
            # get the strike from the contract ID, and price as a single binary option
            strike = self._parse_threshold(contract_id)
            if strike is None:
                return np.nan
            
            # calculate the current time "YES" probability of a binary call with strike = threshold and T - t = time_to_expiry years
            prob_above = self._prob_above(spx=spx_f, strike=strike, sigma=sigma, tau=tau) 

            # since discounting is negligible at 0-1 DTE, we can interpret the "YES" probability as the fair value of the binary option. If it's a "NO ABOVE" contract, then we take 1 - prob_above as the fair value.
            fair_value = prob_above


        elif contract_type == "range":
            midpoint = self._parse_midpoint(contract_id)
            if midpoint is None:
                return np.nan
                        
            below, above = self._range_strikes_from_midpoint(midpoint)
            if below <= 0.0 or above <= 0.0:
                return np.nan
            
            # calculate the current time "YES" probability of a binary call spread with strikes = (below, above) and T - t = time_to_expiry years. 
            """
            We can replicate "YES" KXINX with below and above by going long one binary call with strike = below and short one binary call with strike = above. 
            
            The price of long one binary call with strike = below is the probability that SPX finishes above the "below" strike, which is prob_above_lower = P(S_T > below) = Phi(d2_below).
            The price of short one binary call with strike = above is -1 times the probability that SPX finishes above the "above" strike, which is prob_above_upper = P(S_T > above) = Phi(d2_above).

            Hence, the price of the call spread is prob_above_lower - prob_above_upper, which corresponds to the probability that SPX finishes between the two strikes 
            (since if SPX finishes above the upper strike, both binaries are in the money and their prices cancel out, and if SPX finishes below the lower strike, both binaries are out of the money and worth zero).
            """

            prob_above_lower = self._prob_above(spx=spx_f, strike=below, sigma=sigma, tau=tau)
            prob_above_upper = self._prob_above(spx=spx_f, strike=above, sigma=sigma, tau=tau)
            fair_value = max(0.0, min(1.0, prob_above_lower - prob_above_upper))

        return fair_value

    def _prob_above(self, *, spx: float, strike: float, sigma: float, tau: float) -> float:
        """
        See slide 8 of pitchbook draft
        """

        d2_denom = sigma * math.sqrt(tau)
        d2 = (math.log(spx / strike) - 0.5 * sigma * sigma * tau) / d2_denom
        return self._norm_cdf(d2)

    def _range_strikes_from_midpoint(self, midpoint_token: float) -> tuple[float, float]:
        """
        Given a midpoint token, return the lower and upper strikes for the corresponding range contract.
        For example, if the midpoint token is 6987.0, the bucket width is 25.0, and the midpoint offset is 0.5, 
        then we would return (6975.0, 7000.0) since the midpoint corresponds to a bucket of [6975.0, 7000.0).
        """
        midpoint = float(midpoint_token) + self.kxinx_midpoint_offset
        half_width = 0.5 * self.kxinx_bucket_width
        return midpoint - half_width, midpoint + half_width

    def _time_to_expiry_years(self, *, ts: Any, contract_id: str) -> float:
        """
        Return the time to expiry in years as a float, given the current timestamp and the contract ID.
        If the expiry cannot be parsed, or if the timestamp is invalid, return np.nan for safety.

        If given no timestamp, we use current UTC time.
        """
        
        expiry = self._parse_expiry(contract_id)
        if expiry is None:
            return np.nan

        now_dt = self._to_datetime(ts)
        if now_dt is None:
            now_dt = datetime.now(datetime.timezone.utc).replace(tzinfo=None)

        seconds = (expiry - now_dt).total_seconds()
        if seconds <= 0.0:
            return np.nan
        return seconds / (365.0 * 24.0 * 60.0 * 60.0)

    def _sigma_from_vix(self, vix: Optional[float]) -> float:
        """
        VIX is 30 day annualized volatility in percentage points, so divide by 100 to get a decimal, and use as flat vol.
        Return np.nan if vix is None, and clip to a minimum of 1e-6 to avoid zero or negative vol. 
        Note that VIX can be zero or near-zero in very low-vol environments, but Kalshi contracts should still have some positive value due to the possibility of price moves and time decay.
        """

        if vix is None:
            return np.nan
        
        return max(1e-6, float(vix) / 100.0)
    
    @classmethod
    def _parse_contract_type(cls, contract_id: str) -> Optional[str]:
        cid = contract_id.upper()
        if "KXINXU" in cid:
            return "threshold"
        elif "KXINX" in cid:
            return "range"
        else:
            return None

    @classmethod
    def _parse_threshold(cls, contract_id: str) -> Optional[float]:
        """
        Parse the strike threshold K from a contract ID like "KXINXU-26JAN02H1600-T6799.9999", returning it as a float.
        In the above example, we would return 6799.9999 as a float.
        If the contract ID does not match the expected format, return None.
        """
        m = cls._THRESHOLD_RE.search(contract_id)
        return float(m.group("k")) if m else None

    @classmethod
    def _parse_midpoint(cls, contract_id: str) -> Optional[float]:
        """
        Parse the strike midpoint K from a contract ID like "KXINX-26FEB26H1600-B6987", returning it as a float.
        In the above example, we would return 6987.0 as a float. This corresponds to a range of [6975.0, 7000.0) for a bucket width of 25 and midpoint offset of 0.5.
        If the contract ID does not match the expected format, return None.
        """
        m = cls._MIDPOINT_RE.search(contract_id)
        return float(m.group("k")) if m else None

    @classmethod
    def _parse_expiry(cls, contract_id: str) -> Optional[datetime]:
        """
        For example, for KXINX-26FEB26H1600-B6987:
        - yy = 26
        - mon = FEB -> 2
        - dd = 26
        - hh = 16
        - mm = 00

        For KXINXU-26JAN02H1600-T6799.9999:
        - yy = 26
        - mon = JAN -> 1
        - dd = 02
        - hh = 16
        - mm = 00

        We would then return a datetime object representing 2026-02-26 16:00:00 for the first example, and 2026-01-02 16:00:00 for the second example.
        If the contract ID does not match the expected format, or if the date components are invalid, return None.

        The time is in ET, so we convert to UTC
        """
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
            dt = datetime(year, month, dd, hh, mm)
            # convert from ET to UTC by adding 5 hours (note: this is not a full timezone-aware conversion, but should be sufficient for our purposes since Kalshi contracts expire at 16:00 ET which is 21:00 UTC) 
            dt += timedelta(hours=5)
            return dt
        except ValueError:
            return None

    @staticmethod
    def _to_datetime(ts: Any) -> Optional[datetime]:
        """
        Convert a timestamp to a datetime object. If the input is already a datetime, return it as is (after removing tzinfo). If the input is a UNIX timestamp (int or float), convert it to a datetime. If the input is invalid, return None.
        """
        if isinstance(ts, datetime):
            return ts.replace(tzinfo=None)
        return None

    @staticmethod
    def _norm_cdf(x: float) -> float:
        """
        Standard normal CDF evaluated at x, using math.erf. Returns a value in [0.0, 1.0].
        """
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
