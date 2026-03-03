from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Pricer import Pricer


class PricerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pricer = Pricer()
        self.ts = datetime(2026, 1, 23, 15, 0)
        self.spx = 6900.0
        self.vix = 20.0

    def test_kxinxu_yes_no_are_complements(self) -> None:
        cid_yes = "KXINXU-26JAN23H1600-T6874.9999"
        cid_no = "KXINXU:NO-26JAN23H1600-T6874.9999"

        p_yes = self.pricer.price(contract_id=cid_yes, spx=self.spx, vix=self.vix, ts=self.ts)
        p_no = self.pricer.price(contract_id=cid_no, spx=self.spx, vix=self.vix, ts=self.ts)

        self.assertGreater(p_yes, 0.5)
        self.assertLess(p_no, 0.5)
        self.assertAlmostEqual(p_yes + p_no, 1.0, places=12)

    def test_kxinxu_yes_probability_decreases_with_higher_strike(self) -> None:
        low_strike = "KXINXU-26JAN23H1600-T6874.9999"
        high_strike = "KXINXU-26JAN23H1600-T6974.9999"

        p_low = self.pricer.price(contract_id=low_strike, spx=self.spx, vix=self.vix, ts=self.ts)
        p_high = self.pricer.price(contract_id=high_strike, spx=self.spx, vix=self.vix, ts=self.ts)

        self.assertGreater(p_low, p_high)

    def test_kxinx_yes_no_are_complements(self) -> None:
        cid_yes = "KXINX-26JAN23H1600-B6887"
        cid_no = "KXINX:NO-26JAN23H1600-B6887"

        p_yes = self.pricer.price(contract_id=cid_yes, spx=self.spx, vix=self.vix, ts=self.ts)
        p_no = self.pricer.price(contract_id=cid_no, spx=self.spx, vix=self.vix, ts=self.ts)

        self.assertGreaterEqual(p_yes, 0.0)
        self.assertLessEqual(p_yes, 1.0)
        self.assertAlmostEqual(p_yes + p_no, 1.0, places=12)

    def test_kxinx_yes_matches_binary_call_spread_replication(self) -> None:
        cid_yes = "KXINX-26JAN23H1600-B6887"
        p_yes = self.pricer.price(contract_id=cid_yes, spx=self.spx, vix=self.vix, ts=self.ts)

        midpoint = 6887.0 + 0.5
        half_width = 12.5
        lower = midpoint - half_width
        upper = midpoint + half_width

        sigma = self.vix / 100.0
        tau = (datetime(2026, 1, 23, 16, 0) - self.ts).total_seconds() / (365.0 * 24.0 * 60.0 * 60.0)

        p_above_lower = self.pricer._prob_above(spx=self.spx, strike=lower, sigma=sigma, tau=tau)
        p_above_upper = self.pricer._prob_above(spx=self.spx, strike=upper, sigma=sigma, tau=tau)
        replicated = p_above_lower - p_above_upper

        self.assertAlmostEqual(p_yes, replicated, places=12)

    def test_unparseable_contract_returns_neutral_price(self) -> None:
        p = self.pricer.price(contract_id="UNKNOWN_CONTRACT", spx=self.spx, vix=self.vix, ts=self.ts)
        self.assertAlmostEqual(p, 0.5, places=12)


if __name__ == "__main__":
    unittest.main()
