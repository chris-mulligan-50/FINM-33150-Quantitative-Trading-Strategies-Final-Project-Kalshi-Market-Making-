from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from DeltaHedger import DeltaHedger


class DeltaHedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hedger = DeltaHedger(max_qty_per_tick=50)
        self.ts = datetime(2026, 3, 10, 15, 0)
        self.spx = 7000.0
        self.vix = 20.0
        self.spy = 700.0

    def test_yes_and_no_above_deltas_are_opposites(self) -> None:
        cid_yes = "KXINXU-26MAR10H1600-T7000"
        cid_no = "KXINXU:NO-26MAR10H1600-T7000"
        sigma = self.hedger._sigma_from_vix(self.vix)

        d_yes = self.hedger._contract_delta_spx(
            ts=self.ts, contract_id=cid_yes, spx_price=self.spx, sigma=sigma
        )
        d_no = self.hedger._contract_delta_spx(
            ts=self.ts, contract_id=cid_no, spx_price=self.spx, sigma=sigma
        )

        self.assertGreater(d_yes, 0.0)
        self.assertLess(d_no, 0.0)
        self.assertAlmostEqual(d_yes, -d_no, places=12)

    def test_hedge_order_uses_book_delta_and_spx_to_spy_mapping(self) -> None:
        positions = {
            "KXINXU-26MAR10H1600-T7000": 10,       # YES ABOVE
            "KXINXU:NO-26MAR10H1600-T7000": 5,     # NO ABOVE
        }

        order = self.hedger.hedge(
            ts=self.ts,
            spy_price=self.spy,
            spx_price=self.spx,
            vix=self.vix,
            kalshi_positions=positions,
            current_spy_position=0,
        )

        self.assertIsNotNone(order)
        assert order is not None
        self.assertEqual(order.side, "sell")
        self.assertEqual(order.qty, 1)
        self.assertEqual(order.symbol, "SPY")

    def test_max_qty_per_tick_caps_large_rebalance(self) -> None:
        positions = {"KXINXU-26MAR10H1600-T7000": 10000}

        order = self.hedger.hedge(
            ts=self.ts,
            spy_price=self.spy,
            spx_price=self.spx,
            vix=self.vix,
            kalshi_positions=positions,
            current_spy_position=0,
        )

        self.assertIsNotNone(order)
        assert order is not None
        self.assertEqual(order.qty, 50)

    def test_unparseable_contract_ids_generate_no_hedge(self) -> None:
        positions = {"UNKNOWN_CONTRACT": 100}
        order = self.hedger.hedge(
            ts=self.ts,
            spy_price=self.spy,
            spx_price=self.spx,
            vix=self.vix,
            kalshi_positions=positions,
            current_spy_position=0,
        )
        self.assertIsNone(order)


if __name__ == "__main__":
    unittest.main()
