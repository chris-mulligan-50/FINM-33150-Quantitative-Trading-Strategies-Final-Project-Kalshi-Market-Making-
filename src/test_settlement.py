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
from ExecutionEngine import ExecutionEngine
from MarketMaker import MarketMaker
from PositionManager import PositionManager
from Pricer import Pricer


class SettlementTests(unittest.TestCase):
    def test_settlement_occurs_at_start_of_next_day(self) -> None:
        pm = PositionManager(initial_cash=10.0)
        cid = "KXINXU-26JAN23H1600-T7000"
        pm.apply_kalshi_trade(contract_id=cid, side="buy", qty=3, price=0.40)

        before_settle = datetime(2026, 1, 23, 23, 59, 59)
        pm.settle_expired_contracts(ts=before_settle, settlement_spx=7100.0)
        self.assertEqual(pm.get_kalshi_position(cid), 3)

        settle_ts = datetime(2026, 1, 24, 0, 0, 0)
        pm.settle_expired_contracts(ts=settle_ts, settlement_spx=7100.0)
        self.assertEqual(pm.get_kalshi_position(cid), 0)
        self.assertAlmostEqual(pm.get_cash(), 10.0 - 3 * 0.40 + 3 * 1.0, places=12)

    def test_no_contract_settles_to_complementary_outcome(self) -> None:
        pm = PositionManager(initial_cash=10.0)
        cid_no = "KXINXU:NO-26JAN23H1600-T7000"
        pm.apply_kalshi_trade(contract_id=cid_no, side="buy", qty=2, price=0.25)

        settle_ts = datetime(2026, 1, 24, 0, 0, 0)
        pm.settle_expired_contracts(ts=settle_ts, settlement_spx=7100.0)
        self.assertEqual(pm.get_kalshi_position(cid_no), 0)
        # NO contract settles to 0 when SPX > strike.
        self.assertAlmostEqual(pm.get_cash(), 10.0 - 2 * 0.25, places=12)

    def test_execution_engine_rehedges_after_settlement(self) -> None:
        mm = MarketMaker()
        engine = ExecutionEngine(
            market_maker=mm,
            pricer=Pricer(),
            delta_hedger=DeltaHedger(max_qty_per_tick=100),
            execution_delay_seconds=0,
            quote_contracts=["KXINXU-26JAN23H1600-T7000"],
        )
        cid = "KXINXU-26JAN23H1600-T7000"
        engine.pm.apply_kalshi_trade(contract_id=cid, side="buy", qty=5, price=0.40)
        engine.pm.apply_spy_trade(side="buy", qty=7, price=700.0)

        out = engine.on_tick(
            ts=datetime(2026, 1, 24, 0, 0, 0),
            spx=7100.0,
            vix=20.0,
            spy=700.0,
            contract_ids=[cid],
        )

        self.assertEqual(engine.pm.get_kalshi_position(cid), 0)
        hedge_order = out.get("hedge_order")
        self.assertIsNotNone(hedge_order)
        assert hedge_order is not None
        self.assertEqual(hedge_order.side, "sell")
        self.assertEqual(hedge_order.qty, 7)


if __name__ == "__main__":
    unittest.main()
