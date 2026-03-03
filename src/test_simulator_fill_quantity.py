from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Simulator import Simulator


class _StubPM:
    def get_cash(self) -> float:
        return 10_000.0

    def get_spy_position(self) -> int:
        return 0

    def get_kalshi_positions(self) -> Dict[str, int]:
        return {}


class _StubPricer:
    def price(self, *, contract_id: str, spx: float, vix: float, ts: Any = None) -> float:
        _ = (contract_id, spx, vix, ts)
        return 0.5


class _StubEngine:
    def __init__(self, contract_id: str) -> None:
        self.contract_id = contract_id
        self.pm = _StubPM()
        self.pricer = _StubPricer()
        self.fills_received: List[List[Any]] = []

    def on_tick(
        self,
        *,
        ts: Any,
        spx: float,
        vix: float,
        spy: float,
        contract_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        _ = (ts, spx, vix, spy, contract_ids)
        return {
            "quotes_by_contract": {
                self.contract_id: {
                    "contract_id": self.contract_id,
                    "fair_value": 0.5,
                    "bid": 0.40,
                    "ask": 0.60,
                    "bid_size": 10,
                    "ask_size": 7,
                }
            },
            "hedge_order": None,
        }

    def on_fills(self, fills: List[Any]) -> None:
        self.fills_received.append(list(fills))

    def snapshot_state(self, contract_id: Optional[str] = None) -> Dict[str, Any]:
        _ = contract_id
        return {
            "pos_kalshi": 0,
            "pos_spy": 0,
            "cash": 10_000.0,
            "pending_trades": 0,
            "total_kalshi_inventory": 0,
        }

    def flush(self) -> None:
        return


class SimulatorFillQuantityTests(unittest.TestCase):
    def test_fill_sizes_are_capped_by_market_quantities(self) -> None:
        cid = "KXINXU-26MAR10H1600-T7000"
        engine = _StubEngine(contract_id=cid)
        sim = Simulator(tick_size=0.01)

        all_df = pl.DataFrame(
            {
                "ts": [datetime(2026, 3, 10, 15, 0, 0)],
                "contract_id": [cid],
                "take_bid": [0.38],      # below bid - tick => bid fill condition true
                "take_ask": [0.62],      # above ask + tick => ask fill condition true
                "take_bid_qty": [3.0],   # caps bid fill size from 10 -> 3
                "take_ask_qty": [2.0],   # caps ask fill size from 7 -> 2
                "spx": [7000.0],
                "vix": [20.0],
                "spy": [700.0],
            }
        )

        _ = sim.run(all_df=all_df, execution_engine=engine, contract_ids=[cid])
        self.assertEqual(len(engine.fills_received), 1)
        fills = engine.fills_received[0]
        self.assertEqual(len(fills), 2)

        by_side = {f.side: f for f in fills}
        self.assertIn("buy", by_side)
        self.assertIn("sell", by_side)
        self.assertEqual(by_side["buy"].size, 3)
        self.assertEqual(by_side["sell"].size, 2)


if __name__ == "__main__":
    unittest.main()
