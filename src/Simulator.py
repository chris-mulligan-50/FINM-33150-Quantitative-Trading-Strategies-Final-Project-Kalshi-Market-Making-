"""
Simulator.py

Market simulator for a Kalshi market-making bot.

Responsibilities
----------------
- Consume the long-form `all_df` from DataIngestor:
    ts | contract_id | take_bid | take_ask | best_bid | best_ask | spx | vix | spy | ...

- On each second (ts):
    1) call ExecutionEngine.on_tick(...) to get bot resting quotes & hedge order decision
    2) for each contract row at ts, apply your queue assumption to determine fills:
        - resting bid filled when take_bid <= my_bid - tick
        - resting ask filled when take_ask >= my_ask + tick
    3) send fill intents to ExecutionEngine.on_fills(...)
       (engine applies trades with 1-second delay internally)

- Log, per (ts, contract_id):
    - market snapshot (take_bid/take_ask/best_bid/best_ask)
    - macro snapshot (spx/vix/spy)
    - bot quotes
    - fill flags (whether we'd be filled at this ts)
    - position snapshot (Kalshi inventory for contract, SPY position, cash, pending trades)

Output
------
Returns a Polars DataFrame with one row per (ts, contract_id) containing both market
behavior and bot behavior for that second.

Notes
-----
- Fill detection is deterministic and uses the last-in-queue assumption.
- The engine delay is handled inside ExecutionEngine; the simulator does NOT delay
  the fill detection; it just reports fill intents at the current ts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import polars as pl


@dataclass(frozen=True)
class FillIntent:
    ts: Any
    contract_id: str
    side: str        # "buy" means our bid was hit; "sell" means our ask was lifted
    price: float
    size: int


class Simulator:
    def __init__(self, *, tick_size: float = 0.01) -> None:
        self.tick_size = float(tick_size)

    def run(
        self,
        *,
        all_df: pl.DataFrame,
        execution_engine,
        contract_ids: Optional[Sequence[str]] = None,
        log_engine_state: bool = True,
    ) -> pl.DataFrame:
        """
        Parameters
        ----------
        all_df:
            Output from DataIngestor.load()[0]. Must include:
              ts, contract_id, take_bid, take_ask, spx, vix, spy
            Optionally includes:
              best_bid, best_ask
        execution_engine:
            Instance of ExecutionEngine. Must expose:
              on_tick(ts, spx, vix, spy, contract_ids) -> dict
              on_fills(list[FillIntent]) -> None
              snapshot_state(contract_id) -> dict
              flush() -> None
        contract_ids:
            Optional explicit contract list to quote. If None, inferred from all_df.
        log_engine_state:
            If True, records position/cash/pending in output rows.

        Returns
        -------
        Polars DataFrame: one row per (ts, contract_id).
        """
        required = {"ts", "contract_id", "take_bid", "take_ask", "spx", "vix", "spy"}
        missing = required - set(all_df.columns)
        if missing:
            raise KeyError(f"all_df missing columns: {sorted(missing)}. Has: {all_df.columns}")

        # Infer contract ids if needed
        if contract_ids is None:
            contract_ids = list(all_df["contract_id"].unique())

        # Make sure we iterate in chronological order
        all_df = all_df.sort(["ts", "contract_id"])

        # Partition by ts for efficient looping
        # (ts values can be ints or datetimes; Polars keys are hashable)
        per_ts: Dict[Any, pl.DataFrame] = all_df.partition_by("ts", as_dict=True)
        ts_list = sorted(per_ts.keys())

        records: List[dict] = []

        for ts in ts_list:
            df_ts = per_ts[ts]

            # Macro is constant across all rows at ts; grab first row
            spx, vix, spy = [float(x) for x in df_ts.select(["spx", "vix", "spy"]).row(0)]

            # 1) Engine tick -> produces quotes & optional hedge order
            engine_out = execution_engine.on_tick(
                ts=ts,
                spx=spx,
                vix=vix,
                spy=spy,
                contract_ids=contract_ids,
            )

            quotes_by_contract = engine_out.get("quotes_by_contract", {})

            # 2) Determine fills under last-in-queue assumption
            fill_intents: List[FillIntent] = []

            for row in df_ts.iter_rows(named=True):
                cid = row["contract_id"]

                take_bid_raw = row["take_bid"]
                take_ask_raw = row["take_ask"]
                take_bid = float(take_bid_raw) if take_bid_raw is not None else None
                take_ask = float(take_ask_raw) if take_ask_raw is not None else None

                q = quotes_by_contract.get(cid)

                # Defaults for logging
                my_bid = my_ask = None
                my_bid_size = my_ask_size = None
                fair_value = None

                bid_fill = False
                ask_fill = False

                if q is not None:
                    my_bid = float(q["bid"])
                    my_ask = float(q["ask"])
                    my_bid_size = int(q["bid_size"])
                    my_ask_size = int(q["ask_size"])
                    fair_value = float(q.get("fair_value")) if q.get("fair_value") is not None else None

                    # Queue assumption: we are LAST at our price level
                    # Bid gets hit only if the taker sells through our level by one tick
                    bid_fill = (
                        (my_bid_size > 0)
                        and (take_bid is not None)
                        and (take_bid <= (my_bid - self.tick_size))
                    )

                    # Ask gets lifted only if the taker buys through our level by one tick
                    ask_fill = (
                        (my_ask_size > 0)
                        and (take_ask is not None)
                        and (take_ask >= (my_ask + self.tick_size))
                    )

                    if bid_fill:
                        fill_intents.append(FillIntent(
                            ts=ts, contract_id=cid, side="buy", price=my_bid, size=my_bid_size
                        ))
                    if ask_fill:
                        fill_intents.append(FillIntent(
                            ts=ts, contract_id=cid, side="sell", price=my_ask, size=my_ask_size
                        ))

                # 3) Record one row of behavior
                out = {
                    "ts": ts,
                    "contract_id": cid,
                    "spx": spx,
                    "vix": vix,
                    "spy": spy,
                    "take_bid": row.get("take_bid"),
                    "take_ask": row.get("take_ask"),
                    "best_bid": row.get("best_bid") if "best_bid" in df_ts.columns else None,
                    "best_ask": row.get("best_ask") if "best_ask" in df_ts.columns else None,
                    "fair_value": fair_value,
                    "my_bid": my_bid,
                    "my_ask": my_ask,
                    "my_bid_size": my_bid_size,
                    "my_ask_size": my_ask_size,
                    "bid_fill": bid_fill,
                    "ask_fill": ask_fill,
                }

                if log_engine_state:
                    st = execution_engine.snapshot_state(contract_id=cid)
                    out.update({
                        "pos_kalshi": st.get("pos_kalshi"),
                        "pos_spy": st.get("pos_spy"),
                        "cash": st.get("cash"),
                        "pending_trades": st.get("pending_trades"),
                        "total_kalshi_inventory": st.get("total_kalshi_inventory"),
                    })

                records.append(out)

            # 4) Report fills to ExecutionEngine (it applies them with +1s delay)
            if fill_intents:
                execution_engine.on_fills(fill_intents)

        # 5) Flush remaining delayed trades at end of run
        execution_engine.flush()

        return pl.DataFrame(records)


if __name__ == "__main__":
    # Small smoke test (requires you to wire in real data)
    print("Simulator module loaded. Wire it with DataIngestor + ExecutionEngine to run.")
