"""
Simulator.py

Market simulator for a Kalshi market-making bot.

Responsibilities
----------------
- Consume the long-form `all_df` from DataIngestor:
    ts | contract_id | take_bid | take_ask | spx | vix | spy | ...

- On each second (ts):
    1) call ExecutionEngine.on_tick(...) to get bot resting quotes & hedge order decision
    2) for each contract row at ts, apply your queue assumption to determine fills:
        - resting bid filled when take_bid <= my_bid - tick
        - resting ask filled when take_ask >= my_ask + tick
        - fill size is capped by taker-side market quantity when provided
    3) send fill intents to ExecutionEngine.on_fills(...)
       (engine applies trades with 1-second delay internally)

- After the last tick: flush delayed trades, then settle expired Kalshi contracts
  (PositionManager.settle_expired_contracts) using the final timestamp and SPX.
  Contracts that have passed their settlement time pay out 0 or 1 per contract;
  cash is updated and positions are cleared.

- Log, per (ts, contract_id):
    - market snapshot (take_bid/take_ask)
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
import math
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

    @staticmethod
    def _normalize_ts_key(ts_key: Any) -> Any:
        """
        Polars partition_by(as_dict=True) may return 1-column group keys wrapped
        as tuples/lists. Normalize to the scalar timestamp so downstream modules
        can do timestamp arithmetic and equality checks reliably.
        """
        if isinstance(ts_key, tuple) and len(ts_key) == 1:
            return ts_key[0]
        if isinstance(ts_key, list) and len(ts_key) == 1:
            return ts_key[0]
        return ts_key

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
            Optional:
              take_bid_qty, take_ask_qty (used to cap fill sizes)
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
        base_equity: Optional[float] = None
        prev_equity: Optional[float] = None

        for ts_key in ts_list:
            ts = self._normalize_ts_key(ts_key)
            df_ts = per_ts[ts_key]

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
            kalshi_delta_spx = engine_out.get("kalshi_delta_spx")
            equity = self._portfolio_equity(
                execution_engine=execution_engine,
                quotes_by_contract=quotes_by_contract,
                ts=ts,
                spx=spx,
                vix=vix,
                spy=spy,
            )
            if base_equity is None:
                base_equity = float(equity)
            if prev_equity is None:
                returns = 0.0
            else:
                # "returns" is defined as period-over-period change in PnL.
                returns = float(equity) - float(prev_equity)
            pnl = float(equity) - float(base_equity)
            prev_equity = float(equity)

            # 2) Determine fills under last-in-queue assumption
            fill_intents: List[FillIntent] = []

            for row in df_ts.iter_rows(named=True):
                cid = row["contract_id"]

                take_bid_raw = row["take_bid"]
                take_ask_raw = row["take_ask"]
                take_bid = float(take_bid_raw) if take_bid_raw is not None else None
                take_ask = float(take_ask_raw) if take_ask_raw is not None else None
                take_bid_qty = self._to_nonnegative_int(row.get("take_bid_qty"))
                take_ask_qty = self._to_nonnegative_int(row.get("take_ask_qty"))

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
                        bid_fill_size = int(my_bid_size)
                        if take_bid_qty is not None:
                            bid_fill_size = min(bid_fill_size, int(take_bid_qty))
                        if bid_fill_size > 0:
                            fill_intents.append(FillIntent(
                                ts=ts, contract_id=cid, side="buy", price=my_bid, size=bid_fill_size
                            ))
                    if ask_fill:
                        ask_fill_size = int(my_ask_size)
                        if take_ask_qty is not None:
                            ask_fill_size = min(ask_fill_size, int(take_ask_qty))
                        if ask_fill_size > 0:
                            fill_intents.append(FillIntent(
                                ts=ts, contract_id=cid, side="sell", price=my_ask, size=ask_fill_size
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
                    "take_bid_qty": take_bid_qty,
                    "take_ask_qty": take_ask_qty,
                    "fair_value": fair_value,
                    "my_bid": my_bid,
                    "my_ask": my_ask,
                    "my_bid_size": my_bid_size,
                    "my_ask_size": my_ask_size,
                    "bid_fill": bid_fill,
                    "ask_fill": ask_fill,
                    "portfolio_value": float(equity),
                    "pnl": float(pnl),
                    "returns": float(returns),
                    "kalshi_delta_spx": float(kalshi_delta_spx) if kalshi_delta_spx is not None else None,
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

        # 6) Settle final Kalshi positions at expiry (contracts pay 0 or 1 based on outcome)
        if ts_list:
            last_ts_key = ts_list[-1]
            last_ts = self._normalize_ts_key(last_ts_key)
            last_df = per_ts[last_ts_key]
            last_spx = float(last_df.select("spx").row(0)[0])
            execution_engine.pm.settle_expired_contracts(ts=last_ts, settlement_spx=last_spx)

        return pl.DataFrame(records)

    @staticmethod
    def _portfolio_equity(
        *,
        execution_engine,
        quotes_by_contract: Dict[str, Dict[str, Any]],
        ts: Any,
        spx: float,
        vix: float,
        spy: float,
    ) -> float:
        """
        Mark-to-market equity:
          cash + SPY position value + sum(Kalshi qty * contract fair value)
        """
        pm = execution_engine.pm
        cash = float(pm.get_cash())
        spy_pos = int(pm.get_spy_position())
        kalshi_positions = pm.get_kalshi_positions()

        kalshi_value = 0.0
        for cid, qty in kalshi_positions.items():
            q = int(qty)
            if q == 0:
                continue

            qv = quotes_by_contract.get(cid)
            if qv is not None and qv.get("fair_value") is not None:
                fv = float(qv["fair_value"])
            else:
                try:
                    fv = float(execution_engine.pricer.price(
                        contract_id=cid,
                        spx=float(spx),
                        vix=float(vix),
                        ts=ts,
                    ))
                except TypeError:
                    fv = float(execution_engine.pricer.price(
                        contract_id=cid,
                        spx=float(spx),
                        vix=float(vix),
                    ))
            if not math.isfinite(fv):
                fv = 0.0
            kalshi_value += q * fv

        return cash + float(spy_pos) * float(spy) + kalshi_value

    @staticmethod
    def _to_nonnegative_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            q = int(float(value))
        except Exception:
            return None
        return max(0, q)


if __name__ == "__main__":
    # Small smoke test (requires you to wire in real data)
    print("Simulator module loaded. Wire it with DataIngestor + ExecutionEngine to run.")
