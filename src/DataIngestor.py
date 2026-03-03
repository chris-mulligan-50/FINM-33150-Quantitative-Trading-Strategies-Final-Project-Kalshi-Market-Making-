"""
DataIngestor.py

Loads 1-second data for:
- Kalshi markets from either:
  1) per-contract parquet files, or
  2) cleaned trade-event parquet files (e.g. KXINX/KXINXU)
- SPX price (parquet)
- VIX value (parquet)
- SPY price (parquet)

Outputs:
- all_df:     Polars DataFrame with *all* data (Kalshi + SPX/VIX/SPY), in LONG form
- macro_df:   Polars DataFrame with only SPX/VIX/SPY (for ExecutionEngine)

Design Philosophy
-----------------
• Kalshi data is stored LONG:
    ts | contract_id | take_bid | take_ask | take_bid_qty | take_ask_qty

• Macro data is stored WIDE:
    ts | spx | vix | spy

• all_df = kalshi_df LEFT JOIN macro_df on ts
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import polars as pl


# ============================================================
# CONFIG STRUCTS
# ============================================================

@dataclass(frozen=True)
class KalshiMarketSpec:
    contract_id: str
    path: str

    ts_col: str = "ts"
    take_bid_col: str = "take_bid"
    take_ask_col: str = "take_ask"
    take_bid_qty_col: Optional[str] = None
    take_ask_qty_col: Optional[str] = None


@dataclass(frozen=True)
class MacroSpec:
    path: str
    ts_col: str = "ts"
    value_col: str = "value"


@dataclass(frozen=True)
class KalshiCleanSpec:
    path: str
    contract_id_col: str = "ticker"
    ts_col: str = "ts"
    price_col: str = "price"
    quantity_col: str = "quantity"
    liquidity_event_col: str = "liquidity_event"
    take_bid_event_value: str = "hit_bid"
    take_ask_event_value: str = "lift_offer"


# ============================================================
# DATA INGESTOR
# ============================================================

class DataIngestor:

    def __init__(self, *, tz: Optional[str] = None):
        """
        tz: optional timezone if timestamps are datetime
        """
        self.tz = tz

    def load(
        self,
        *,
        kalshi_markets: Optional[Sequence[KalshiMarketSpec]] = None,
        kalshi_clean_files: Optional[Sequence[KalshiCleanSpec]] = None,
        spx: MacroSpec,
        vix: MacroSpec,
        spy: MacroSpec,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        if kalshi_markets and kalshi_clean_files:
            raise ValueError("Pass either kalshi_markets or kalshi_clean_files, not both.")
        if not kalshi_markets and not kalshi_clean_files:
            raise ValueError("You must pass kalshi_markets or kalshi_clean_files.")

        macro_df = self._load_macro(spx=spx, vix=vix, spy=spy)
        if kalshi_clean_files:
            kalshi_df = self._load_kalshi_clean_files(kalshi_clean_files)
        else:
            kalshi_df = self._load_kalshi(kalshi_markets or [])

        # Join macro onto kalshi
        all_df = (
            kalshi_df
            .join(macro_df, on="ts", how="left")
            .sort(["ts", "contract_id"])
        )

        macro_df = macro_df.sort("ts")

        return all_df, macro_df

    # --------------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------------

    def _standardize_ts(self, df: pl.DataFrame, ts_col: str) -> pl.DataFrame:

        if ts_col not in df.columns:
            raise KeyError(f"Missing timestamp column '{ts_col}'")

        dtype = df[ts_col].dtype

        if dtype == pl.Utf8:
            df = df.with_columns(
                pl.col(ts_col).str.to_datetime(strict=False)
            )

        if self.tz and df[ts_col].dtype == pl.Datetime:
            df = df.with_columns(
                pl.col(ts_col).dt.replace_time_zone(self.tz)
            )

        return df

    def _load_macro(self, *, spx: MacroSpec, vix: MacroSpec, spy: MacroSpec) -> pl.DataFrame:

        spx_df = pl.read_parquet(spx.path)
        vix_df = pl.read_parquet(vix.path)
        spy_df = pl.read_parquet(spy.path)

        spx_df = (
            self._standardize_ts(spx_df, spx.ts_col)
            .select([
                pl.col(spx.ts_col).alias("ts"),
                pl.col(spx.value_col).cast(pl.Float64).alias("spx")
            ])
        )

        vix_df = (
            self._standardize_ts(vix_df, vix.ts_col)
            .select([
                pl.col(vix.ts_col).alias("ts"),
                pl.col(vix.value_col).cast(pl.Float64).alias("vix")
            ])
        )

        spy_df = (
            self._standardize_ts(spy_df, spy.ts_col)
            .select([
                pl.col(spy.ts_col).alias("ts"),
                pl.col(spy.value_col).cast(pl.Float64).alias("spy")
            ])
        )

        macro_df = (
            spx_df
            .join(vix_df, on="ts", how="inner")
            .join(spy_df, on="ts", how="inner")
        )

        return macro_df

    def _load_kalshi(self, markets: Sequence[KalshiMarketSpec]) -> pl.DataFrame:
        if not markets:
            raise ValueError("kalshi_markets is empty.")

        dfs = []

        for m in markets:
            df = pl.read_parquet(m.path)
            df = self._standardize_ts(df, m.ts_col)

            cols = [m.ts_col, m.take_bid_col, m.take_ask_col]
            if m.take_bid_qty_col is not None:
                cols.append(m.take_bid_qty_col)
            if m.take_ask_qty_col is not None:
                cols.append(m.take_ask_qty_col)

            df = df.select(cols)

            with_cols = [
                pl.lit(m.contract_id).alias("contract_id"),
                pl.col(m.take_bid_col).cast(pl.Float64).alias("take_bid"),
                pl.col(m.take_ask_col).cast(pl.Float64).alias("take_ask"),
            ]
            if m.take_bid_qty_col is not None:
                with_cols.append(pl.col(m.take_bid_qty_col).cast(pl.Float64).alias("take_bid_qty"))
            else:
                with_cols.append(pl.lit(None).cast(pl.Float64).alias("take_bid_qty"))
            if m.take_ask_qty_col is not None:
                with_cols.append(pl.col(m.take_ask_qty_col).cast(pl.Float64).alias("take_ask_qty"))
            else:
                with_cols.append(pl.lit(None).cast(pl.Float64).alias("take_ask_qty"))
            df = df.with_columns(with_cols)

            df = df.select([
                pl.col(m.ts_col).alias("ts"),
                "contract_id",
                "take_bid",
                "take_ask",
                "take_bid_qty",
                "take_ask_qty",
            ])

            dfs.append(df)

        return pl.concat(dfs, how="vertical")

    def _load_kalshi_clean_files(self, files: Sequence[KalshiCleanSpec]) -> pl.DataFrame:
        if not files:
            raise ValueError("kalshi_clean_files is empty.")

        dfs = []

        for spec in files:
            df = pl.read_parquet(spec.path)
            df = self._standardize_ts(df, spec.ts_col)

            required_cols = {
                spec.contract_id_col,
                spec.ts_col,
                spec.price_col,
                spec.quantity_col,
                spec.liquidity_event_col,
            }
            missing = required_cols - set(df.columns)
            if missing:
                raise KeyError(f"{spec.path} missing required columns: {sorted(missing)}")

            df = (
                df.select([
                    pl.col(spec.ts_col).alias("ts"),
                    pl.col(spec.contract_id_col).cast(pl.Utf8).alias("contract_id"),
                    pl.col(spec.price_col).cast(pl.Float64).alias("price"),
                    pl.col(spec.quantity_col).cast(pl.Float64).alias("quantity"),
                    pl.col(spec.liquidity_event_col).cast(pl.Utf8).alias("liquidity_event"),
                ])
                .with_columns([
                    # Macro data is at 1Hz. Bucket event timestamps to the same cadence.
                    pl.col("ts").dt.truncate("1s").alias("ts"),
                ])
                .group_by(["ts", "contract_id"])
                .agg([
                    pl.when(pl.col("liquidity_event") == spec.take_bid_event_value)
                    .then(pl.col("price"))
                    .otherwise(None)
                    .min()
                    .alias("take_bid"),
                    pl.when(pl.col("liquidity_event") == spec.take_bid_event_value)
                    .then(pl.col("quantity"))
                    .otherwise(None)
                    .sum()
                    .alias("take_bid_qty"),
                    pl.when(pl.col("liquidity_event") == spec.take_ask_event_value)
                    .then(pl.col("price"))
                    .otherwise(None)
                    .max()
                    .alias("take_ask"),
                    pl.when(pl.col("liquidity_event") == spec.take_ask_event_value)
                    .then(pl.col("quantity"))
                    .otherwise(None)
                    .sum()
                    .alias("take_ask_qty"),
                ])
                .filter(
                    pl.col("take_bid").is_not_null() | pl.col("take_ask").is_not_null()
                )
                .select(["ts", "contract_id", "take_bid", "take_ask", "take_bid_qty", "take_ask_qty"])
            )

            dfs.append(df)

        return pl.concat(dfs, how="vertical")