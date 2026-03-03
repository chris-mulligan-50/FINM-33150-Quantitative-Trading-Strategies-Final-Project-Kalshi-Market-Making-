"""
main.py

CLI runner that wires:
DataIngestor -> MarketMaker -> ExecutionEngine -> Simulator
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List
import time
import functools

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from DataIngestor import DataIngestor, KalshiCleanSpec, KalshiMarketSpec, MacroSpec
from DeltaHedger import NoHedgeDeltaHedger
from ExecutionEngine import ExecutionEngine
from MarketMaker import MarketMaker
from Simulator import Simulator

def timer(func):
    """
    Decorator that prints the execution time of the decorated function.
    """
    import functools
    import time

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function '{func.__name__}' executed in {end - start:.4f} seconds.")
        return result
    return wrapper_timer

def _detect_macro_value_col(path: str, fallback_name: str) -> str:
    """
    Pick a value column from a macro parquet file.
    Priority:
      1) exact series name (spx/vix/spy)
      2) generic 'value'
      3) first numeric non-ts/date-like column
    """
    cols = pl.read_parquet_schema(path)
    if fallback_name in cols:
        return fallback_name
    if "value" in cols:
        return "value"
    for preferred in ("close", "price", "last", "mid", "open"):
        if preferred in cols:
            return preferred

    excluded = {"ts", "timestamp", "date", "datetime", "time"}
    for name, dtype in cols.items():
        if name.lower() in excluded:
            continue
        if dtype in {
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.Float32,
            pl.Float64,
        }:
            return name

    raise ValueError(
        f"Could not infer value column for {path}. "
        f"Expected one of '{fallback_name}' or 'value', or any numeric non-time column."
    )


def _parse_kalshi_market_specs(items: List[str]) -> List[KalshiMarketSpec]:
    specs: List[KalshiMarketSpec] = []
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --kalshi-market value '{item}'. Use CONTRACT_ID=/path/to/file.parquet"
            )
        contract_id, path = item.split("=", 1)
        contract_id = contract_id.strip()
        path = path.strip()
        if not contract_id or not path:
            raise ValueError(
                f"Invalid --kalshi-market value '{item}'. Use CONTRACT_ID=/path/to/file.parquet"
            )
        specs.append(KalshiMarketSpec(contract_id=contract_id, path=path))
    return specs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Kalshi market-making simulation.")

    # Macro inputs (validated after parse so we can show a custom hint)
    parser.add_argument("--spx", help="Path to SPX parquet.")
    parser.add_argument("--vix", help="Path to VIX parquet.")
    parser.add_argument("--spy", help="Path to SPY parquet.")

    # Exactly one Kalshi source type
    parser.add_argument(
        "--kalshi-market",
        action="append",
        default=[],
        help="Kalshi market spec in format CONTRACT_ID=/path/to/market.parquet. Repeat per contract.",
    )
    parser.add_argument(
        "--kalshi-clean",
        action="append",
        default=[],
        help="Path to cleaned Kalshi event parquet. Repeat for multiple files.",
    )

    # Optional runtime params
    parser.add_argument("--tick-size", type=float, default=0.01, help="Tick size for Simulator/MarketMaker.")
    parser.add_argument("--base-spread", type=float, default=0.02, help="Base spread for MarketMaker.")
    parser.add_argument(
        "--out-of-market-spread-ticks",
        type=int,
        default=0,
        metavar="N",
        help="Widen bid/ask spread by N ticks outside regular hours (9:30–16:00 ET). Default: 0.",
    )
    parser.add_argument("--execution-delay", type=int, default=1, help="Execution delay in seconds.")
    parser.add_argument("--default-quote-size", type=int, default=None, help="Optional quote size override.")
    parser.add_argument("--timezone", default=None, help="Optional timezone for DataIngestor timestamp handling.")
    parser.add_argument(
        "--output",
        default="simulation_output.parquet",
        help="Output file path (.parquet or .csv). Default: simulation_output.parquet",
    )
    parser.add_argument(
        "--no-hedge",
        action="store_true",
        help="Disable SPY delta hedging; run strategy with no hedges (for comparison).",
    )

    return parser

@timer
def main() -> None:
    parser = _build_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        print(
            "\nExample:\n"
            "  python main.py --spx /path/spx.parquet --vix /path/vix.parquet --spy /path/spy.parquet "
            "--kalshi-market CONTRACT_A=/path/a.parquet --kalshi-market CONTRACT_B=/path/b.parquet"
        )
        return

    args = parser.parse_args()

    missing_macro = [name for name in ("spx", "vix", "spy") if not getattr(args, name)]
    if missing_macro:
        parser.error(
            "Missing required macro args: "
            + ", ".join(f"--{name}" for name in missing_macro)
            + ". Run `python main.py -h` for usage."
        )

    use_markets = len(args.kalshi_market) > 0
    use_clean = len(args.kalshi_clean) > 0
    if use_markets == use_clean:
        raise ValueError(
            "Pass exactly one Kalshi source: either --kalshi-market (one or more) "
            "or --kalshi-clean (one or more)."
        )

    ingestor = DataIngestor(tz=args.timezone)
    spx_spec = MacroSpec(path=args.spx, value_col=_detect_macro_value_col(args.spx, "spx"))
    vix_spec = MacroSpec(path=args.vix, value_col=_detect_macro_value_col(args.vix, "vix"))
    spy_spec = MacroSpec(path=args.spy, value_col=_detect_macro_value_col(args.spy, "spy"))

    if use_markets:
        kalshi_markets: List[KalshiMarketSpec] = _parse_kalshi_market_specs(args.kalshi_market)
        kalshi_clean_files = None
        contract_ids = [m.contract_id for m in kalshi_markets]
    else:
        kalshi_markets = None
        kalshi_clean_files = [KalshiCleanSpec(path=p) for p in args.kalshi_clean]
        contract_ids = None

    all_df, macro_df = ingestor.load(
        kalshi_markets=kalshi_markets,
        kalshi_clean_files=kalshi_clean_files,
        spx=spx_spec,
        vix=vix_spec,
        spy=spy_spec,
    )

    # Some timestamps in Kalshi data may not have an exact macro match.
    # Fill macro series through time, then drop any still-missing rows.
    pre_rows = all_df.height
    all_df = (
        all_df.sort(["ts", "contract_id"])
        .with_columns([
            pl.col("spx").forward_fill().backward_fill(),
            pl.col("vix").forward_fill().backward_fill(),
            pl.col("spy").forward_fill().backward_fill(),
        ])
        .filter(
            pl.col("spx").is_not_null()
            & pl.col("vix").is_not_null()
            & pl.col("spy").is_not_null()
        )
    )
    dropped_rows = pre_rows - all_df.height

    market_maker = MarketMaker(
        tick_size=float(args.tick_size),
        base_spread=float(args.base_spread),
        out_of_market_spread_ticks=int(args.out_of_market_spread_ticks),
    )
    delta_hedger = NoHedgeDeltaHedger() if args.no_hedge else None
    engine = ExecutionEngine(
        market_maker=market_maker,
        delta_hedger=delta_hedger,
        execution_delay_seconds=int(args.execution_delay),
        default_quote_size=args.default_quote_size,
    )
    simulator = Simulator(tick_size=float(args.tick_size))

    result_df = simulator.run(
        all_df=all_df,
        execution_engine=engine,
        contract_ids=contract_ids,
        log_engine_state=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        result_df.write_csv(str(output_path))
    else:
        result_df.write_parquet(str(output_path))

    print(f"Loaded all_df rows: {all_df.height}")
    print(f"Loaded macro_df rows: {macro_df.height}")
    print(f"Dropped rows with unresolved macro data: {dropped_rows}")
    print(f"Simulation output rows: {result_df.height}")
    print(f"Saved output: {output_path}")

if __name__ == "__main__":
    main()