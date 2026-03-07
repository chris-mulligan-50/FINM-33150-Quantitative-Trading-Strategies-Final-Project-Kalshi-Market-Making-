from datetime import datetime
import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pytz
from IPython.display import display

# ── Global constants (mirrored in notebook) ───────────────────────────────────
INITIAL_CAPITAL = 10_000.0
ANN_SECONDS     = 252 * 6.5 * 3600   # trading seconds per year


def fmt_ts(ts):
    return ts.strftime("%b %d, %Y")


def fmt_date(d):
    return datetime.strptime(d, "%Y-%m-%d").strftime("%b %d, %Y")


def format_summary(spx, vix, spy, kxinx, kxinxu):
    """
    Formats a summary table of the datasets used in the project, including their source,
    frequency, period covered, number of records, and coverage details.
    """

    rows = [
        (
            "SPX",
            "Bloomberg",
            "1-second",
            f"{fmt_ts(spx['ts'].min())} – {fmt_ts(spx['ts'].max())}",
            f"{spx.height:,}",
            "Extended hours",
        ),
        (
            "VIX",
            "Bloomberg",
            "1-second",
            f"{fmt_ts(vix['ts'].min())} – {fmt_ts(vix['ts'].max())}",
            f"{vix.height:,}",
            "Extended hours",
        ),
        (
            "SPY",
            "Databento",
            "1-second OHLCV",
            f"{fmt_ts(spy['ts'].min())} – {fmt_ts(spy['ts'].max())}",
            f"{spy.height:,}",
            "RTH only (9:30–16:00 ET)",
        ),
        (
            "KXINX (range)",
            "Kalshi API",
            "Trade-event",
            f"{fmt_date(kxinx['date'].min())} – {fmt_date(kxinx['date'].max())}",
            f"{kxinx.height:,}",
            f"{kxinx['ticker'].n_unique()} contracts, {kxinx['date'].n_unique()} trading days",
        ),
        (
            "KXINXU (threshold)",
            "Kalshi API",
            "Trade-event",
            f"{fmt_date(kxinxu['date'].min())} – {fmt_date(kxinxu['date'].max())}",
            f"{kxinxu.height:,}",
            f"{kxinxu['ticker'].n_unique()} contracts, {kxinxu['date'].n_unique()} trading days",
        ),
    ]

    summary = pd.DataFrame(
        rows, columns=["Dataset", "Source", "Frequency", "Period", "Records", "Coverage"]
    )
    return summary


def format_all_df_summary(all_df):
    pre_rows = all_df.height
    all_df = (
        all_df.sort(["ts", "contract_id"])
        .with_columns(
            [
                pl.col("spx").forward_fill().backward_fill(),
                pl.col("vix").forward_fill().backward_fill(),
                pl.col("spy").forward_fill().backward_fill(),
            ]
        )
        .filter(
            pl.col("spx").is_not_null()
            & pl.col("vix").is_not_null()
            & pl.col("spy").is_not_null()
        )
    )
    return all_df, pre_rows


def plot_market_environment(spx_daily, vix_daily):
    """
    Plots the daily closing levels of the S&P 500 (SPX) and the CBOE VIX over the period of January 2 to February 26, 2026.
    The first subplot shows the SPX index level, while the second subplot shows the VIX level.
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    fig.suptitle("Market Environment: January 2 – February 26, 2026", fontsize=13, fontweight="bold")

    ax1.plot(spx_daily["date"], spx_daily["spx"], color="steelblue", linewidth=1.5)
    ax1.set_ylabel("Index Level")
    ax1.set_title("S&P 500 (SPX) — Daily Close")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")

    ax2.plot(vix_daily["date"], vix_daily["vix"], color="tomato", linewidth=1.5)
    ax2.set_ylabel("VIX Level")
    ax2.set_title("CBOE VIX — Daily Close")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.show()


def kalshi_vwap(kxinx, kxinxu, spx_close_daily):
    kxinx_vwap = (
        kxinx.group_by(["date", "ticker", "bracket_floor", "bracket_cap", "side_of_close"])
        .agg(
            (pl.col("price") * pl.col("quantity")).sum().alias("pq"),
            pl.col("quantity").sum().alias("total_qty"),
        )
        .with_columns((pl.col("pq") / pl.col("total_qty")).alias("vwap"))
        .join(spx_close_daily, on="date", how="left")
        .with_columns(
            pl.when(pl.col("side_of_close") == "above")
            .then(pl.col("bracket_floor") - pl.col("spx_close"))
            .otherwise(pl.col("spx_close") - pl.col("bracket_cap"))
            .alias("distance_from_close")
        )
    )

    kxinxu_vwap = (
        kxinxu.group_by(["date", "ticker", "threshold", "side_of_close"])
        .agg(
            (pl.col("price") * pl.col("quantity")).sum().alias("pq"),
            pl.col("quantity").sum().alias("total_qty"),
        )
        .with_columns((pl.col("pq") / pl.col("total_qty")).alias("vwap"))
        .join(spx_close_daily, on="date", how="left")
        .with_columns(
            pl.when(pl.col("side_of_close") == "above")
            .then(pl.col("threshold") - pl.col("spx_close"))
            .otherwise(pl.col("spx_close") - pl.col("threshold"))
            .alias("distance_from_close")
        )
    )

    return kxinx_vwap, kxinxu_vwap


def plot_trade_activity_by_hour(kxinx, kxinxu):
    k_all = pl.concat(
        [
            kxinx.select(["ts", "quantity"]),
            kxinxu.select(["ts", "quantity"]),
        ]
    )
    hour_agg = (
        k_all.with_columns(pl.col("ts").dt.hour().alias("hour"))
        .group_by("hour")
        .agg(pl.len().alias("n_trades"), pl.col("quantity").sum().alias("volume"))
        .sort("hour")
        .to_pandas()
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    rth_mask = (hour_agg["hour"] >= 9) & (hour_agg["hour"] < 16)
    colors = ["#1565c0" if m else "#90caf9" for m in rth_mask]
    ax1.bar(hour_agg["hour"], hour_agg["n_trades"], color=colors, edgecolor="white", linewidth=0.4)
    ax1.set_xlabel("Hour of Day (ET)")
    ax1.set_ylabel("Number of Trades")
    ax1.set_title("Trade Count by Hour of Day\n(dark = RTH 9:30–16:00 ET)")
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(hour_agg["hour"], hour_agg["volume"] / 1e6, color=colors, edgecolor="white", linewidth=0.4)
    ax2.set_xlabel("Hour of Day (ET)")
    ax2.set_ylabel("Volume (millions of contracts)")
    ax2.set_title("Trade Volume by Hour of Day\n(dark = RTH 9:30–16:00 ET)")
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Intraday Trade Activity — All Kalshi Contracts, Jan–Feb 2026", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return hour_agg, rth_mask


def plot_trade_volume_by_contract_type(kxinx, kxinxu):
    kxinx_daily = kxinx.group_by("date").agg(pl.len().alias("trades")).sort("date").to_pandas()
    kxinxu_daily = kxinxu.group_by("date").agg(pl.len().alias("trades")).sort("date").to_pandas()
    kxinx_daily["date"] = pd.to_datetime(kxinx_daily["date"])
    kxinxu_daily["date"] = pd.to_datetime(kxinxu_daily["date"])

    fig, ax = plt.subplots(figsize=(13, 4))
    width = pd.Timedelta(hours=7)
    ax.bar(
        kxinx_daily["date"] - width / 2,
        kxinx_daily["trades"],
        width=width,
        color="#2196F3",
        alpha=0.85,
        label="KXINX (range)",
    )
    ax.bar(
        kxinxu_daily["date"] + width / 2,
        kxinxu_daily["trades"],
        width=width,
        color="#FF5722",
        alpha=0.85,
        label="KXINXU (threshold)",
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    fig.autofmt_xdate(rotation=30)
    ax.set_ylabel("Number of Trades")
    ax.set_title("Daily Trade Volume by Contract Type", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def display_contract_samples(kxinxu, kxinx):
    print("=== KXINXU (threshold) sample ===")
    display(
        kxinxu.select(
            ["ticker", "contract_desc", "threshold", "ts", "price", "quantity", "liquidity_event"]
        )
        .head(3)
        .to_pandas()
    )

    print("\n=== KXINX (range) sample ===")
    display(
        kxinx.select(
            [
                "ticker",
                "contract_desc",
                "bracket_floor",
                "bracket_cap",
                "ts",
                "price",
                "quantity",
                "liquidity_event",
            ]
        )
        .head(3)
        .to_pandas()
    )


def display_summary_table(summary):
    display(summary.set_index("Dataset"))


def display_all_df_overview(all_df, pre_rows, macro_df):
    print(f"Kalshi trade buckets (rows in all_df): {all_df.height:,}  ({pre_rows - all_df.height} dropped after gap-fill)")
    print(f"Macro rows (1s SPX/VIX/SPY):           {macro_df.height:,}")
    print(f"Unique contracts in all_df:             {all_df['contract_id'].n_unique()}")
    print()
    display(all_df.head(5).to_pandas())


def print_market_stats(spx_daily, vix_daily):
    spx_start = spx_daily["spx"].iloc[0]
    spx_end = spx_daily["spx"].iloc[-1]
    spx_ret = (spx_end / spx_start - 1) * 100
    print(
        f"SPX:  start={spx_start:.2f}  end={spx_end:.2f}  return={spx_ret:+.2f}%"
        f"  daily range=[{spx_daily['spx'].min():.2f}, {spx_daily['spx'].max():.2f}]"
    )
    print(
        f"VIX:  min={vix_daily['vix'].min():.2f}  mean={vix_daily['vix'].mean():.2f}"
        f"  max={vix_daily['vix'].max():.2f}"
    )


def print_trade_activity_stats(hour_agg, rth_mask):
    last_hour = hour_agg[hour_agg["hour"] == 15]["n_trades"].values[0]
    other_rth = hour_agg[(hour_agg["hour"] >= 9) & (hour_agg["hour"] < 15)]["n_trades"].mean()
    print(f"Hour 15 (3–4PM ET): {last_hour:,} trades  ({last_hour / other_rth:.1f}x the avg of other RTH hours)")
    print(
        f"After-hours (16:00–09:29): {hour_agg[~rth_mask]['n_trades'].sum():,} trades"
        f"  ({hour_agg[~rth_mask]['n_trades'].sum() / (hour_agg['n_trades'].sum()) * 100:.1f}% of total)"
    )


def plot_intraday_price_convergence(kxinx, spx_close_daily):
    busiest_date = (
        kxinx.group_by("date").agg(pl.len().alias("n")).sort("n", descending=True).head(1)["date"].item()
    )
    spx_close_val = spx_close_daily.filter(pl.col("date") == busiest_date)["spx_close"].item()

    day_rth = (
        kxinx.filter(pl.col("date") == busiest_date)
        .filter((pl.col("ts").dt.hour() > 9) | ((pl.col("ts").dt.hour() == 9) & (pl.col("ts").dt.minute() >= 30)))
        .filter(pl.col("ts").dt.hour() < 16)
        .select(["ts", "bracket_floor", "bracket_cap", "price"])
        .sort("ts")
    )

    atm_floor = float((int(spx_close_val) // 25) * 25)
    all_floors = sorted(day_rth["bracket_floor"].unique().to_list())
    target_floors = sorted(sorted(all_floors, key=lambda f: abs(f - atm_floor))[:5])

    day_pd = day_rth.filter(pl.col("bracket_floor").is_in(target_floors)).to_pandas()
    day_pd["ts"] = pd.to_datetime(day_pd["ts"]).dt.tz_localize(None)

    palette = ["#d32f2f", "#ff7043", "#4CAF50", "#42a5f5", "#1565c0"]
    color_map = {f: c for f, c in zip(target_floors, palette)}

    fig, ax = plt.subplots(figsize=(13, 5))
    for floor in target_floors:
        sub = day_pd[day_pd["bracket_floor"] == floor].sort_values("ts")
        if sub.empty:
            continue
        c = color_map[floor]
        dist = abs(floor - atm_floor)
        direc = "above" if floor >= atm_floor else "below"
        label = f"[{floor:.0f}, {floor + 25:.0f})  ({dist:.0f} pts {direc} close)"
        ax.scatter(sub["ts"], sub["price"], s=5, alpha=0.35, color=c, label=label)
        roll = sub.set_index("ts")["price"].resample("5min").mean().dropna()
        ax.plot(roll.index, roll.values, lw=2, color=c)

    ax.axhline(1.0, color="gray", ls="--", lw=0.9, label="Settles YES ($1.00)")
    ax.axhline(0.0, color="gray", ls=":", lw=0.9, label="Settles NO  ($0.00)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.set_xlabel("Time (ET) — RTH only")
    ax.set_ylabel("Yes-price ($)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(
        f"Intraday KXINX Price Convergence — {busiest_date}\n"
        f"SPX close = {spx_close_val:.2f}  |  5 brackets nearest ATM",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_implied_probability_vs_distance(kxinx_vwap, kxinxu_vwap):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    bin_edges = np.arange(0, 201, 5)
    bin_labels = np.arange(2.5, 200, 5)

    for ax, vwap_df, title, color, otm_filter in [
        (
            axes[0],
            kxinx_vwap,
            "KXINX — Range Brackets (all OTM in dataset)",
            "#2196F3",
            pl.col("distance_from_close").is_not_null() & (pl.col("distance_from_close") >= 0),
        ),
        (
            axes[1],
            kxinxu_vwap,
            "KXINXU — Threshold Contracts (OTM only)",
            "#FF5722",
            (pl.col("side_of_close") == "above")
            & pl.col("distance_from_close").is_not_null()
            & (pl.col("distance_from_close") >= 0),
        ),
    ]:
        df = vwap_df.filter(otm_filter & pl.col("vwap").is_not_null()).to_pandas()
        ax.scatter(df["distance_from_close"], df["vwap"], s=8, alpha=0.25, color=color, rasterized=True)
        df["dist_bin"] = pd.cut(df["distance_from_close"], bins=bin_edges, labels=bin_labels)
        binned = (
            df.groupby("dist_bin", observed=True)["vwap"]
            .mean()
            .reset_index()
            .assign(dist_bin=lambda d: d["dist_bin"].astype(float))
            .dropna()
            .sort_values("dist_bin")
        )
        ax.plot(binned["dist_bin"], binned["vwap"], color="black", lw=2.5, label="Bin mean (5-pt)")
        ax.axvline(0, color="red", ls="--", lw=1, alpha=0.8, label="At-the-money edge")
        ax.set_xlabel("Distance from SPX Close (index pts, OTM side)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(left=0)

    axes[0].set_ylabel("Contract VWAP ($)")
    plt.suptitle("Implied Probability vs Distance from SPX Close", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_pricing_error_by_tte(valid):
    tte_order = ["< 30 min", "30 min\u20131 h", "1\u20132 h", "2\u20134 h", "4\u20138 h", "8\u201324 h"]
    palette_tte = ["#1565c0", "#1976d2", "#42a5f5", "#ff7043", "#e64a19", "#b71c1c"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=False)
    axes = axes.flatten()
    for i, (bucket, color) in enumerate(zip(tte_order, palette_tte)):
        subset = valid[valid["tte_bucket"] == bucket]["error"].dropna()
        if subset.empty:
            continue
        axes[i].hist(subset, bins=40, range=(-0.5, 0.5), color=color, alpha=0.8, density=True)
        axes[i].axvline(0, color="black", lw=1, ls="--")
        axes[i].axvline(subset.mean(), color="red", lw=1.5, ls="-", label=f"mean={subset.mean():.3f}")
        axes[i].set_title(f"{bucket}  (n={len(subset):,})", fontsize=10)
        axes[i].set_xlabel("Market − Model price")
        axes[i].legend(fontsize=8)
        axes[i].grid(alpha=0.3)

    plt.suptitle("Pricing Error (Market − Model) by Time-to-Expiry Bucket — KXINXU", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_model_vs_market(valid):
    fig, ax = plt.subplots(figsize=(7, 6))
    sample = valid.sample(min(5000, len(valid)), random_state=42)
    ax.scatter(sample["model_price"], sample["price"], alpha=0.15, s=6, color="#1565c0", rasterized=True)
    ax.plot([0, 1], [0, 1], color="red", lw=1.5, ls="--", label="Perfect calibration")
    ax.set_xlabel("Model price (Black-Scholes)")
    ax.set_ylabel("Traded price (market)")
    ax.set_title("Model Price vs Traded Price — KXINXU", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def display_pricing_error_summary(valid):
    summary_tbl = (
        valid.groupby("tte_bucket", observed=True)["error"]
        .agg(
            N="count",
            Mean_Error="mean",
            MAE=lambda x: x.abs().mean(),
            Std="std",
        )
        .reset_index()
        .rename(columns={"tte_bucket": "TTE Bucket"})
    )
    summary_tbl[["Mean_Error", "MAE", "Std"]] = summary_tbl[["Mean_Error", "MAE", "Std"]].round(4)
    display(summary_tbl.set_index("TTE Bucket"))
    print(f"\nOverall: N={len(valid):,}  Mean error={valid['error'].mean():.4f}  MAE={valid['error'].abs().mean():.4f}")


def compute_model_vs_market_validation(kxinxu, spx, vix, pricer):
    kxinxu_pd = kxinxu.to_pandas()
    spx_pd2 = spx.to_pandas().sort_values("ts").drop_duplicates("ts")
    vix_pd2 = vix.to_pandas().sort_values("ts").drop_duplicates("ts")
    kxinxu_pd = kxinxu_pd.sort_values("ts")

    kxinxu_pd = pd.merge_asof(kxinxu_pd, spx_pd2[["ts", "spx"]], on="ts", direction="backward")
    kxinxu_pd = pd.merge_asof(kxinxu_pd, vix_pd2[["ts", "vix"]], on="ts", direction="backward")

    def parse_expiry(ticker):
        months = {
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
        m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})H(\d{2})(\d{2})", ticker)
        if not m:
            return None
        yy, mon, dd, hh, mm = m.groups()
        mo = months.get(mon)
        return datetime(2000 + int(yy), mo, int(dd), int(hh), int(mm)) if mo else None

    kxinxu_pd["expiry"] = kxinxu_pd["ticker"].apply(parse_expiry)
    kxinxu_pd["ts_naive"] = kxinxu_pd["ts"].dt.tz_localize(None)
    kxinxu_pd["hours_tte"] = (kxinxu_pd["expiry"] - kxinxu_pd["ts_naive"]).dt.total_seconds() / 3600
    kxinxu_pd["tau"] = kxinxu_pd["hours_tte"] / (365 * 24)

    def model_price(row):
        if pd.isna(row["spx"]) or pd.isna(row["vix"]) or pd.isna(row["tau"]) or row["tau"] <= 0:
            return np.nan
        return pricer.price(contract_id=row["ticker"], spx=row["spx"], vix=row["vix"], ts=row["ts"])

    kxinxu_pd["model_price"] = kxinxu_pd.apply(model_price, axis=1)
    kxinxu_pd["error"] = kxinxu_pd["price"] - kxinxu_pd["model_price"]

    valid = kxinxu_pd.dropna(subset=["error", "hours_tte"])
    valid = valid[(valid["price"] > 0.01) & (valid["price"] < 0.99) & (valid["hours_tte"] > 0)]
    valid["tte_bucket"] = pd.cut(
        valid["hours_tte"],
        bins=[0, 0.5, 1, 2, 4, 8, 24],
        labels=["< 30 min", "30 min\u20131 h", "1\u20132 h", "2\u20134 h", "4\u20138 h", "8\u201324 h"],
    )

    return valid, kxinxu_pd


def run_model_price_vs_market_section(kxinxu, spx, vix, pricer):
    valid, kxinxu_pd = compute_model_vs_market_validation(kxinxu, spx, vix, pricer)
    plot_pricing_error_by_tte(valid)
    plot_model_vs_market(valid)
    display_pricing_error_summary(valid)
    return valid, kxinxu_pd


def plot_pnl_over_time(ts_pd, pnl_ts, label, title):
    plt.figure(figsize=(14, 5))
    plt.plot(ts_pd, pnl_ts, label=label)
    plt.xlabel("Time")
    plt.ylabel("PnL ($)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_performance_metrics(df_ts):
    _rets = df_ts.filter(pl.col("returns").is_finite() & pl.col("returns").is_not_null())["returns"]
    rets_np = _rets.to_numpy()

    mean_return = float(np.mean(rets_np))
    return_variance = float(np.var(rets_np))
    return_std = float(np.std(rets_np))
    sec_per_year = 252 * 6.5 * 3600
    sharpe_ratio = (mean_return / return_std * (sec_per_year**0.5)) if return_std and return_std != 0 else float("nan")
    var_95 = float(np.percentile(rets_np, 5))
    cvar_95 = float(np.mean(rets_np[rets_np <= var_95]))
    return mean_return, return_variance, sharpe_ratio, var_95, cvar_95


def print_performance_metrics(mean_return, return_variance, sharpe_ratio, var_95, cvar_95):
    print("Performance metrics (from 1-second returns):")
    print(f"  Mean Return:     {mean_return:.6f}")
    print(f"  Return Variance: {return_variance:.8f}")
    print(f"  Sharpe Ratio:    {sharpe_ratio:.4f}")
    print(f"  VaR (5%):        {var_95:.6f}")
    print(f"  CVaR (5%):       {cvar_95:.6f}")


def prepare_simulation_timeseries(df):
    df_ts = df.sort("ts").unique(subset=["ts"], keep="last").sort("ts")
    df_ts = df_ts.with_columns((pl.col("take_ask") - pl.col("take_bid")).alias("market_spread"))
    return df_ts


def run_main_results_overview(df):
    df_ts = prepare_simulation_timeseries(df)
    ts_pd = df_ts["ts"].to_pandas()
    pnl_ts = df_ts["pnl"].to_pandas()
    plot_pnl_over_time(ts_pd, pnl_ts, label="PnL", title="PnL Over Time")
    mean_return, return_variance, sharpe_ratio, var_95, cvar_95 = compute_performance_metrics(df_ts)
    print_performance_metrics(mean_return, return_variance, sharpe_ratio, var_95, cvar_95)
    return df_ts, ts_pd, pnl_ts, (mean_return, return_variance, sharpe_ratio, var_95, cvar_95)


def run_variant_results_overview(df_variant, label, title):
    df_variant_sorted = df_variant.sort("ts")
    if not isinstance(df_variant_sorted["ts"][0], (str,)):
        ts_variant = df_variant_sorted["ts"].to_pandas()
    else:
        ts_variant = pl.Series(df_variant_sorted["ts"]).to_pandas()
    pnl_variant = df_variant_sorted["pnl"]

    plot_pnl_over_time(ts_variant, pnl_variant, label=label, title=title)

    df_ts_variant = prepare_simulation_timeseries(df_variant)
    mean_return, return_variance, sharpe_ratio, var_95, cvar_95 = compute_performance_metrics(df_ts_variant)
    print_performance_metrics(mean_return, return_variance, sharpe_ratio, var_95, cvar_95)
    return df_ts_variant, ts_variant, pnl_variant, (mean_return, return_variance, sharpe_ratio, var_95, cvar_95)


def plot_rolling_drawdown(df_ts):
    pnl_ts = df_ts["pnl"]
    cummax_1s = pnl_ts.cum_max()
    dd_1s = cummax_1s - pnl_ts
    ts_1s = df_ts["ts"]

    hourly_df = (
        df_ts.sort("ts")
        .with_columns(pl.col("ts").dt.truncate("1h").alias("ts_hour"))
        .group_by("ts_hour")
        .agg(pl.col("portfolio_value").last().alias("close"))
        .sort("ts_hour")
    )
    hourly_df = hourly_df.with_columns((pl.col("close") / pl.col("close").shift(1) - 1).alias("hourly_return")).filter(
        pl.col("hourly_return").is_not_null()
    )
    hourly_df = hourly_df.with_columns((1 + pl.col("hourly_return")).cum_prod().alias("cum_wealth")).with_columns(
        (pl.col("cum_wealth").cum_max() - pl.col("cum_wealth")).alias("dd_1h")
    )
    ts_1h = hourly_df["ts_hour"]
    dd_1h = hourly_df["dd_1h"]

    daily_df = (
        df_ts.sort("ts")
        .with_columns(pl.col("ts").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.col("portfolio_value").last().alias("close"))
        .sort("date")
    )
    daily_df = daily_df.with_columns((pl.col("close") / pl.col("close").shift(1) - 1).alias("daily_return")).filter(
        pl.col("daily_return").is_not_null()
    )
    daily_df = daily_df.with_columns((1 + pl.col("daily_return")).cum_prod().alias("cum_wealth")).with_columns(
        (pl.col("cum_wealth").cum_max() - pl.col("cum_wealth")).alias("dd_1d")
    )
    ts_1d = daily_df["date"]
    dd_1d = daily_df["dd_1d"]

    initial_value = df_ts.sort("ts").select(pl.col("portfolio_value").first()).item()
    dd_1h_dollars = dd_1h * initial_value
    dd_1d_dollars = dd_1d * initial_value

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
    axes[0].fill_between(ts_1s.to_pandas(), 0, dd_1s.to_numpy(), color="steelblue", alpha=0.7)
    axes[0].set_ylabel("Drawdown ($)")
    axes[0].set_title("Rolling Drawdown (1 sec, from PnL)")
    axes[0].grid(True, alpha=0.3)
    axes[1].fill_between(ts_1h.to_pandas(), 0, dd_1h_dollars.to_numpy(), color="coral", alpha=0.7)
    axes[1].set_ylabel("Drawdown ($)")
    axes[1].set_title("Rolling Drawdown from Hourly Returns")
    axes[1].grid(True, alpha=0.3)
    axes[2].fill_between(ts_1d.to_pandas(), 0, dd_1d_dollars.to_numpy(), color="seagreen", alpha=0.7)
    axes[2].set_ylabel("Drawdown ($)")
    axes[2].set_title("Rolling Drawdown from Daily Returns")
    axes[2].set_xlabel("Time")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_inventory_and_spy_position(ts_pd, df_ts):
    plt.figure(figsize=(14, 4))
    plt.plot(ts_pd, df_ts["total_kalshi_inventory"].to_numpy(), color="steelblue", label="Kalshi inventory")
    plt.ylabel("Kalshi inventory (contracts)")
    plt.xlabel("Time")
    plt.title("Kalshi Inventory Over Time")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 4))
    plt.plot(ts_pd, df_ts["pos_spy"].to_numpy(), color="coral", label="SPY position")
    plt.ylabel("SPY position (shares)")
    plt.xlabel("Time")
    plt.title("SPY Position Over Time")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_max_unique_kalshi_contracts_held(sim_df):
    contracts_per_ts = (
        sim_df.filter(pl.col("pos_kalshi").fill_null(0) != 0)
        .group_by("ts")
        .agg(pl.col("contract_id").n_unique().alias("num_unique_contracts"))
    )

    max_unique_contracts = contracts_per_ts["num_unique_contracts"].max()
    print("Most unique Kalshi contracts held at once:", max_unique_contracts)

    max_times = contracts_per_ts.filter(pl.col("num_unique_contracts") == max_unique_contracts)
    print("Timestamp(s) with max unique contracts held:\n", max_times)
    return max_unique_contracts, max_times, contracts_per_ts


def plot_total_kalshi_trade_volume_by_hour(df):
    df_hour = df.with_columns(pl.col("ts").dt.hour().alias("hour"))
    vol = df_hour.filter(pl.col("bid_fill") | pl.col("ask_fill")).with_columns(
        (
            pl.col("my_bid_size") * pl.col("bid_fill").cast(pl.Int64)
            + pl.col("my_ask_size") * pl.col("ask_fill").cast(pl.Int64)
        ).alias("fill_qty")
    )
    hourly_vol = vol.group_by("hour").agg(pl.col("fill_qty").sum().alias("volume")).sort("hour")
    plt.figure(figsize=(10, 4))
    plt.bar(
        hourly_vol["hour"].to_numpy(),
        hourly_vol["volume"].to_numpy(),
        color="steelblue",
        edgecolor="navy",
        alpha=0.8,
    )
    plt.xlabel("Hour of day (ET)")
    plt.ylabel("Total Kalshi trade volume (contracts)")
    plt.title("Total Kalshi Trade Volume by Hour of Day")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()
    return hourly_vol


def plot_cumulative_trade_count_histogram(df, bins=60):
    trade_ts = pd.to_datetime(df["ts"].to_pandas())
    plt.figure(figsize=(13, 4))
    plt.hist(
        trade_ts,
        bins=bins,
        cumulative=True,
        color="steelblue",
        alpha=0.8,
        edgecolor="white",
    )
    plt.xlabel("Time")
    plt.ylabel("Cumulative Number of Trades")
    plt.title("Cumulative Number of Trades Over Entire Period")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.show()
    return trade_ts


def plot_average_pnl_by_hour(df):
    if "pnl" not in df.columns:
        raise ValueError("No 'pnl' column found in trades_df—you may need to join or compute it first.")

    pnl_by_hour = (
        df.with_columns(pl.col("ts").dt.hour().alias("hour_of_day"))
        .group_by("hour_of_day")
        .agg(pl.col("pnl").mean().alias("avg_pnl"))
        .sort("hour_of_day")
    )
    pnl_pdf = pnl_by_hour.to_pandas()

    plt.figure(figsize=(13, 6))
    plt.plot(pnl_pdf["hour_of_day"], pnl_pdf["avg_pnl"], marker="o", color="C1", label="Average PnL")
    plt.title("Average PnL by Hour of Day (US EST, 1hr bins)")
    plt.xlabel("Hour of Day (EST)")
    plt.ylabel("Average PnL")
    plt.xticks(sorted(pnl_pdf["hour_of_day"].unique()))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.show()
    return pnl_by_hour, pnl_pdf


def plot_average_spread_by_hour(df):
    if not all(col in df.columns for col in ["ts", "my_bid", "my_ask"]):
        raise ValueError("Expected columns 'ts', 'my_bid', 'my_ask' in sim_df.")

    spread_by_hour = (
        df.with_columns(
            [
                (pl.col("my_ask") - pl.col("my_bid")).alias("spread"),
                pl.col("ts").dt.hour().alias("hour_of_day"),
            ]
        )
        .group_by("hour_of_day")
        .agg(pl.col("spread").mean().alias("avg_spread"))
        .sort("hour_of_day")
    )
    spread_pdf = spread_by_hour.to_pandas()

    plt.figure(figsize=(13, 6))
    plt.plot(
        spread_pdf["hour_of_day"],
        spread_pdf["avg_spread"],
        marker="o",
        color="C2",
        label="Average Spread (ask-bid)",
    )
    plt.title("Average Market Spread by Hour of Day (US EST, 1hr bins)")
    plt.xlabel("Hour of Day (EST)")
    plt.ylabel("Average Spread")
    plt.xticks(sorted(spread_pdf["hour_of_day"].unique()))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.show()
    return spread_by_hour, spread_pdf


def plot_spread_for_date_est(df, target_date="2026-01-15"):
    jan6_df = df.filter(pl.col("ts").dt.date().cast(str) == target_date)

    if jan6_df.height == 0:
        print("No data available for Jan 6th.")
    else:
        jan6_pdf = jan6_df.select(
            [
                pl.col("ts"),
                (pl.col("my_ask") - pl.col("my_bid")).alias("spread"),
            ]
        ).to_pandas()

        if jan6_pdf["ts"].dt.tz is None or str(jan6_pdf["ts"].dt.tz) != "America/New_York":
            jan6_pdf["ts"] = jan6_pdf["ts"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")

        plt.figure(figsize=(14, 6))
        plt.plot(
            jan6_pdf["ts"],
            jan6_pdf["spread"],
            marker="o",
            linestyle="-",
            color="C3",
            label="Spread (ask-bid)",
        )
        plt.title("Market Spread on January 6th, 2026 (EST)")
        plt.xlabel("Time (EST)")
        plt.ylabel("Spread")
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.HourLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=pytz.timezone("America/New_York")))
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


def compute_and_plot_returns_over_time(df_ts):
    df_ts_pd = df_ts.to_pandas()
    df_ts_pd = df_ts_pd.set_index("ts")
    rets_1s = df_ts_pd["returns"].dropna()
    rets_1h = rets_1s.resample("1h").sum()
    rets_1d = rets_1s.resample("1D").sum()

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=False)
    axes[0].plot(rets_1s.index, rets_1s.values, linewidth=0.3, alpha=0.8, color="steelblue")
    axes[0].set_ylabel("Return")
    axes[0].set_title("1-Second Returns Over Time")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(rets_1h.index, rets_1h.values, linewidth=0.8, color="coral")
    axes[1].set_ylabel("Return")
    axes[1].set_title("Hourly Returns Over Time")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(rets_1d.index, rets_1d.values, linewidth=1, color="seagreen")
    axes[2].set_ylabel("Return")
    axes[2].set_title("Daily Returns Over Time")
    axes[2].set_xlabel("Time")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return rets_1s, rets_1h, rets_1d


def plot_return_histograms_with_percentiles(rets_1s, rets_1h, rets_1d):
    p5_1s = np.percentile(rets_1s.dropna(), 5)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.hist(rets_1s.dropna(), bins=80, color="steelblue", alpha=0.8, edgecolor="white", density=True)
    ax.axvline(p5_1s, color="red", linestyle="--", linewidth=2, label=f"5th %ile = {p5_1s:.4f}")
    ax.set_xlabel("1-sec return")
    ax.set_ylabel("Density")
    ax.set_title("Second Returns – Histogram with 5th Percentile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def correlation_with_spx_and_scatter(df_ts, rets_1d):
    spx_vals = df_ts["spx"].to_numpy()
    spx_rets = np.diff(spx_vals) / spx_vals[:-1]
    strat_rets = df_ts["returns"].to_numpy()[1:]
    valid = np.isfinite(spx_rets) & np.isfinite(strat_rets)
    corr_1s = np.corrcoef(strat_rets[valid], spx_rets[valid])[0, 1]

    strat_daily = rets_1d.dropna()
    spx_daily = df_ts.to_pandas().set_index("ts")["spx"].resample("1D").last().pct_change()
    common_idx = strat_daily.index.intersection(spx_daily.index)
    strat_d = strat_daily.reindex(common_idx).dropna()
    spx_d = spx_daily.reindex(common_idx).dropna()
    valid = strat_d.notna() & spx_d.notna()
    s1 = strat_d[valid].values
    s2 = spx_d[valid].values
    corr_d = np.corrcoef(s1, s2)[0, 1] if len(s1) > 1 else np.nan

    print("Correlation with SPX returns:")
    print(f"  1-second (aligned): {corr_1s:.4f}")
    print(f"  Daily:             {corr_d:.4f}")

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(s2, s1, alpha=0.6, s=20)
    ax.set_xlabel("SPX daily return")
    ax.set_ylabel("Strategy daily return")
    ax.set_title(f"Strategy vs SPX Daily Returns (corr = {corr_d:.3f})")
    ax.axhline(0, color="gray", linestyle="--")
    ax.axvline(0, color="gray", linestyle="--")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return corr_1s, corr_d


def correlation_with_spx_and_scatter_no_hedge(df_ts_no_hedge):
    spx_vals = df_ts_no_hedge.select("spx").to_numpy().flatten()
    spx_rets = np.diff(spx_vals) / spx_vals[:-1]
    strat_rets = df_ts_no_hedge.select("returns").to_numpy().flatten()[1:]
    valid = np.isfinite(spx_rets) & np.isfinite(strat_rets)
    corr_1s = np.corrcoef(strat_rets[valid], spx_rets[valid])[0, 1]

    df_ts_no_hedge_daily = (
        df_ts_no_hedge.with_columns(pl.col("ts").cast(pl.Datetime("us")))
        .group_by_dynamic("ts", every="1d", closed="right")
        .agg(
            [
                pl.last("spx").alias("spx_last"),
                pl.last("returns").alias("strat_ret_last"),
            ]
        )
        .sort("ts")
    )

    spx_daily = df_ts_no_hedge_daily.select("spx_last").to_numpy().flatten()
    strat_daily = df_ts_no_hedge_daily.select("strat_ret_last").to_numpy().flatten()

    spx_d_rets = np.diff(spx_daily) / spx_daily[:-1]
    strat_d_rets = strat_daily[1:]
    valid = np.isfinite(spx_d_rets) & np.isfinite(strat_d_rets)
    s2 = spx_d_rets[valid]
    s1 = strat_d_rets[valid]
    corr_d = np.corrcoef(s1, s2)[0, 1] if len(s1) > 1 else np.nan

    print("Correlation with SPX returns (No Hedge):")
    print(f"  1-second (aligned): {corr_1s:.4f}")
    print(f"  Daily:             {corr_d:.4f}")

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(s2, s1, alpha=0.6, s=20)
    ax.set_xlabel("SPX daily return")
    ax.set_ylabel("Strategy daily return")
    ax.set_title(f"Strategy vs SPX Daily Returns (No Hedge, corr = {corr_d:.3f})")
    ax.axhline(0, color="gray", linestyle="--")
    ax.axvline(0, color="gray", linestyle="--")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return corr_1s, corr_d

    p5_1h = np.percentile(rets_1h.dropna(), 5)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.hist(rets_1h.dropna(), bins=50, color="coral", alpha=0.8, edgecolor="white", density=True)
    ax.axvline(p5_1h, color="red", linestyle="--", linewidth=2, label=f"5th %ile = {p5_1h:.4f}")
    ax.set_xlabel("Hourly return")
    ax.set_ylabel("Density")
    ax.set_title("Hourly Returns – Histogram with 5th Percentile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    p5_1d = np.percentile(rets_1d.dropna(), 5)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.hist(rets_1d.dropna(), bins=40, color="seagreen", alpha=0.8, edgecolor="white", density=True)
    ax.axvline(p5_1d, color="red", linestyle="--", linewidth=2, label=f"5th %ile = {p5_1d:.4f}")
    ax.set_xlabel("Daily return")
    ax.set_ylabel("Density")
    ax.set_title("Daily Returns – Histogram with 5th Percentile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ── Section 5: Performance Analysis Functions ─────────────────────────────────

def portfolio_ts(dframe):
    """One row per second (last state per timestamp)."""
    return dframe.sort("ts").unique(subset=["ts"], keep="last").sort("ts")


def perf_stats(dframe, label, rf_annual=0.0425):
    ts   = portfolio_ts(dframe)
    pv   = ts["portfolio_value"].to_numpy()
    t    = ts["ts"].to_pandas()
    rets = np.diff(pv) / INITIAL_CAPITAL        # return on initial capital per second
    rf_s = rf_annual / ANN_SECONDS

    # Sharpe
    excess  = rets - rf_s
    sharpe  = excess.mean() / excess.std() * np.sqrt(ANN_SECONDS) if excess.std() > 0 else np.nan

    # Sortino (downside deviation only)
    neg     = excess[excess < 0]
    sortino = excess.mean() / neg.std() * np.sqrt(ANN_SECONDS) if len(neg) > 0 and neg.std() > 0 else np.nan

    # Max Drawdown
    run_max       = np.maximum.accumulate(pv)
    max_dd_pct    = ((run_max - pv) / run_max).max()
    max_dd_dollar = (run_max - pv).max()

    # Annualized return (trading-second basis)
    ann_ret = (pv[-1] / pv[0]) ** (ANN_SECONDS / len(pv)) - 1

    # Calmar = annualized return / max drawdown %
    calmar  = ann_ret / max_dd_pct if max_dd_pct > 0 else np.nan

    # CAGR (calendar-day basis)
    cal_days = (t.iloc[-1] - t.iloc[0]).total_seconds() / 86_400
    cagr     = (pv[-1] / pv[0]) ** (365 / cal_days) - 1 if cal_days > 0 else np.nan

    return {
        "Scenario"      : label,
        "Total PnL ($)" : f"${pv[-1] - INITIAL_CAPITAL:,.2f}",
        "Period Return" : f"{(pv[-1]/pv[0]-1)*100:.2f}%",
        "Ann. Return"   : f"{ann_ret*100:.1f}%",
        "Sharpe"        : f"{sharpe:.3f}",
        "Sortino"       : f"{sortino:.3f}",
        "Calmar"        : f"{calmar:.3f}",
        "Max DD ($)"    : f"${max_dd_dollar:,.2f}",
        "Max DD (%)"    : f"{max_dd_pct*100:.2f}%",
    }


def final_pnl(dframe):
    ts = portfolio_ts(dframe)
    return ts["portfolio_value"][-1] - INITIAL_CAPITAL


def var_cvar(dframe, label, confidences=(0.95, 0.99)):
    ts   = portfolio_ts(dframe)
    rets = np.diff(ts["portfolio_value"].to_numpy())    # dollar P&L per second
    row  = {"Scenario": label}
    for c in confidences:
        q    = np.percentile(rets, (1 - c) * 100)
        cvar = rets[rets <= q].mean()
        row[f"VaR  {int(c*100)}% ($)"]  = f"${q:,.3f}"
        row[f"CVaR {int(c*100)}% ($)"] = f"${cvar:,.3f}"
    return row


def add_regime(dframe):
    return dframe.with_columns(
        pl.when(pl.col("vix") < 17).then(pl.lit("Low (<17)"))
          .when(pl.col("vix") <= 20).then(pl.lit("Med (17-20)"))
          .otherwise(pl.lit("High (>20)"))
          .alias("regime")
    )


def get_daily_sharpe(dframe, rf_annual=0.0425):
    """Per-trading-day Sharpe ratio."""
    ts   = portfolio_ts(dframe)
    pv   = ts["portfolio_value"].to_numpy()
    t    = ts["ts"].to_pandas()
    rets = pd.Series(np.diff(pv) / INITIAL_CAPITAL, index=t.iloc[1:].values)
    rf_s = rf_annual / ANN_SECONDS
    exc  = rets - rf_s
    results = {}
    for day, grp in exc.groupby(exc.index.date):
        grp = grp.dropna()
        if len(grp) < 10: continue
        s = grp.mean() / grp.std() * np.sqrt(ANN_SECONDS) if grp.std() > 0 else np.nan
        results[pd.Timestamp(day)] = s
    return pd.Series(results).dropna()


def get_hourly_rolling_sharpe(dframe, window=8, rf_annual=0.0425):
    """Rolling Sharpe on hourly-resampled portfolio value (row-count window)."""
    ts     = portfolio_ts(dframe)
    pv     = ts["portfolio_value"].to_numpy()
    t      = ts["ts"].to_pandas()
    pv_h   = pd.Series(pv, index=t).resample("1h").last().dropna()
    rets_h = pv_h.pct_change().dropna()
    rf_h   = rf_annual / (252 * 6.5)     # hourly risk-free rate
    exc_h  = rets_h - rf_h
    ann_h  = np.sqrt(252 * 6.5)          # annualise from hourly
    roll   = exc_h.rolling(window, min_periods=3)
    return (roll.mean() / roll.std() * ann_h).dropna()


def _to_pd_datetime(s):
    # Already datetime-like: leave it alone
    if pd.api.types.is_datetime64_any_dtype(s):
        return s

    # Numeric epoch: assume ns coming from Polars/Python timestamp pipeline
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="ns")

    # Strings / python datetimes / object dtype
    return pd.to_datetime(s)

def adverse_deltas(offset_secs, all_df, fills_df):
    """
    Accepts either pandas or polars DataFrames.
    Returns numpy array of favorable deltas.
    """

    # Convert to pandas if needed
    all_pd = all_df.to_pandas().copy() if isinstance(all_df, pl.DataFrame) else all_df.copy()
    fills_pd = fills_df.to_pandas().copy() if isinstance(fills_df, pl.DataFrame) else fills_df.copy()

    # Robust timestamp conversion
    all_pd["ts"] = _to_pd_datetime(all_pd["ts"])
    fills_pd["ts"] = _to_pd_datetime(fills_pd["ts"])

    # Build future lookup time
    fills_pd["future_ts"] = fills_pd["ts"] + pd.Timedelta(seconds=offset_secs)

    # Sort for merge_asof
    all_pd = all_pd.sort_values(["contract_id", "ts"]).reset_index(drop=True)
    fills_pd = fills_pd.sort_values(["contract_id", "future_ts"]).reset_index(drop=True)

    rhs = (
        all_pd[["contract_id", "ts", "mid"]]
        .rename(columns={"ts": "lookup_ts", "mid": "future_mid"})
        .sort_values(["contract_id", "lookup_ts"])
        .reset_index(drop=True)
    )

    # sort keys
    fills_pd["contract_id"] = fills_pd["contract_id"].astype(str)
    rhs["contract_id"] = rhs["contract_id"].astype(str)

    fills_pd["future_ts"] = _to_pd_datetime(fills_pd["future_ts"]).dt.as_unit("ns")
    rhs["lookup_ts"] = _to_pd_datetime(rhs["lookup_ts"]).dt.as_unit("ns")

    fills_pd = fills_pd.sort_values(["future_ts", "contract_id"]).reset_index(drop=True)
    rhs = rhs.sort_values(["lookup_ts", "contract_id"]).reset_index(drop=True)

    merged = pd.merge_asof(
        fills_pd,
        rhs,
        left_on="future_ts",
        right_on="lookup_ts",
        by="contract_id",
        direction="forward",
        tolerance=pd.Timedelta(seconds=offset_secs * 3),
    )

    merged = merged.dropna(subset=["future_mid"]).copy()
    merged["delta"] = merged["future_mid"] - merged["mid"]

    # ask_fill => sold => favorable if price falls
    # bid_fill => bought => favorable if price rises
    merged["favorable"] = np.where(
        merged["ask_fill"],
        -merged["delta"],
        merged["delta"],
    )

    return merged["favorable"].to_numpy()


def parse_expiry(ticker):
    """Parse expiry datetime from a KXINXU ticker string."""
    months = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,
              "AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})H(\d{2})(\d{2})", ticker)
    if not m: return None
    yy, mon, dd, hh, mm = m.groups()
    mo = months.get(mon)
    return datetime(2000+int(yy), mo, int(dd), int(hh), int(mm)) if mo else None


def model_price(row, pricer):
    """Compute Black-Scholes model price for a single KXINXU row."""
    if pd.isna(row["spx"]) or pd.isna(row["vix"]) or pd.isna(row["tau"]) or row["tau"] <= 0:
        return np.nan
    return pricer.price(contract_id=row["ticker"], spx=row["spx"], vix=row["vix"], ts=row["ts"])


def display_performance_summary_table(df, df_no_fees, df_no_hedge):
    rows = [
        perf_stats(df, "Base (fees + hedge)"),
        perf_stats(df_no_fees, "No Fees (hedge on)"),
        perf_stats(df_no_hedge, "No Hedge (fees on)"),
    ]
    summary = pd.DataFrame(rows).set_index("Scenario")
    display(summary.T)
    return summary


def plot_pnl_waterfall_and_attribution(df, df_no_fees, df_no_hedge):
    pnl_base = final_pnl(df)
    pnl_no_fees = final_pnl(df_no_fees)
    pnl_no_hedge = final_pnl(df_no_hedge)

    fee_drag = pnl_no_fees - pnl_base
    hedge_drag = pnl_no_hedge - pnl_base

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    stages = ["Gross\n(no fees)", "−Fee Drag", "Net PnL\n(base)"]
    starts = [0, pnl_no_fees, 0]
    heights = [pnl_no_fees, -fee_drag, pnl_base]
    colors = ["#2196F3", "#F44336", "#4CAF50"]

    for i, (lbl, s, h, c) in enumerate(zip(stages, starts, heights, colors)):
        ax1.bar(lbl, h, bottom=s, color=c, edgecolor="white", linewidth=1.2, width=0.5)
        sign = "+" if h >= 0 else ""
        ax1.text(i, s + h + 15, f"{sign}${h:,.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_ylabel("PnL ($)")
    ax1.set_title("Waterfall: Gross → Fees → Net", fontweight="bold")
    ax1.set_ylim(-100, pnl_no_fees * 1.2)

    labels = ["No Fees\n(hedge on)", "No Hedge\n(fees on)", "Base\n(fees+hedge)"]
    vals = [pnl_no_fees, pnl_no_hedge, pnl_base]
    clrs = ["#2196F3", "#FF9800", "#4CAF50"]

    bars = ax2.bar(labels, vals, color=clrs, edgecolor="white", linewidth=1.2, width=0.5)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 15, f"${v:,.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Total PnL ($)")
    ax2.set_title("Scenario Comparison", fontweight="bold")
    ax2.set_ylim(-100, max(vals) * 1.2)

    hedge_label = f"Hedge drag: −${hedge_drag:,.0f}" if hedge_drag > 0 else f"Hedge benefit: +${-hedge_drag:,.0f}"
    fig.text(0.5, -0.03, f"Fee cost: −${fee_drag:,.0f}    |    {hedge_label}", ha="center", fontsize=10, color="gray")
    fig.suptitle("PnL Waterfall & Scenario Attribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return pnl_base, pnl_no_fees, pnl_no_hedge, fee_drag, hedge_drag


def display_var_cvar_and_pnl_distribution(df, df_no_fees, df_no_hedge):
    tbl = pd.DataFrame([
        var_cvar(df, "Base (fees + hedge)"),
        var_cvar(df_no_fees, "No Fees (hedge on)"),
        var_cvar(df_no_hedge, "No Hedge (fees on)"),
    ]).set_index("Scenario")
    display(tbl)

    ts_b = portfolio_ts(df)
    rets_dol = np.diff(ts_b["portfolio_value"].to_numpy())
    var95 = np.percentile(rets_dol, 5)
    var99 = np.percentile(rets_dol, 1)
    x_lo, x_hi = np.percentile(rets_dol, [0.5, 99.5])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(rets_dol, bins=100, color="#2196F3", alpha=0.7, edgecolor="none", range=(x_lo, x_hi))
    ax.axvline(var95, color="orange", linestyle="--", linewidth=1.5, label=f"VaR 95%: ${var95:,.3f}")
    ax.axvline(var99, color="red", linestyle="--", linewidth=1.5, label=f"VaR 99%: ${var99:,.3f}")
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("Per-Second Dollar P&L ($)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Per-Second P&L — Base Scenario", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.show()

    n_clipped = ((rets_dol < x_lo) | (rets_dol > x_hi)).sum()
    print(f"{n_clipped} outlier seconds outside view  |  full range: ${rets_dol.min():,.2f} to ${rets_dol.max():,.2f}")
    return tbl, ts_b, rets_dol, var95, var99, x_lo, x_hi, n_clipped


def plot_sharpe_ratio_analysis(df, df_no_fees, df_no_hedge):
    sh_d_base = get_daily_sharpe(df)
    sh_d_nf = get_daily_sharpe(df_no_fees)
    sh_d_nh = get_daily_sharpe(df_no_hedge)
    sh_r_base = get_hourly_rolling_sharpe(df)
    sh_r_nf = get_hourly_rolling_sharpe(df_no_fees)
    sh_r_nh = get_hourly_rolling_sharpe(df_no_hedge)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8))

    clrs = ["#4CAF50" if v > 0 else "#F44336" for v in sh_d_base.values]
    x = np.arange(len(sh_d_base))
    ax1.bar(x, sh_d_base.values, color=clrs, alpha=0.85, width=0.75)
    step = max(1, len(sh_d_base) // 10)
    tick_locs = x[::step]
    tick_lbls = [d.strftime("%b %d") for d in sh_d_base.index[::step]]
    ax1.set_xticks(tick_locs)
    ax1.set_xticklabels(tick_lbls, rotation=30, ha="right", fontsize=8)
    ax1.axhline(0, color="black", lw=0.8, linestyle="--")
    pos_days = (sh_d_base > 0).sum()
    ax1.set_ylabel("Daily Sharpe")
    ax1.set_title(f"Daily Sharpe Ratio — Base Scenario  ({pos_days}/{len(sh_d_base)} positive days)", fontweight="bold")

    ax2.plot(sh_r_base.index, sh_r_base.values, color="#2196F3", lw=0.9, label="Base (fees+hedge)", alpha=0.9)
    ax2.plot(sh_r_nf.index, sh_r_nf.values, color="#4CAF50", lw=0.9, label="No Fees", alpha=0.8)
    ax2.plot(sh_r_nh.index, sh_r_nh.values, color="#FF9800", lw=0.9, label="No Hedge", alpha=0.8)
    ax2.axhline(0, color="black", lw=0.8, linestyle="--")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.set_ylabel("Rolling Sharpe (8-hour window)")
    ax2.set_title("Rolling 8-Trading-Hour Sharpe — All Scenarios", fontweight="bold")
    ax2.legend()

    fig.suptitle("Sharpe Ratio Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return sh_d_base, sh_d_nf, sh_d_nh, sh_r_base, sh_r_nf, sh_r_nh


def display_vix_regime_analysis(df, df_ts):
    ts_reg = add_regime(df_ts).with_columns(pl.col("pnl").diff().alias("pnl_chg"))
    regime_pnl = (
        ts_reg.group_by("regime")
        .agg([pl.col("pnl_chg").sum().alias("pnl"), pl.col("vix").mean().alias("avg_vix")])
        .sort("avg_vix")
    )

    df_reg = add_regime(df).with_columns([
        (pl.col("bid_fill") | pl.col("ask_fill")).alias("any_fill"),
        (pl.col("my_bid").is_not_null() & (pl.col("my_bid") > 0)).alias("quoting"),
    ])
    fill_reg = (
        df_reg.group_by("regime")
        .agg([
            pl.col("any_fill").sum().alias("fills"),
            pl.col("quoting").sum().alias("quoted"),
            (pl.col("my_ask") - pl.col("my_bid")).filter(pl.col("my_bid") > 0).mean().alias("avg_spread"),
            pl.col("vix").mean().alias("avg_vix"),
        ])
        .with_columns((pl.col("fills") / pl.col("quoted") * 100).alias("fill_rate_pct"))
        .sort("avg_vix")
    )

    reg_sum = (
        fill_reg.join(regime_pnl.select(["regime", "pnl"]), on="regime")
        .select(["regime", "fills", "fill_rate_pct", "avg_spread", "pnl", "avg_vix"])
    ).to_pandas().set_index("regime")
    reg_sum.columns = ["Fills", "Fill Rate (%)", "Avg Spread", "PnL ($)", "Avg VIX"]
    display(reg_sum.round(3))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    pal = ["#4CAF50", "#FF9800", "#F44336"]
    regs = reg_sum.index.tolist()

    axes[0].bar(regs, reg_sum["Fill Rate (%)"], color=pal)
    axes[0].set_title("Fill Rate (%) by VIX Regime", fontweight="bold")
    axes[0].set_ylabel("Fill Rate (%)")

    axes[1].bar(regs, reg_sum["Avg Spread"], color=pal)
    axes[1].set_title("Avg Quoted Spread by VIX Regime", fontweight="bold")
    axes[1].set_ylabel("Spread (prob. pts)")

    axes[2].bar(regs, reg_sum["PnL ($)"], color=pal)
    axes[2].set_title("Cumulative PnL ($) by VIX Regime", fontweight="bold")
    axes[2].set_ylabel("PnL ($)")

    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    plt.suptitle("VIX Regime Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return ts_reg, regime_pnl, df_reg, fill_reg, reg_sum


def plot_fill_rate_and_adverse_selection(df):
    df_fill = df.with_columns([
        pl.col("ts").dt.date().alias("date"),
        (pl.col("bid_fill") | pl.col("ask_fill")).alias("any_fill"),
        (pl.col("my_bid").is_not_null() & (pl.col("my_bid") > 0)).alias("quoting"),
    ])
    daily_fill = (
        df_fill.group_by("date")
        .agg([pl.col("any_fill").sum().alias("fills"), pl.col("quoting").sum().alias("quoted")])
        .with_columns((pl.col("fills") / pl.col("quoted") * 100).alias("fill_rate_pct"))
        .sort("date")
    ).to_pandas()

    fig1, ax1 = plt.subplots(figsize=(13, 3))
    ax1.bar(daily_fill["date"].astype(str), daily_fill["fill_rate_pct"], color="#4CAF50", alpha=0.85)
    ax1.set_ylabel("Fill Rate (%)")
    ax1.set_xlabel("Date")
    ax1.set_title("Daily Fill Rate — % of Quoted Rows Resulting in a Fill", fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    plt.show()

    print(f"Overall fill rate: {daily_fill['fill_rate_pct'].mean():.2f}%  |  " f"Total fills: {daily_fill['fills'].sum():,}")

    df_mid = df.with_columns(
        pl.when(pl.col("take_bid").is_not_null() & pl.col("take_ask").is_not_null())
        .then((pl.col("take_bid") + pl.col("take_ask")) / 2)
        .otherwise((pl.col("my_bid") + pl.col("my_ask")) / 2)
        .alias("mid")
    )
    all_pd = df_mid.select(["ts", "contract_id", "mid"]).to_pandas()
    fills_pd = df_mid.filter(pl.col("bid_fill") | pl.col("ask_fill")).select(
        ["ts", "contract_id", "mid", "bid_fill", "ask_fill"]
    ).to_pandas()

    print("Computing post-fill adverse selection...")
    d1 = adverse_deltas(1, all_pd, fills_pd)
    d5 = adverse_deltas(5, all_pd, fills_pd)
    d30 = adverse_deltas(30, all_pd, fills_pd)
    print("Done.")

    labels = ["+1 second", "+5 seconds", "+30 seconds"]
    datasets = [d1, d5, d30]

    fig2, (ax_bar, ax_hist) = plt.subplots(1, 2, figsize=(13, 5))

    for_pct = [(d > 0).mean() * 100 for d in datasets]
    zero_pct = [(d == 0).mean() * 100 for d in datasets]
    adv_pct = [(d < 0).mean() * 100 for d in datasets]

    x = np.arange(len(labels))
    width = 0.5
    ax_bar.bar(x, for_pct, width, label="Favorable (moved our way)", color="#4CAF50", alpha=0.85)
    ax_bar.bar(x, zero_pct, width, bottom=for_pct, label="No change", color="#9E9E9E", alpha=0.7)
    ax_bar.bar(
        x,
        adv_pct,
        width,
        bottom=[f + z for f, z in zip(for_pct, zero_pct)],
        label="Adverse (moved against us)",
        color="#F44336",
        alpha=0.85,
    )

    for bar, f, z, a in zip(x, for_pct, zero_pct, adv_pct):
        if f > 1:
            ax_bar.text(bar, f / 2, f"{f:.1f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        if z > 2:
            ax_bar.text(bar, f + z / 2, f"{z:.1f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        if a > 1:
            ax_bar.text(
                bar,
                f + z + a / 2,
                f"{a:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylabel("% of fills")
    ax_bar.set_title("Post-Fill Mid Movement Split\n(favorable / no-change / adverse)", fontweight="bold")
    ax_bar.legend(loc="upper right", fontsize=8)

    colors = ["#2196F3", "#4CAF50", "#FF9800"]
    any_nz = False
    for deltas, lbl, clr in zip(datasets, labels, colors):
        nz = deltas[deltas != 0]
        if len(nz) < 5:
            continue
        any_nz = True
        lo, hi = np.percentile(nz, [5, 95])
        ax_hist.hist(
            nz,
            bins=30,
            range=(lo, hi),
            alpha=0.55,
            color=clr,
            label=f"{lbl}  (n={len(nz)}, mean={nz.mean():.4f})",
            density=True,
        )
        ax_hist.axvline(nz.mean(), color=clr, lw=1.5, linestyle="--")

    ax_hist.axvline(0, color="black", lw=1.0, linestyle="--", label="zero")
    ax_hist.set_xlabel("Favorable mid change (prob. pts)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Non-Zero Post-Fill Mid Moves Only\n(clipped 5th–95th pct)", fontweight="bold")
    if any_nz:
        ax_hist.legend(fontsize=8)

    fig2.suptitle("Post-Fill Adverse Selection Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return df_fill, daily_fill, df_mid, all_pd, fills_pd, d1, d5, d30, labels, datasets


def display_contract_type_breakdown(df):
    df_typed = df.with_columns([
        pl.when(pl.col("contract_id").str.starts_with("KXINXU"))
        .then(pl.lit("KXINXU (threshold)"))
        .otherwise(pl.lit("KXINX (range)"))
        .alias("ctype"),
        (pl.col("bid_fill") | pl.col("ask_fill")).alias("any_fill"),
        (pl.col("my_bid").is_not_null() & (pl.col("my_bid") > 0)).alias("quoting"),
    ])

    type_stats = (
        df_typed.group_by("ctype")
        .agg([
            pl.col("contract_id").n_unique().alias("n_contracts"),
            pl.col("any_fill").sum().alias("total_fills"),
            pl.col("quoting").sum().alias("quoted_rows"),
            (pl.col("my_ask") - pl.col("my_bid")).filter(pl.col("my_bid") > 0).mean().alias("avg_spread"),
            pl.col("my_bid_size").filter(pl.col("quoting")).mean().alias("avg_size"),
        ])
        .with_columns((pl.col("total_fills") / pl.col("quoted_rows") * 100).alias("fill_rate_pct"))
        .sort("ctype")
    ).to_pandas().set_index("ctype")

    type_stats.columns = ["Contracts", "Fills", "Quoted Rows", "Avg Spread", "Avg Size", "Fill Rate (%)"]
    display(type_stats[["Contracts", "Fills", "Fill Rate (%)", "Avg Spread", "Avg Size"]].round(3))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    pal = ["#2196F3", "#FF9800"]
    ctypes = type_stats.index.tolist()

    axes[0].bar(ctypes, type_stats["Fills"], color=pal)
    axes[0].set_title("Total Fills", fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[1].bar(ctypes, type_stats["Fill Rate (%)"], color=pal)
    axes[1].set_title("Fill Rate (%)", fontweight="bold")
    axes[1].set_ylabel("Fill Rate (%)")
    axes[2].bar(ctypes, type_stats["Avg Spread"], color=pal)
    axes[2].set_title("Avg Quoted Spread", fontweight="bold")
    axes[2].set_ylabel("Spread (prob. pts)")

    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    plt.suptitle("KXINX vs KXINXU — Contract Type Breakdown", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return df_typed, type_stats
