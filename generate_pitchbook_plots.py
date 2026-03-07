#!/usr/bin/env python3
"""
Generate pitchbook figures from simulation output parquets.
Saves all figures to pitchbook_figs/.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import polars as pl
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
OUTDIR = "pitchbook_figs"
BASE_PATH   = "simulation_output.parquet"
NOFEES_PATH = "simulation_no_fees.parquet"
NOHEDGE_PATH = "simulation_no_hedge.parquet"
INITIAL_CAPITAL = 10_000.0
ANN_SECONDS = 252 * 6.5 * 3600

MAROON    = "#800000"
STEELBLUE = "#2c7bb6"
ORANGE    = "#d7631c"
GREEN     = "#2e7d32"
RED       = "#c62828"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

os.makedirs(OUTDIR, exist_ok=True)


def save_fig(name):
    path = os.path.join(OUTDIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def portfolio_ts(df):
    """Portfolio-level time series: unique by ts, keep last row."""
    return df.sort("ts").unique(subset=["ts"], keep="last").sort("ts")


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading simulation parquets...")
df_base    = pl.read_parquet(BASE_PATH)
df_nofees  = pl.read_parquet(NOFEES_PATH)
df_nohedge = pl.read_parquet(NOHEDGE_PATH)

ts_base    = portfolio_ts(df_base)
ts_nofees  = portfolio_ts(df_nofees)
ts_nohedge = portfolio_ts(df_nohedge)

ts_pd = ts_base["ts"].to_pandas()

print(f"  Base rows: {len(ts_base):,}, No-fees: {len(ts_nofees):,}, No-hedge: {len(ts_nohedge):,}")


# ── 1. Market Environment ──────────────────────────────────────────────────────
print("\n[1/9] Market environment...")
spx_daily = (
    ts_base
    .with_columns(pl.col("ts").dt.date().alias("date"))
    .group_by("date")
    .agg(pl.col("spx").last().alias("spx"), pl.col("vix").last().alias("vix"))
    .sort("date")
    .to_pandas()
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
fig.suptitle("Market Environment: January 2 – February 26, 2026", fontsize=13, fontweight="bold")

ax1.plot(spx_daily["date"], spx_daily["spx"], color=STEELBLUE, linewidth=1.8)
ax1.set_ylabel("Index Level")
ax1.set_title("S\&P 500 (SPX) — Daily Close")
ax1.grid(True, alpha=0.25)
ax1.set_facecolor("#f9f9f9")

ax2.plot(spx_daily["date"], spx_daily["vix"], color="tomato", linewidth=1.8)
ax2.set_ylabel("VIX Level")
ax2.set_title("CBOE VIX — Daily Close")
ax2.grid(True, alpha=0.25)
ax2.set_facecolor("#f9f9f9")

save_fig("fig_market_env.png")


# ── 2. PnL Over Time (base) ───────────────────────────────────────────────────
print("[2/9] PnL curve (base)...")
pnl_pd = ts_base["pnl"].to_pandas()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(ts_pd, pnl_pd, color=MAROON, linewidth=1.2)
ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)
ax.fill_between(ts_pd, 0, pnl_pd, where=(pnl_pd >= 0), color=GREEN, alpha=0.12, label="Positive PnL")
ax.fill_between(ts_pd, 0, pnl_pd, where=(pnl_pd < 0), color=RED,   alpha=0.12, label="Negative PnL")
ax.set_xlabel("Date")
ax.set_ylabel("PnL ($)")
ax.set_title("Cumulative PnL — Base Strategy (Fees + Hedge)", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.25)
ax.set_facecolor("#f9f9f9")
save_fig("fig_pnl_base.png")


# ── 3. PnL Comparison (3 scenarios) ──────────────────────────────────────────
print("[3/9] PnL comparison...")
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(ts_nofees["ts"].to_pandas(),  ts_nofees["pnl"].to_pandas(),  label="No Fees (hedge on) — $1,645.67", color=STEELBLUE, linewidth=1.3)
ax.plot(ts_nohedge["ts"].to_pandas(), ts_nohedge["pnl"].to_pandas(), label="No Hedge (fees on) — $1,425.84",  color=ORANGE,    linewidth=1.3)
ax.plot(ts_pd,                        pnl_pd,                        label="Base (fees + hedge) — $961.20",   color=MAROON,    linewidth=1.3)
ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
ax.set_xlabel("Date")
ax.set_ylabel("PnL ($)")
ax.set_title("PnL Comparison: Base vs. Variants", fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.25)
ax.set_facecolor("#f9f9f9")
save_fig("fig_pnl_comparison.png")


# ── 4. Rolling Drawdown ───────────────────────────────────────────────────────
print("[4/9] Rolling drawdown...")
pnl_np  = ts_base["pnl"].to_numpy()
cummax  = np.maximum.accumulate(np.maximum(pnl_np, 0))
drawdown = cummax - pnl_np

fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(ts_pd, 0, drawdown, color=MAROON, alpha=0.65, label="Drawdown from peak")
ax.set_ylabel("Drawdown ($)")
ax.set_title("Rolling Drawdown — Base Strategy", fontweight="bold")
ax.set_xlabel("Date")
ax.grid(True, alpha=0.25)
ax.set_facecolor("#f9f9f9")
ax.legend()
save_fig("fig_drawdown.png")


# ── 5. Daily Returns Bar Chart ────────────────────────────────────────────────
print("[5/9] Daily returns...")
pv_pd = ts_base["portfolio_value"].to_pandas()
df_ts_pd = pd.DataFrame({"ts": ts_pd, "pv": pv_pd}).set_index("ts")
daily_pv   = df_ts_pd["pv"].resample("1D").last().dropna()
daily_rets = daily_pv.pct_change().dropna() * 100

bar_colors = [GREEN if r >= 0 else RED for r in daily_rets.values]
fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(daily_rets.index, daily_rets.values, color=bar_colors, alpha=0.8, edgecolor="white", linewidth=0.3, width=0.8)
ax.axhline(0, color="black", linewidth=0.6)
ax.set_xlabel("Date")
ax.set_ylabel("Daily Return (%)")
ax.set_title("Daily Returns — Base Strategy  (green = profitable day)", fontweight="bold")
ax.grid(axis="y", alpha=0.25)
ax.set_facecolor("#f9f9f9")
save_fig("fig_daily_returns.png")


# ── 6. Return Histogram (1-second) ────────────────────────────────────────────
print("[6/9] Return histogram...")
pv_np   = ts_base["portfolio_value"].to_numpy()
rets    = np.diff(pv_np) / INITIAL_CAPITAL
rets_finite = rets[np.isfinite(rets)]

p5  = np.percentile(rets_finite, 5)
p1  = np.percentile(rets_finite, 1)
p99 = np.percentile(rets_finite, 99)
p95 = np.percentile(rets_finite, 95)

# Clip display to ±p99.5 range
clip_lo = np.percentile(rets_finite, 0.5)
clip_hi = np.percentile(rets_finite, 99.5)
rets_display = np.clip(rets_finite, clip_lo, clip_hi)

fig, ax = plt.subplots(figsize=(12, 4))
ax.hist(rets_display, bins=120, color=STEELBLUE, alpha=0.75, edgecolor="white", density=True)
ax.axvline(p5, color=RED,     linestyle="--", linewidth=1.8, label=f"5th pct = {p5*INITIAL_CAPITAL:.2f}$/sec")
ax.axvline(p1, color=MAROON,  linestyle="--", linewidth=1.8, label=f"1st pct = {p1*INITIAL_CAPITAL:.2f}$/sec")
ax.set_xlabel("1-Second Return (fraction of $10k capital)")
ax.set_ylabel("Density")
ax.set_title("1-Second Return Distribution — Base Strategy\n(clipped at 0.5th/99.5th %ile; tail events at ±$500 excluded from display)",
             fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.25)
ax.set_facecolor("#f9f9f9")
save_fig("fig_returns_hist.png")


# ── 7. Kalshi Inventory + SPY Position ───────────────────────────────────────
print("[7/9] Inventory + SPY position...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=False)

ax1.plot(ts_pd, ts_base["total_kalshi_inventory"].to_numpy(), color=STEELBLUE, linewidth=0.8)
ax1.axhline(0, color="black", linewidth=0.5, alpha=0.5)
ax1.set_ylabel("Net Contracts")
ax1.set_title("Total Kalshi Inventory (signed)", fontweight="bold")
ax1.grid(True, alpha=0.25)
ax1.set_facecolor("#f9f9f9")

ax2.plot(ts_pd, ts_base["pos_spy"].to_numpy(), color="coral", linewidth=0.8)
ax2.axhline(0, color="black", linewidth=0.5, alpha=0.5)
ax2.set_ylabel("SPY Shares")
ax2.set_title("SPY Delta Hedge Position (signed)", fontweight="bold")
ax2.set_xlabel("Date")
ax2.grid(True, alpha=0.25)
ax2.set_facecolor("#f9f9f9")

save_fig("fig_inventory.png")


# ── 8. Fill Volume by Hour ────────────────────────────────────────────────────
print("[8/9] Fill volume by hour...")
fills = df_base.filter(pl.col("bid_fill") | pl.col("ask_fill"))
fill_df = fills.with_columns([
    pl.col("ts").dt.hour().alias("hour"),
    (
        pl.col("my_bid_size") * pl.col("bid_fill").cast(pl.Int64)
      + pl.col("my_ask_size") * pl.col("ask_fill").cast(pl.Int64)
    ).alias("fill_qty"),
])
hourly_vol = fill_df.group_by("hour").agg(pl.col("fill_qty").sum()).sort("hour").to_pandas()

rth_mask = (hourly_vol["hour"] >= 9) & (hourly_vol["hour"] < 16)
colors = [MAROON if m else "#d4a0a0" for m in rth_mask]

fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(hourly_vol["hour"], hourly_vol["fill_qty"], color=colors, edgecolor="white", linewidth=0.4)
ax.set_xlabel("Hour of Day (ET)")
ax.set_ylabel("Filled Contracts")
ax.set_title("Strategy Fill Volume by Hour of Day\n(dark = Regular Trading Hours 9:30–16:00 ET)", fontweight="bold")
ax.grid(axis="y", alpha=0.25)
ax.set_facecolor("#f9f9f9")
save_fig("fig_fill_volume.png")


# ── 9. Average Spread by Hour ─────────────────────────────────────────────────
print("[9/9] Average spread by hour...")
spread_df = (
    df_base
    .with_columns([
        (pl.col("my_ask") - pl.col("my_bid")).alias("spread"),
        pl.col("ts").dt.hour().alias("hour"),
    ])
    .filter((pl.col("spread") > 0) & (pl.col("spread") < 1))
    .group_by("hour")
    .agg(pl.col("spread").mean().alias("avg_spread"))
    .sort("hour")
    .to_pandas()
)

rth_mask_s = (spread_df["hour"] >= 9) & (spread_df["hour"] < 16)
colors_s = [MAROON if m else "#d4a0a0" for m in rth_mask_s]

fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(spread_df["hour"], spread_df["avg_spread"], color=colors_s, edgecolor="white", linewidth=0.4)
ax.set_xlabel("Hour of Day (ET)")
ax.set_ylabel("Avg Bid-Ask Spread")
ax.set_title("Average Quoted Spread by Hour of Day\n(dark = Regular Trading Hours; widens outside RTH)", fontweight="bold")
ax.grid(axis="y", alpha=0.25)
ax.set_facecolor("#f9f9f9")
save_fig("fig_spread_by_hour.png")


print("\nAll figures saved successfully.")
