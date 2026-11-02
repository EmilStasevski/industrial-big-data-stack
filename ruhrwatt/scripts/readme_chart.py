"""README chart from the backtest outputs: one test week of forecasts, and MAE per model.

    make backtest && make readme-chart      # writes docs/img/backtest.svg (commit it)

Left: actual day-ahead price, the best model and "same quarter-hour yesterday" for the last
full test week, with the calibrated P10-P90 range. Right: MAE of every model on identical rows.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from ruhrwatt.gold import mart  # noqa: E402

PRICE = "price_eur_mwh"
INK, MUTED, GRID = "#1f2328", "#656d76", "#d8dee4"
ACTUAL, MODEL, NAIVE, BAND = "#1f2328", "#0969da", "#bf8700", "#0969da"
LABELS = {
    "naive_d1": "yesterday (naive)",
    "naive_d7": "last week (naive)",
    "lightgbm_no_fc": "LightGBM, no TSO forecasts",
    "hgb": "HistGradientBoosting",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "ridge": "Ridge",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=Path, default=None, help="gold folder (default: data/gold)")
    ap.add_argument("--out", type=Path, default=Path("docs/img/backtest.svg"))
    args = ap.parse_args()
    gold = mart.gold_dir(args.gold)
    preds = pl.read_parquet(gold / "backtest_predictions.parquet")
    scores = pl.read_parquet(gold / "backtest_scores.parquet").sort("mae")
    best = scores["model"][0] if scores["model"][0] != "naive_d1" else scores["model"][1]

    # last full Monday-to-Sunday week in the test period
    last_day = preds["date_berlin"].max()
    end = last_day - timedelta(days=(last_day.weekday() + 1) % 7)  # last Sunday on or before
    start = end - timedelta(days=6)
    week = preds.filter(pl.col("date_berlin").is_between(start, end)).sort("ts_utc")
    ts = week["ts_berlin"].dt.replace_time_zone(None).to_list()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.3, 1]}, layout="constrained"
    )
    fig.patch.set_facecolor("white")

    if "p10" in week.columns:
        ax.fill_between(
            ts, week["p10"], week["p90"], color=BAND, alpha=0.13, lw=0, label="P10–P90 range"
        )
    ax.plot(ts, week["pred_naive_d1"], color=NAIVE, lw=1, ls=(0, (3, 2)), label=LABELS["naive_d1"])
    ax.plot(ts, week[f"pred_{best}"], color=MODEL, lw=1.4, label=LABELS.get(best, best))
    ax.plot(ts, week[PRICE], color=ACTUAL, lw=1.1, label="actual")
    ax.set_title(
        f"Day-ahead price, test week {start:%d.%m.}–{end:%d.%m.%Y}",
        loc="left",
        color=INK,
        fontsize=11,
    )
    ax.set_ylabel("EUR/MWh")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d.%m."))
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, ncols=2)

    rows = scores.reverse()
    names = [LABELS.get(m, m) for m in rows["model"]]
    colors = [
        MODEL if m == best else (NAIVE if m.startswith("naive") else "#8c959f")
        for m in rows["model"]
    ]
    bx.barh(names, rows["mae"], color=colors, height=0.62)
    ref = float(scores.filter(pl.col("model") == "naive_d1")["mae"][0])
    for y, (mae, m) in enumerate(zip(rows["mae"], rows["model"], strict=True)):
        note = f"{mae:.1f}" + ("" if m == "naive_d1" else f"  ({1 - mae / ref:+.0%})")
        bx.text(mae + ref * 0.015, y, note, va="center", fontsize=8.5, color=INK)
    bx.set_xlim(0, ref * 1.45)
    bx.set_title("MAE, EUR/MWh (skill)", loc="left", color=INK, fontsize=11)
    bx.tick_params(axis="y", length=0)
    bx.grid(axis="x", color=GRID, lw=0.6)
    bx.set_axisbelow(True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format=args.out.suffix.lstrip(".") or "svg")
    print(f"chart -> {args.out}  (best: {best}, {scores['n'][0]:,} quarter-hours scored)")


if __name__ == "__main__":
    main()
