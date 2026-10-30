"""Command line interface: ``uv run ruhrwatt ...``."""

from __future__ import annotations

from datetime import date
from typing import Annotated

import httpx
import typer

from ruhrwatt import config
from ruhrwatt.download import smard

app = typer.Typer(help="Ruhrwatt: NRW industrial energy data toolkit.", no_args_is_help=True)
smard_app = typer.Typer(help="SMARD (Bundesnetzagentur) market data.", no_args_is_help=True)
app.add_typer(smard_app, name="smard")


@smard_app.command("list")
def list_datasets() -> None:
    """Show the registered SMARD datasets."""
    for ds in smard.DATASETS.values():
        flag = "verified" if ds.verified else "unverified id"
        typer.echo(
            f"{ds.name:18} filter={ds.filter_id:<5} regions={','.join(ds.regions):14} {flag}"
        )


def _run(dataset: str, region: str, resolution: str, since: date | None, refresh: int) -> int:
    ds = smard.DATASETS[dataset]
    root = config.raw_dir("smard")
    with smard.make_client() as client:
        stats = smard.download_series(
            client,
            root,
            ds.filter_id,
            region,
            resolution,
            since=since,
            refresh_weeks=refresh,
            delay=config.request_delay(),
        )
    typer.echo(
        f"{dataset}/{region}/{resolution}: fetched={stats.fetched} "
        f"skipped={stats.skipped} failed={stats.failed}"
    )
    for err in stats.errors[:5]:
        typer.echo(f"  ! {err}", err=True)
    return stats.failed


@smard_app.command("download")
def download(
    dataset: Annotated[str, typer.Option(help="Dataset name, see `smard list`.")],
    region: Annotated[str, typer.Option(help="e.g. DE-LU, DE, Amprion.")],
    resolution: Annotated[
        str, typer.Option(help="quarterhour|hour|day|week|month|year")
    ] = "quarterhour",
    since: Annotated[
        str | None, typer.Option(help="ISO date, e.g. 2025-10-01. Default: full history.")
    ] = None,
    refresh_weeks: Annotated[int, typer.Option(help="Re-fetch the newest N weeks.")] = 2,
) -> None:
    """Download one SMARD series into data/raw/smard."""
    if dataset not in smard.DATASETS:
        raise typer.BadParameter(f"unknown dataset {dataset!r}; run `ruhrwatt smard list`")
    if resolution not in smard.RESOLUTIONS:
        raise typer.BadParameter(f"resolution must be one of {smard.RESOLUTIONS}")
    try:
        failed = _run(
            dataset, region, resolution, date.fromisoformat(since) if since else None, refresh_weeks
        )
    except httpx.HTTPError as exc:
        typer.echo(f"Could not reach SMARD: {exc}", err=True)
        raise typer.Exit(2) from exc
    if failed:
        raise typer.Exit(1)


@smard_app.command("download-all")
def download_all(
    since: Annotated[str | None, typer.Option(help="ISO date.")] = None,
    resolution: str = "quarterhour",
    refresh_weeks: int = 2,
) -> None:
    """Download every registered dataset for its default regions."""
    since_d = date.fromisoformat(since) if since else None
    total_failed = 0
    try:
        for ds in smard.DATASETS.values():
            for region in ds.regions:
                try:
                    total_failed += _run(ds.name, region, resolution, since_d, refresh_weeks)
                except httpx.HTTPStatusError as exc:
                    # SMARD answers 404 for an unknown filter/region: report it, keep going.
                    typer.echo(
                        f"{ds.name}/{region}/{resolution}: FAILED, no index "
                        f"(HTTP {exc.response.status_code}; filter id {ds.filter_id} "
                        "may be wrong, check with `ruhrwatt smard probe`)",
                        err=True,
                    )
                    total_failed += 1
    except httpx.TransportError as exc:
        typer.echo(f"Could not reach SMARD: {exc}", err=True)
        raise typer.Exit(2) from exc
    if total_failed:
        raise typer.Exit(1)


@smard_app.command("probe")
def probe(
    filter_ids: Annotated[list[int], typer.Argument(help="SMARD filter ids to check.")],
    region: str = "DE",
    resolution: str = "quarterhour",
) -> None:
    """Check filter ids live: does the series exist, and what do its values look like?

    Prints the last complete week: count, mean, min, max and the night share
    (mean at 21:00-01:59 UTC / overall mean; ~0 for PV, ~1 for wind and load).
    """
    import json as _json

    with smard.make_client() as client:
        for fid in filter_ids:
            try:
                stamps = smard.fetch_index(client, fid, region, resolution)
                ts = stamps[-2] if len(stamps) > 1 else stamps[-1]
                body = client.get(smard.week_url(fid, region, resolution, ts)).content
            except httpx.HTTPError as exc:
                typer.echo(f"{fid:>5} {region}: not available ({exc})")
                continue
            rows = [(t, v) for t, v in _json.loads(body)["series"] if v is not None]
            if not rows:
                typer.echo(f"{fid:>5} {region}: exists, but the last week has no values")
                continue
            vals = [v for _, v in rows]
            night = [v for t, v in rows if datetime_hour_utc(t) in (21, 22, 23, 0, 1)]
            mean = sum(vals) / len(vals)
            ratio = (sum(night) / len(night) / mean) if night and mean else float("nan")
            typer.echo(
                f"{fid:>5} {region}: n={len(vals)} mean={mean:,.1f} min={min(vals):,.1f} "
                f"max={max(vals):,.1f} night/mean={ratio:.2f}"
            )


def datetime_hour_utc(ts_ms: int) -> int:
    return (ts_ms // 3_600_000) % 24


def _bronze_one(dataset: str, region: str, resolution: str) -> bool:
    """Convert one series; print a one-line summary. Returns True on success."""
    from ruhrwatt.bronze import smard as bronze

    try:
        s = bronze.to_bronze(dataset, region, resolution)
    except (FileNotFoundError, ValueError, OSError) as exc:
        typer.echo(f"{dataset}/{region}/{resolution}: FAILED ({exc})", err=True)
        return False
    typer.echo(
        f"{dataset}/{region}/{resolution}: rows={s['rows']} nulls={s['nulls']} "
        f"gaps={s['gaps']} dup={s['duplicates']} misaligned={s['misaligned']} "
        f"native={s['native_rows']}"
    )
    return True


@smard_app.command("to-bronze")
def to_bronze(
    dataset: Annotated[str, typer.Option(help="Dataset name, see `smard list`.")],
    region: Annotated[str, typer.Option(help="e.g. DE-LU, DE, Amprion.")],
    resolution: str = "quarterhour",
) -> None:
    """Parse raw files into bronze Parquet with the Rust core."""
    if dataset not in smard.DATASETS:
        raise typer.BadParameter(f"unknown dataset {dataset!r}; run `ruhrwatt smard list`")
    if not _bronze_one(dataset, region, resolution):
        raise typer.Exit(1)


@smard_app.command("to-bronze-all")
def to_bronze_all(resolution: str = "quarterhour") -> None:
    """Convert every registered dataset that has been downloaded."""
    ok = True
    for ds in smard.DATASETS.values():
        for region in ds.regions:
            ok = _bronze_one(ds.name, region, resolution) and ok
    if not ok:
        raise typer.Exit(1)


silver_app = typer.Typer(help="Silver layer: joined, featured tables.", no_args_is_help=True)
app.add_typer(silver_app, name="silver")


def _report_gates(results: list) -> bool:
    """Print every gate; True when no error-level gate failed."""
    ok = True
    for g in results:
        if g.ok:
            mark = "ok  "
        elif g.severity == "warn":
            mark = "WARN"
        else:
            mark, ok = "FAIL", False
        typer.echo(f"  [{mark}] {g.name}: {g.detail}")
    return ok


@silver_app.command("build")
def silver_build() -> None:
    """Join all bronze series into silver tables and run the quality gates."""
    import polars as pl

    from ruhrwatt.silver import energy, gates

    try:
        out = energy.build()
    except FileNotFoundError as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc
    qh, daily = out["qh"], out["daily"]
    complete = qh.filter(pl.col("is_complete"))
    full_days = daily.filter(pl.col("is_complete_day"))

    typer.echo(f"energy_15min: {qh.height} rows x {qh.width} cols -> {out['paths']['15min']}")
    typer.echo(f"energy_daily: {daily.height} days -> {out['paths']['daily']}")
    typer.echo(
        f"  range {qh['ts_berlin'][0]:%Y-%m-%d %H:%M} .. {qh['ts_berlin'][-1]:%Y-%m-%d %H:%M} "
        f"(Berlin); complete through {complete['ts_berlin'].max():%Y-%m-%d %H:%M}"
    )
    neg_hours = full_days["price_negative_qh"].sum() / 4
    typer.echo(
        f"  base price {full_days['price_base'].mean():.2f} EUR/MWh, "
        f"Amprion load-weighted {full_days['amp_load_weighted_price'].mean():.2f}; "
        f"negative-price hours {neg_hours:.0f}; "
        f"wind+PV share DE {full_days['de_wind_pv_mwh'].sum() / full_days['de_load_mwh'].sum():.1%}"
        f", Amprion {full_days['amp_wind_pv_mwh'].sum() / full_days['amp_load_mwh'].sum():.1%}"
    )
    typer.echo("quality gates:")
    if not _report_gates(gates.check(qh, energy.off_grid_rows())):
        raise typer.Exit(1)


gold_app = typer.Typer(
    help="Gold layer: forecasting features and load shifting.", no_args_is_help=True
)
app.add_typer(gold_app, name="gold")


def _shift_table(summary) -> None:
    typer.echo(f"  {'plant':15} {'days':>5} {'EUR/MWh saved':>14} {'share':>7} {'EUR/year':>12}")
    for r in summary.iter_rows(named=True):
        typer.echo(
            f"  {r['plant']:15} {r['days']:>5} {r['savings_eur_per_mwh']:>14.2f} "
            f"{r['savings_share']:>7.1%} {r['savings_eur_per_year']:>12,.0f}"
        )


@gold_app.command("build")
def gold_build() -> None:
    """Build features, load-shifting results and the monthly mart from silver."""
    from ruhrwatt.gold import mart, shift

    try:
        out = mart.build()
    except FileNotFoundError as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc
    f, m = out["features"], out["monthly"]
    typer.echo(
        f"price_features_15min: {f.height} rows x {f.width} cols -> {out['paths']['features']}"
    )
    typer.echo(f"load_shift_daily: {out['shift'].height} rows -> {out['paths']['shift']}")
    typer.echo(f"market_monthly: {m.height} months -> {out['paths']['monthly']}")
    typer.echo("load shifting within each day (illustrative plants, wholesale energy only):")
    _shift_table(shift.summary(out["shift"]))
    for p in shift.PRESETS.values():
        typer.echo(
            f"  {p.name}: {p.label}; {p.avg_mw:g} MW avg, {p.min_frac:.0%}-{p.max_frac:.0%} of avg"
        )


@gold_app.command("backtest")
def gold_backtest(
    test_weeks: Annotated[int, typer.Option(help="Most recent weeks to test on.")] = 8,
    retrain_days: Annotated[int, typer.Option(help="Retrain every N days (walk-forward).")] = 7,
    models: Annotated[
        str, typer.Option(help="Comma-separated subset, e.g. naive_d1,lightgbm,catboost.")
    ] = ",".join(
        (
            "naive_d1",
            "naive_d7",
            "ridge",
            "hgb",
            "lightgbm",
            "xgboost",
            "catboost",
            "lightgbm_no_fc",
        )
    ),
    intervals: Annotated[bool, typer.Option(help="Also fit P10/P90 quantile models.")] = True,
) -> None:
    """Walk-forward comparison of price forecasting models (weekly retraining)."""
    import polars as pl

    from ruhrwatt.gold import backtest, mart

    path = mart.gold_dir() / "price_features_15min.parquet"
    if not path.exists():
        typer.echo(f"FAILED: {path} missing; run `make gold`", err=True)
        raise typer.Exit(1)
    chosen = tuple(m.strip() for m in models.split(",") if m.strip())
    try:
        r = backtest.run(
            pl.read_parquet(path),
            test_weeks,
            retrain_days,
            chosen,
            intervals,
            progress=lambda msg: typer.echo(f"  {msg}"),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    out = mart.gold_dir()
    r["predictions"].write_parquet(out / "backtest_predictions.parquet")
    r["scores_df"].write_parquet(out / "backtest_scores.parquet")
    r["by_hour"].write_parquet(out / "backtest_by_hour.parquet")
    if r["importance"] is not None:
        r["importance"].write_parquet(out / "feature_importance.parquet")

    first, last = r["folds"][0][0], r["folds"][-1][1]
    typer.echo(
        f"test {first} .. {last} (exclusive), {len(r['folds'])} folds, "
        f"{r['test_rows']} quarter-hours scored on identical rows"
    )
    typer.echo(f"  {'model':15} {'MAE':>7} {'RMSE':>7} {'vs naive_d1':>12}   EUR/MWh")
    for i, sc in enumerate(r["scores"]):
        mark = "  <- best" if i == 0 else ""
        typer.echo(
            f"  {sc.model:15} {sc.mae:>7.2f} {sc.rmse:>7.2f} {sc.skill_vs_naive_d1:>+12.1%}{mark}"
        )
    if r["interval"]:
        iv = r["interval"]
        typer.echo(
            f"  P10-P90 price range (target: 80 % of prices inside): "
            f"raw quantiles {iv['coverage_raw']:.0%} (width {iv['width_raw']:.1f}), "
            f"conformal {iv['coverage_p10_p90']:.0%} (width {iv['mean_width']:.1f} EUR/MWh)"
        )
    if r["importance"] is not None:
        top = ", ".join(
            f"{row['feature']} {row['share']:.0%}"
            for row in r["importance"].head(6).iter_rows(named=True)
        )
        typer.echo(f"  top LightGBM features (gain): {top}")
    typer.echo(f"  results -> {out}/backtest_*.parquet, feature_importance.parquet")


@gold_app.command("shift")
def gold_shift(
    avg_mw: Annotated[float, typer.Option(help="Average electrical load of the plant, MW.")],
    min_frac: Annotated[float, typer.Option(help="Minimum load as share of average.")] = 0.8,
    max_frac: Annotated[float, typer.Option(help="Maximum load as share of average.")] = 1.2,
) -> None:
    """Savings from shifting load within each day for your own plant."""
    from ruhrwatt.gold import mart, shift

    try:
        plant = shift.Plant("custom", avg_mw, min_frac, max_frac)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    qh, daily = mart.read_silver()
    _shift_table(shift.summary(shift.run(qh, daily, [plant])))


if __name__ == "__main__":
    app()
