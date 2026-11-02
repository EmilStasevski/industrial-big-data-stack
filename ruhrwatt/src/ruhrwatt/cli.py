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


@gold_app.command("export-serving")
def gold_export_serving(
    days: Annotated[int, typer.Option(help="Days of 15-minute prices to include.")] = 31,
) -> None:
    """Write the JSON snapshots the Go API serves (data/serving/)."""
    from ruhrwatt.gold import serving

    try:
        r = serving.export(days=days)
    except (FileNotFoundError, OSError) as exc:
        typer.echo(f"FAILED: {exc}; run `make gold` first", err=True)
        raise typer.Exit(1) from exc
    f = r["freshness"]
    typer.echo(
        f"serving snapshot -> {r['dir']}: {r['months']} months, {r['price_days']} price days "
        f"(prices up to {f['last_price_utc']})"
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


spark_app = typer.Typer(
    help="PySpark versions of the heavy stages (needs Java 17 or 21).", no_args_is_help=True
)
app.add_typer(spark_app, name="spark")


@spark_app.command("build")
def spark_build() -> None:
    """Silver tables and gold features with Spark -> data/spark/, then the silver gates."""
    from ruhrwatt.silver import gates
    from ruhrwatt.spark import pipeline
    from ruhrwatt.spark.session import JavaMissingError

    try:
        out = pipeline.build()
    except (FileNotFoundError, JavaMissingError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(
        f"spark: {out['rows']} quarter-hours; session start {out['startup_s']:.1f} s, "
        f"pipeline {out['run_s']:.1f} s"
    )
    for name, path in out["paths"].items():
        typer.echo(f"  {name:9} -> {path}")
    typer.echo("quality gates (on the Spark output):")
    # Spark does not keep row order through joins; the gates expect time order.
    if not _report_gates(gates.check(pipeline.to_polars(out["qh"]).sort("ts_utc"))):
        raise typer.Exit(1)


@spark_app.command("parity")
def spark_parity() -> None:
    """Build every table with Spark and with Polars from the same bronze and compare."""
    from ruhrwatt.spark import pipeline
    from ruhrwatt.spark.session import JavaMissingError

    try:
        results = pipeline.parity()
    except (FileNotFoundError, JavaMissingError) as exc:
        typer.echo(f"FAILED: {exc}", err=True)
        raise typer.Exit(1) from exc
    ok = True
    for table, diffs in results.items():
        if not diffs:
            typer.echo(f"  [ok  ] {table}: identical (floats within 1e-9)")
            continue
        ok = False
        typer.echo(f"  [FAIL] {table}:")
        for d in diffs[:10]:
            typer.echo(f"         {d.column}: {d.problem}")
    if not ok:
        raise typer.Exit(1)


rag_app = typer.Typer(help="Local RAG assistant (Ollama).", no_args_is_help=True)
app.add_typer(rag_app, name="rag")


def _rag_fail(exc: Exception) -> None:
    typer.echo(f"FAILED: {exc}", err=True)
    raise typer.Exit(1) from exc


def _print_answer(ans, verbose: bool) -> None:
    typer.echo(ans.text)
    if ans.sources:
        typer.echo("\nSources:")
        for src in ans.sources:
            typer.echo(f"  - {src}")
    if verbose:
        typer.echo(f"\n[{len(ans.tool_calls)} tool calls, {ans.steps} steps, {ans.seconds:.1f} s]")
        for c in ans.tool_calls:
            typer.echo(f"  {c['name']}({c['arguments']})")


@rag_app.command("doctor")
def rag_doctor() -> None:
    """Check Ollama, models, data and index; show VRAM use of loaded models."""
    from ruhrwatt.rag.app import doctor

    ok = True
    for name, good, detail in doctor():
        ok = ok and good
        typer.echo(f"  [{'ok  ' if good else 'FAIL'}] {name}: {detail}")
    if not ok:
        raise typer.Exit(1)


@rag_app.command("index")
def rag_index() -> None:
    """Build or refresh the docs index (only changed chunks are embedded again)."""
    from ruhrwatt.rag import docs
    from ruhrwatt.rag.llm import OllamaClient, OllamaError

    try:
        r = docs.build_index(OllamaClient().embed)
    except OllamaError as exc:
        _rag_fail(exc)
    typer.echo(
        f"docs index: {r['chunks']} chunks ({r['embedded']} embedded, {r['reused']} reused) "
        f"-> {r['path']}"
    )


@rag_app.command("ask")
def rag_ask(
    question: Annotated[str, typer.Argument(help="Your question, in English or German.")],
    model: Annotated[str | None, typer.Option(help="Ollama model, e.g. qwen3:14b.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show tool calls.")] = False,
) -> None:
    """Ask one question."""
    from dataclasses import replace

    from ruhrwatt.rag.app import make_agent
    from ruhrwatt.rag.config import Settings
    from ruhrwatt.rag.llm import OllamaError

    s = Settings() if model is None else replace(Settings(), model=model)
    try:
        _print_answer(make_agent(s).ask(question), verbose)
    except OllamaError as exc:
        _rag_fail(exc)


@rag_app.command("chat")
def rag_chat(
    model: Annotated[str | None, typer.Option(help="Ollama model, e.g. qwen3:14b.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show tool calls.")] = False,
) -> None:
    """Interactive chat (keeps context). Empty line or Ctrl-D to quit, /reset to clear."""
    from dataclasses import replace

    from ruhrwatt.rag.app import make_agent
    from ruhrwatt.rag.config import Settings
    from ruhrwatt.rag.llm import OllamaError

    s = Settings() if model is None else replace(Settings(), model=model)
    agent = make_agent(s)
    typer.echo(f"Ruhrwatt assistant ({s.model}). Ask about prices, NRW, load shifting, forecasts.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        if q == "/reset":
            agent.reset()
            typer.echo("(context cleared)")
            continue
        try:
            _print_answer(agent.ask(q), verbose)
        except OllamaError as exc:
            typer.echo(f"FAILED: {exc}", err=True)


@rag_app.command("eval")
def rag_eval(
    model: Annotated[
        list[str] | None, typer.Option(help="Model(s) to compare; repeat the option.")
    ] = None,
) -> None:
    """Run the evaluation set (truth computed from the data) for one or more models."""
    from dataclasses import replace

    import polars as pl

    from ruhrwatt import config as cfg
    from ruhrwatt.rag import evaluate
    from ruhrwatt.rag.app import make_agent
    from ruhrwatt.rag.config import Settings
    from ruhrwatt.rag.llm import OllamaError

    models = model or [Settings().model]
    try:
        cases = evaluate.build_cases()
    except Exception as exc:  # missing data / tables
        _rag_fail(exc)
    frames = []
    for m in models:
        s = replace(Settings(), model=m)
        typer.echo(f"\n{m}: {len(cases)} questions")
        try:
            results = evaluate.run(
                lambda s=s: make_agent(s),
                cases,
                progress=lambda r: typer.echo(
                    f"  [{'pass' if r.passed else 'FAIL'}] {r.id:26} {r.seconds:5.1f} s  "
                    f"tools={','.join(r.tools_used) or '-'}"
                    + ("" if r.passed else f"\n         {r.detail}")
                ),
            )
        except OllamaError as exc:
            _rag_fail(exc)
        n_ok = sum(r.passed for r in results)
        avg = sum(r.seconds for r in results) / len(results)
        typer.echo(f"  => {n_ok}/{len(results)} passed, {avg:.1f} s per question")
        frames.append(evaluate.to_frame(results, m))
    out = cfg.data_dir() / "rag" / "eval_results.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.concat(frames).write_parquet(out)
    typer.echo(f"\nanswers and checks -> {out}")


@rag_app.command("batch")
def rag_batch(
    file: Annotated[
        str, typer.Argument(help="JSON file with questions.")
    ] = "questions/fragen_de.json",
    model: Annotated[str | None, typer.Option(help="Ollama model, e.g. qwen3:14b.")] = None,
) -> None:
    """Run all questions of a JSON file; writes a Markdown report and the raw JSON."""
    import json
    import re
    from dataclasses import replace
    from pathlib import Path

    from ruhrwatt import config as cfg
    from ruhrwatt.rag import batch
    from ruhrwatt.rag.app import make_agent
    from ruhrwatt.rag.config import Settings
    from ruhrwatt.rag.llm import OllamaError

    s = Settings() if model is None else replace(Settings(), model=model)
    try:
        items = batch.load_questions(Path(file))
    except (OSError, ValueError, KeyError) as exc:
        _rag_fail(exc)
    n = sum(len(it.get("conversation") or [1]) for it in items)
    typer.echo(f"{s.model}: {n} questions from {file}")
    try:
        results = batch.run(
            lambda: make_agent(s),
            items,
            progress=lambda r: typer.echo(
                f"  {r['id']:28} {r.get('seconds', 0):5.1f} s  "
                + (
                    f"ERROR {r['error']}"
                    if r.get("error")
                    else f"tools={','.join(t['name'] for t in r['tools']) or '-'}"
                )
            ),
            fatal=(OllamaError,),
        )
    except OllamaError as exc:
        _rag_fail(exc)
    out = cfg.data_dir() / "rag"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"batch_{re.sub(r'[^A-Za-z0-9]+', '_', s.model)}"
    (out / f"{stem}.json").write_text(
        json.dumps({"model": s.model, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / f"{stem}.md").write_text(batch.to_markdown(results, s.model), encoding="utf-8")
    typer.echo(f"\nreport -> {out / (stem + '.md')}  (raw: {stem}.json)")


if __name__ == "__main__":
    app()
