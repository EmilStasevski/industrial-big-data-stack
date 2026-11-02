"""Gold layer tests on synthetic data (offline)."""

from __future__ import annotations

import math
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy.optimize import linprog

from ruhrwatt.gold import backtest, features, mart, shift
from ruhrwatt.silver import energy
from synthetic import START, _value, utc, write_bronze

PRICE = energy.PRICE


def _noisy_price(ts: datetime) -> float:
    """Daily shape plus deterministic noise, so baselines and models differ."""
    rng = random.Random(int(ts.timestamp()))
    day = ts.timetuple().tm_yday
    return _value(PRICE, ts) + 25 * math.sin(day / 3) + rng.gauss(0, 15)


def _index_price(ts: datetime) -> float:
    """A unique value per quarter-hour (its index since START), to check exact lags."""
    return (ts - START).total_seconds() / 900


def silver(tmp: Path, start: datetime, end: datetime, **kw) -> tuple[pl.DataFrame, pl.DataFrame]:
    out = energy.build(write_bronze(tmp / "bronze", start, end, **kw), tmp / "silver")
    return out["qh"], out["daily"]


def at(df: pl.DataFrame, ts: datetime) -> dict:
    return df.filter(pl.col("ts_utc") == ts).row(0, named=True)


# --------------------------------------------------------------------------- features


@pytest.fixture
def autumn_features(tmp_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    qh, daily = silver(tmp_path, START, utc(2025, 10, 29), override={PRICE: _index_price})
    return features.build_features(qh, daily), qh


def test_one_row_per_quarter_hour_including_fall_back(autumn_features) -> None:
    feats, qh = autumn_features
    assert feats.height == qh.height
    assert feats["ts_utc"].is_unique().all()
    assert feats.filter(pl.col("date_berlin") == date(2025, 10, 26)).height == 100


def test_lags_follow_berlin_wall_clock(autumn_features) -> None:
    feats, qh = autumn_features
    # Monday 03:00 local after the clocks went back: yesterday 03:00 local is 25 h earlier.
    row = at(feats, utc(2025, 10, 27, 3))
    assert row["price_lag_1d"] == at(qh, utc(2025, 10, 26, 3))[PRICE]
    assert row["price_lag_7d"] == at(qh, utc(2025, 10, 20, 3))[PRICE]
    assert row["qh_of_day"] == 12


def test_repeated_autumn_hour_uses_first_occurrence(autumn_features) -> None:
    feats, qh = autumn_features
    first_0215 = qh.filter(
        (pl.col("date_berlin") == date(2025, 10, 26)) & (pl.col("hour_berlin") == 2)
    ).row(1, named=True)  # 02:15 CEST (first pass)
    row = at(feats, utc(2025, 10, 27, 2, 15))
    assert row["price_lag_1d"] == first_0215[PRICE]


def test_missing_spring_hour_gives_null_lag(tmp_path: Path) -> None:
    qh, daily = silver(tmp_path, utc(2025, 3, 28), utc(2025, 4, 2))
    feats = features.build_features(qh, daily)
    row = at(feats, utc(2025, 3, 31, 2))  # 02:00 did not exist on 2025-03-30
    assert row["price_lag_1d"] is None
    assert row["price_lag_2d"] is not None


def test_previous_day_stats(autumn_features, tmp_path: Path) -> None:
    feats, qh = autumn_features
    day_before = qh.filter(pl.col("date_berlin") == date(2025, 10, 22))[PRICE]
    row = at(feats, utc(2025, 10, 23, 12))
    assert row["price_d1_mean"] == pytest.approx(day_before.mean())
    assert row["price_d1_max"] == day_before.max()
    assert row["price_d1_spread"] == day_before.max() - day_before.min()


def test_features_never_use_actuals_of_the_delivery_day(tmp_path: Path) -> None:
    """Changing every actual (price, load, generation) from day D on must not change any
    feature of day D. Only the day-D forecasts are allowed in."""
    end = utc(2025, 10, 29)
    d = utc(2025, 10, 24)
    a_qh, a_daily = silver(tmp_path / "a", START, end)
    actuals = [c for c in energy.value_columns() if "_fc_" not in c]
    shock = {c: (lambda ts, c=c: _value(c, ts) * (3 if ts >= d else 1)) for c in actuals}
    b_qh, b_daily = silver(tmp_path / "b", START, end, override=shock)
    day = pl.col("date_berlin") == date(2025, 10, 24)
    fa = features.build_features(a_qh, a_daily).filter(day)
    fb = features.build_features(b_qh, b_daily).filter(day)
    cols = features.feature_columns()
    assert fa.select(cols).equals(fb.select(cols))
    assert not fa[PRICE].equals(fb[PRICE])  # the target itself did change


def test_day_d_forecasts_are_features(autumn_features) -> None:
    feats, qh = autumn_features
    ts = utc(2025, 10, 28, 13)  # late enough in the fixture to have a 7-day lag
    row, src = at(feats, ts), at(qh, ts)
    for c in features.FORECASTS:
        assert row[c] == src[c]
    day = qh.filter(pl.col("date_berlin") == date(2025, 10, 28))["de_fc_wind_pv_mwh"]
    assert row["de_fc_wind_pv_day_mean"] == pytest.approx(day.mean())
    assert row["de_fc_residual_proxy"] == pytest.approx(
        row["de_load_lag_7d"] - row["de_fc_wind_pv_mwh"]
    )


def test_history_only_feature_set_has_no_forecasts() -> None:
    no_fc = features.feature_columns(forecasts=False)
    assert not any("_fc_" in c for c in no_fc)
    assert set(no_fc) < set(features.feature_columns())


# --------------------------------------------------------------------------- load shifting


@pytest.fixture
def noisy(tmp_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    return silver(tmp_path, utc(2025, 10, 20), utc(2025, 10, 31), override={PRICE: _noisy_price})


def _lp_optimum(prices: np.ndarray, plant: shift.Plant) -> float:
    flat = plant.avg_mw * shift.QH_HOURS
    n = len(prices)
    res = linprog(
        prices,
        A_eq=np.ones((1, n)),
        b_eq=[flat * n],
        bounds=[(plant.min_frac * flat, plant.max_frac * flat)] * n,
        method="highs",
    )
    assert res.success
    return float(res.fun)


@pytest.mark.parametrize("plant", list(shift.PRESETS.values()), ids=lambda p: p.name)
def test_schedule_is_optimal_and_feasible(noisy, plant: shift.Plant) -> None:
    qh, daily = noisy
    sched = shift.schedule(shift.priced_days(qh, daily), plant)
    flat = plant.avg_mw * shift.QH_HOURS
    for day in [date(2025, 10, 22), date(2025, 10, 26)]:  # normal day and the 25-hour day
        d = sched.filter(pl.col("date_berlin") == day)
        x, p = d["shifted_mwh"].to_numpy(), d[PRICE].to_numpy()
        assert x.sum() == pytest.approx(flat * len(d))  # daily energy unchanged
        assert x.min() >= plant.min_frac * flat - 1e-9
        assert x.max() <= plant.max_frac * flat + 1e-9
        assert float(p @ x) == pytest.approx(_lp_optimum(p, plant), rel=1e-9, abs=1e-6)


def test_savings_are_never_negative_and_zero_without_flexibility(noisy) -> None:
    qh, daily = noisy
    days = shift.run(qh, daily, [*shift.PRESETS.values(), shift.Plant("rigid", 50, 1.0, 1.0)])
    assert (days["savings_eur"] >= -1e-6).all()
    rigid = days.filter(pl.col("plant") == "rigid")
    assert rigid["savings_eur"].abs().max() == pytest.approx(0.0, abs=1e-6)
    assert rigid["moved_mwh"].max() == pytest.approx(0.0, abs=1e-9)


def test_more_flexibility_saves_more(noisy) -> None:
    qh, daily = noisy
    plants = [shift.Plant(f"f{k}", 100, 1 - k / 10, 1 + k / 10) for k in (1, 3, 5)]
    s = shift.summary(shift.run(qh, daily, plants)).sort("plant")
    assert list(s["savings_eur"]) == sorted(s["savings_eur"])


def test_only_days_with_complete_prices_are_used(tmp_path: Path) -> None:
    qh, daily = silver(tmp_path, utc(2025, 10, 20), utc(2025, 10, 23, 12))
    days = shift.run(qh, daily, [shift.PRESETS["generic_10pct"]])
    assert days["date_berlin"].to_list() == [
        date(2025, 10, 20),
        date(2025, 10, 21),
        date(2025, 10, 22),
    ]


@pytest.mark.parametrize("args", [(0, 0.9, 1.1), (100, 1.1, 1.2), (100, 0.8, 0.9), (100, -0.1, 1)])
def test_invalid_plants_are_rejected(args) -> None:
    with pytest.raises(ValueError):
        shift.Plant("bad", *args)


# --------------------------------------------------------------------------- mart and backtest


def test_monthly_mart(noisy) -> None:
    qh, daily = noisy
    days = shift.run(qh, daily, list(shift.PRESETS.values()))
    m = mart.market_monthly(daily, days)
    assert m.height == 1 and m["month"][0] == date(2025, 10, 1)
    full = daily.filter(pl.col("is_complete_day"))
    assert m["negative_price_hours"][0] == full["price_negative_qh"].sum() / 4
    assert m["days"][0] == full.height
    for name in shift.PRESETS:
        assert m[f"shift_savings_eur_mwh_{name}"][0] > 0


@pytest.fixture
def ten_weeks(tmp_path: Path) -> pl.DataFrame:
    qh, daily = silver(tmp_path, utc(2025, 9, 1), utc(2025, 11, 10), override={PRICE: _noisy_price})
    return features.build_features(qh, daily)


def test_folds_are_contiguous_weekly_blocks(ten_weeks: pl.DataFrame) -> None:
    blocks = backtest.folds(ten_weeks, test_weeks=3, retrain_days=7)
    assert len(blocks) == 3
    assert all(b - a == timedelta(days=7) for a, b in blocks)
    assert all(blocks[i][1] == blocks[i + 1][0] for i in range(len(blocks) - 1))
    assert blocks[-1][1] - timedelta(days=1) == ten_weeks.drop_nulls(PRICE)["date_berlin"].max()


def test_walk_forward_scores_models_on_the_same_rows(ten_weeks: pl.DataFrame) -> None:
    models = ("naive_d7", "ridge", "lightgbm", "lightgbm_no_fc")
    r = backtest.run(ten_weeks, test_weeks=2, models=models)
    names = [s.model for s in r["scores"]]
    assert set(names) == {"naive_d1", *models}  # naive_d1 is added as the reference
    assert len({s.n for s in r["scores"]}) == 1
    ref = next(s for s in r["scores"] if s.model == "naive_d1")
    assert ref.skill_vs_naive_d1 == 0
    assert [s.mae for s in r["scores"]] == sorted(s.mae for s in r["scores"])
    assert r["predictions"]["fold"].n_unique() == 2
    assert 0 < r["interval"]["coverage_p10_p90"] <= 1
    assert r["importance"]["share"].sum() == pytest.approx(1.0)
    assert set(r["by_hour"].columns) == {"hour_berlin", *names}


def test_each_fold_trains_only_on_earlier_days(ten_weeks: pl.DataFrame, monkeypatch) -> None:
    seen: list[tuple[int, int]] = []
    real = backtest._matrix

    def spy(df: pl.DataFrame, cols: list[str]):
        seen.append((df.height, df["date_berlin"].max().toordinal()))
        return real(df, cols)

    monkeypatch.setattr(backtest, "_matrix", spy)
    r = backtest.run(ten_weeks, test_weeks=2, models=("ridge",), intervals=False)
    # calls come in (train, test) pairs per fold
    for (a, _b), (_, train_max) in zip(r["folds"], seen[0::2], strict=True):
        assert train_max < a.toordinal()


def test_all_three_boosting_libraries_run(tmp_path: Path) -> None:
    qh, daily = silver(tmp_path, utc(2025, 9, 1), utc(2025, 10, 6), override={PRICE: _noisy_price})
    r = backtest.run(
        features.build_features(qh, daily),
        test_weeks=1,
        models=("hgb", "lightgbm", "xgboost", "catboost"),
        intervals=False,
    )
    assert all(np.isfinite(s.mae) for s in r["scores"])


def test_conformal_margin_hits_the_target_coverage() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(0, 10, 5000)
    lo, hi = np.full_like(y, -3.0), np.full_like(y, 3.0)  # far too narrow for sd 10
    m = backtest.conformal_margin(lo, hi, y, 0.8)
    assert m > 0
    fresh = rng.normal(0, 10, 20000)
    inside = ((fresh >= lo[0] - m) & (fresh <= hi[0] + m)).mean()
    assert inside == pytest.approx(0.8, abs=0.02)
    # an over-wide range is shrunk
    assert backtest.conformal_margin(lo * 20, hi * 20, y, 0.8) < 0


def test_conformal_range_is_recalibrated_daily(ten_weeks: pl.DataFrame) -> None:
    r = backtest.run(ten_weeks, test_weeks=2, models=("naive_d1",))
    p = r["predictions"]
    assert {"p10_raw", "p90_raw", "p10", "p90", "conformal_margin"} <= set(p.columns)
    per_day = p.group_by("date_berlin").agg(pl.col("conformal_margin").n_unique().alias("n"))
    assert per_day["n"].max() == 1  # one margin per day
    assert p["conformal_margin"].n_unique() > 1  # ... that changes from day to day
    scale = (p["p90_raw"] - p["p10_raw"]).clip(lower_bound=backtest.WIDTH_FLOOR)
    assert (p["p10_raw"] - p["p10"]).to_numpy() == pytest.approx(
        (p["conformal_margin"] * scale).to_numpy()
    )
    assert set(r["interval"]) == {"coverage_raw", "width_raw", "coverage_p10_p90", "mean_width"}


def test_daily_margin_never_uses_the_day_itself_or_later() -> None:
    rng = np.random.default_rng(1)
    days = np.repeat(np.arange("2026-01-01", "2026-01-29", dtype="datetime64[D]"), 96)
    scores = rng.normal(0, 1, len(days))
    test = days >= np.datetime64("2026-01-15")
    base = backtest.daily_margins(days, scores, days[test], 0.8, 14)
    shocked = scores.copy()
    shocked[days >= np.datetime64("2026-01-20")] += 50  # huge misses from Jan 20 on
    after = backtest.daily_margins(days, shocked, days[test], 0.8, 14)
    unaffected = days[test] <= np.datetime64("2026-01-20")
    assert after[unaffected] == pytest.approx(base[unaffected])
    assert (after[days[test] == np.datetime64("2026-01-21")] > base.max()).all()


def test_scaled_margin_adapts_to_volatility() -> None:
    # same relative miss: a wide range gets a wider absolute margin than a narrow one
    lo, hi, y = np.array([0.0, 0.0]), np.array([10.0, 100.0]), np.array([15.0, 150.0])
    s = backtest.scaled_scores(lo, hi, y)
    assert s == pytest.approx([0.5, 0.5])


def test_unknown_model_is_rejected(ten_weeks: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown models"):
        backtest.run(ten_weeks, models=("prophet",))


def test_backtest_needs_enough_history(autumn_features) -> None:
    feats, _ = autumn_features
    with pytest.raises(ValueError, match="not enough data"):
        backtest.run(feats, test_weeks=52)


def test_cli_gold_commands(tmp_path: Path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from ruhrwatt.cli import app

    write_bronze(
        tmp_path / "bronze" / "smard",
        utc(2025, 9, 1),
        utc(2025, 11, 10),
        override={PRICE: _noisy_price},
    )
    monkeypatch.setenv("RUHRWATT_DATA_DIR", str(tmp_path))
    run = CliRunner().invoke
    assert run(app, ["silver", "build"]).exit_code == 0
    r = run(app, ["gold", "build"])
    assert r.exit_code == 0, r.output
    assert "steel_eaf" in r.output
    for name in ("price_features_15min", "load_shift_daily", "market_monthly"):
        assert (tmp_path / "gold" / f"{name}.parquet").exists()
    r = run(app, ["gold", "backtest", "--test-weeks", "2", "--models", "naive_d1,ridge,lightgbm"])
    assert r.exit_code == 0, r.output
    assert "<- best" in r.output and "conformal" in r.output
    for name in ("predictions", "scores", "by_hour"):
        assert (tmp_path / "gold" / f"backtest_{name}.parquet").exists()
    assert run(app, ["gold", "backtest", "--models", "nope"]).exit_code != 0
    r = run(app, ["gold", "shift", "--avg-mw", "40", "--min-frac", "0.5", "--max-frac", "1.5"])
    assert r.exit_code == 0 and "custom" in r.output
    assert run(app, ["gold", "shift", "--avg-mw", "40", "--min-frac", "1.2"]).exit_code != 0


def test_serving_snapshot(tmp_path: Path) -> None:
    import json

    from ruhrwatt.gold import serving

    silver(tmp_path, utc(2025, 9, 1), utc(2025, 10, 29), override={PRICE: _noisy_price})
    mart.build(tmp_path / "silver", tmp_path / "gold")
    r = serving.export(tmp_path / "silver", tmp_path / "gold", tmp_path / "serving", days=7)
    out = tmp_path / "serving"
    fresh = json.loads((out / "freshness.json").read_text())
    prices = json.loads((out / "prices.json").read_text())["days"]
    months = json.loads((out / "monthly.json").read_text())["months"]
    assert len(prices) == 7 == r["price_days"] and fresh["price_days"] == sorted(prices)
    assert len(prices["2025-10-26"]) == 100  # 25-hour day
    first = prices["2025-10-27"][0]
    assert first["utc"] == "2025-10-26T23:00:00Z" and first["local"] == "00:00"
    assert months[0]["month"] == "2025-09" and 0 < months[0]["de_renewables_share"] < 100
    assert fresh["last_price_utc"].endswith("Z") and not list(out.glob("*.part"))
