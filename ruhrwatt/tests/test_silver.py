"""Silver layer tests on synthetic bronze files (offline)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from ruhrwatt.bronze.smard import bronze_path
from ruhrwatt.silver import energy, gates
from synthetic import END, PAD, START, _value, utc, write_bronze


@pytest.fixture
def autumn(tmp_path: Path) -> Path:
    return write_bronze(tmp_path / "bronze", START, END, padded_until=PAD)


# --------------------------------------------------------------------------- 15-min table


def test_grid_is_continuous_and_padding_trimmed(autumn: Path) -> None:
    qh = energy.build_15min(autumn)
    assert qh["ts_utc"][0] == START
    assert qh["ts_utc"][-1] == END - timedelta(minutes=15)
    assert (qh["ts_utc"].diff().drop_nulls() == timedelta(minutes=15)).all()
    assert qh["is_complete"].all()


def test_all_series_become_columns(autumn: Path) -> None:
    qh = energy.build_15min(autumn)
    assert len(energy.series()) == 25
    for col in energy.value_columns():
        assert col in qh.columns
    assert qh.columns[:3] == ["ts_utc", "ts_berlin", "date_berlin"]


def test_berlin_calendar_handles_fall_back(autumn: Path) -> None:
    qh = energy.build_15min(autumn)
    sunday = qh.filter(pl.col("date_berlin") == date(2025, 10, 26))
    assert len(sunday) == 100  # 25 hours
    # 02:00-02:59 local occurs twice
    assert sunday.filter(pl.col("hour_berlin") == 2).height == 8


def test_calendar_flags(autumn: Path) -> None:
    qh = energy.build_15min(autumn)
    row = qh.filter(pl.col("ts_utc") == utc(2025, 10, 21, 9)).row(0, named=True)  # Tue 09:00
    assert row["weekday"] == 2 and row["is_peak"] and row["is_workday"]
    row = qh.filter(pl.col("ts_utc") == utc(2025, 10, 21, 20)).row(0, named=True)  # Tue 20:00
    assert not row["is_peak"]
    row = qh.filter(pl.col("ts_utc") == utc(2025, 10, 25, 12)).row(0, named=True)  # Saturday
    assert row["is_weekend"] and not row["is_peak"] and not row["is_workday"]


def test_nrw_holidays_include_regional_days() -> None:
    days = energy.nrw_holidays(range(2025, 2027))
    assert date(2025, 10, 3) in days  # Tag der Deutschen Einheit (national)
    assert date(2025, 11, 1) in days  # Allerheiligen (NRW, not every state)
    assert date(2026, 6, 4) in days  # Fronleichnam 2026
    assert date(2025, 10, 31) not in days  # Reformationstag is not a holiday in NRW


def test_energy_features(autumn: Path) -> None:
    qh = energy.build_15min(autumn)
    r = qh.filter(pl.col("ts_utc") == utc(2025, 10, 22, 12)).row(0, named=True)
    assert r["de_wind_pv_mwh"] == pytest.approx(
        r["de_wind_onshore_mwh"] + r["de_wind_offshore_mwh"] + r["de_pv_mwh"]
    )
    assert r["amp_wind_pv_mwh"] == pytest.approx(r["amp_wind_onshore_mwh"] + r["amp_pv_mwh"])
    assert r["de_renewables_mwh"] == pytest.approx(
        r["de_wind_pv_mwh"] + r["de_biomass_mwh"] + r["de_hydro_mwh"]
    )
    assert r["de_fc_wind_pv_mwh"] == pytest.approx(
        r["de_fc_wind_onshore_mwh"] + r["de_fc_wind_offshore_mwh"] + r["de_fc_pv_mwh"]
    )
    assert "amp_wind_offshore_mwh" not in qh.columns
    assert r["de_fossil_mwh"] == pytest.approx(
        r["de_lignite_mwh"] + r["de_hard_coal_mwh"] + r["de_gas_mwh"]
    )
    assert r["de_wind_pv_share"] == pytest.approx(r["de_wind_pv_mwh"] / r["de_load_mwh"])
    assert r["amp_load_share_of_de"] == pytest.approx(0.3)


def test_price_flags_and_native_cutoff(tmp_path: Path) -> None:
    root = write_bronze(tmp_path / "b", utc(2025, 9, 30, 12), utc(2025, 10, 1, 12))
    qh = energy.build_15min(root)
    before = qh.filter(pl.col("ts_utc") < utc(2025, 10, 1))
    after = qh.filter(pl.col("ts_utc") >= utc(2025, 10, 1))
    assert not before["price_is_native_15min"].any()
    assert after["price_is_native_15min"].all()
    neg = qh.filter(pl.col("price_is_negative"))
    assert len(neg) > 0 and (neg[energy.PRICE] < 0).all()


def test_rolling_features_never_look_ahead(tmp_path: Path) -> None:
    a = energy.build_15min(write_bronze(tmp_path / "a", START, END))
    spike = utc(2025, 10, 23, 12)
    b = energy.build_15min(
        write_bronze(
            tmp_path / "b",
            START,
            END,
            override={energy.PRICE: lambda ts: 3000.0 if ts >= spike else _value(energy.PRICE, ts)},
        )
    )
    cols = ["price_mean_24h", "price_std_24h", "price_delta_15min"]
    before = pl.col("ts_utc") < spike
    assert a.filter(before).select(cols).equals(b.filter(before).select(cols))
    assert not a.filter(~before).select(cols).equals(b.filter(~before).select(cols))


def test_staggered_publication_marks_tail_incomplete(tmp_path: Path) -> None:
    # Price published 1 day further than everything else (day-ahead).
    root = write_bronze(tmp_path / "b", START, END, padded_until=PAD)
    write_bronze(tmp_path / "p", START, END + timedelta(days=1), padded_until=PAD)
    price = bronze_path("day_ahead_price", "DE-LU", "quarterhour", root)
    price.write_bytes(
        bronze_path("day_ahead_price", "DE-LU", "quarterhour", tmp_path / "p").read_bytes()
    )
    qh = energy.build_15min(root)
    assert qh["ts_utc"][-1] == END + timedelta(days=1) - timedelta(minutes=15)
    tail = qh.filter(pl.col("ts_utc") >= END)
    assert len(tail) == 96 and not tail["is_complete"].any()
    assert tail[energy.PRICE].null_count() == 0


def test_missing_bronze_series_gives_clear_error(tmp_path: Path) -> None:
    root = write_bronze(tmp_path / "b", START, END, skip=("amp_pv_mwh",))
    with pytest.raises(FileNotFoundError, match="gen_pv/Amprion"):
        energy.build_15min(root)


# --------------------------------------------------------------------------- daily table


def test_daily_dst_and_completeness(autumn: Path) -> None:
    daily = energy.build_daily(energy.build_15min(autumn)).sort("date_berlin")
    by_day = {r["date_berlin"]: r for r in daily.iter_rows(named=True)}
    assert by_day[date(2025, 10, 26)]["n_qh"] == 100
    assert by_day[date(2025, 10, 26)]["expected_qh"] == 100
    assert by_day[date(2025, 10, 26)]["is_complete_day"]
    assert by_day[date(2025, 10, 21)]["expected_qh"] == 96
    assert not by_day[date(2025, 10, 27)]["is_complete_day"]  # ends at noon


def test_daily_spring_forward_has_92(tmp_path: Path) -> None:
    root = write_bronze(tmp_path / "b", utc(2025, 3, 29), utc(2025, 4, 1))
    daily = energy.build_daily(energy.build_15min(root))
    row = daily.filter(pl.col("date_berlin") == date(2025, 3, 30)).row(0, named=True)
    assert row["n_qh"] == row["expected_qh"] == 92 and row["is_complete_day"]


def test_daily_prices(autumn: Path) -> None:
    qh = energy.build_15min(autumn)
    day = date(2025, 10, 21)
    one = qh.filter(pl.col("date_berlin") == day)
    r = energy.build_daily(qh).filter(pl.col("date_berlin") == day).row(0, named=True)
    p = one[energy.PRICE]
    assert r["price_base"] == pytest.approx(p.mean())
    assert r["price_peak"] == pytest.approx(one.filter(pl.col("is_peak"))[energy.PRICE].mean())
    assert r["price_spread"] == pytest.approx(p.max() - p.min())
    assert r["price_negative_qh"] == (p < 0).sum()
    load = one["de_load_mwh"]
    assert r["de_load_weighted_price"] == pytest.approx((p * load).sum() / load.sum())
    assert r["de_load_mwh"] == pytest.approx(load.sum())


# --------------------------------------------------------------------------- gates


def test_clean_data_passes_all_gates(autumn: Path) -> None:
    result = gates.check(energy.build_15min(autumn))
    assert gates.failed(result) == []
    assert all(g.ok for g in result)


def test_pv_label_swap_is_caught(tmp_path: Path) -> None:
    root = write_bronze(
        tmp_path / "b",
        START,
        END,
        override={"de_pv_mwh": lambda ts: _value("de_wind_onshore_mwh", ts)},
    )
    names = {g.name for g in gates.failed(gates.check(energy.build_15min(root)))}
    # PV is no longer dark at night, and its forecast no longer matches it
    assert names == {"de_pv_dark_at_night", "de_fc_pv_tracks_actual"}


def test_amprion_larger_than_germany_is_caught(tmp_path: Path) -> None:
    root = write_bronze(tmp_path / "b", START, END, override={"amp_gas_mwh": lambda ts: 9_999.0})
    failed = gates.failed(gates.check(energy.build_15min(root)))
    assert [g.name for g in failed] == ["amprion_within_de"]
    assert "gas_mwh" in failed[0].detail


def test_single_inconsistent_row_is_only_a_warning(tmp_path: Path) -> None:
    odd = utc(2025, 10, 22, 7)
    root = write_bronze(
        tmp_path / "b",
        START,
        END,
        override={"de_pv_mwh": lambda ts: 0.0 if ts == odd else _value("de_pv_mwh", ts)},
    )
    result = {g.name: g for g in gates.check(energy.build_15min(root))}
    gate = result["amprion_within_de"]
    assert not gate.ok and gate.severity == "warn"
    assert "pv_mwh @ 2025-10-22 07:00" in gate.detail
    assert gates.failed(list(result.values())) == []


def test_bad_price_and_negative_generation_are_caught(tmp_path: Path) -> None:
    root = write_bronze(
        tmp_path / "b",
        START,
        END,
        override={
            energy.PRICE: lambda ts: 80_000.0,
            "de_lignite_mwh": lambda ts: -1.0,
            "amp_lignite_mwh": lambda ts: -1.0,
        },
    )
    names = {g.name for g in gates.failed(gates.check(energy.build_15min(root)))}
    assert names == {"price_range", "non_negative"}


def test_internal_null_is_a_warning_not_an_error(tmp_path: Path) -> None:
    hole = utc(2025, 10, 22, 3)
    root = write_bronze(
        tmp_path / "b",
        START,
        END,
        override={"de_gas_mwh": lambda ts: None if ts == hole else _value("de_gas_mwh", ts)},
    )
    result = {g.name: g for g in gates.check(energy.build_15min(root))}
    assert not result["internal_nulls"].ok
    assert result["internal_nulls"].severity == "warn"
    assert "de_gas_mwh" in result["internal_nulls"].detail


# --------------------------------------------------------------------------- end to end


def test_build_writes_both_tables(autumn: Path, tmp_path: Path) -> None:
    out = energy.build(autumn, tmp_path / "silver")
    qh = pl.read_parquet(out["paths"]["15min"])
    daily = pl.read_parquet(out["paths"]["daily"])
    assert qh.equals(out["qh"]) and daily.equals(out["daily"])
    assert qh.schema["ts_utc"] == pl.Datetime("ms", "UTC")
    assert qh.schema["ts_berlin"] == pl.Datetime("ms", "Europe/Berlin")
    assert not list((tmp_path / "silver").glob("*.part"))


def test_cli_build_reports_gates(tmp_path: Path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from ruhrwatt.cli import app

    write_bronze(tmp_path / "bronze" / "smard", START, END, padded_until=PAD)
    monkeypatch.setenv("RUHRWATT_DATA_DIR", str(tmp_path))
    result = CliRunner().invoke(app, ["silver", "build"])
    assert result.exit_code == 0, result.output
    assert "[FAIL]" not in result.output
    assert "amprion_within_de" in result.output
    assert (tmp_path / "silver" / "energy_daily.parquet").exists()


def test_wrong_forecast_id_is_caught(tmp_path: Path) -> None:
    # "forecast" that is really PV: does not track onshore wind
    root = write_bronze(
        tmp_path / "b",
        START,
        END,
        override={"amp_fc_wind_onshore_mwh": lambda ts: _value("amp_pv_mwh", ts)},
    )
    names = {g.name for g in gates.failed(gates.check(energy.build_15min(root)))}
    assert names == {"amp_fc_wind_onshore_tracks_actual"}


def test_daily_sums_are_null_not_zero_without_data(tmp_path: Path) -> None:
    # prices published a day further than everything else: the last day has no load data
    root = write_bronze(tmp_path / "b", START, END, padded_until=PAD)
    p = write_bronze(tmp_path / "p", START, END + timedelta(days=1), padded_until=PAD)
    src = bronze_path("day_ahead_price", "DE-LU", "quarterhour", p)
    bronze_path("day_ahead_price", "DE-LU", "quarterhour", root).write_bytes(src.read_bytes())
    daily = energy.build_daily(energy.build_15min(root))
    last = daily.filter(pl.col("date_berlin") == date(2025, 10, 28)).row(0, named=True)
    assert last["price_base"] is not None
    assert last["de_load_mwh"] is None and last["de_load_weighted_price"] is None
    assert last["de_wind_pv_share"] is None


def test_missing_java_gives_install_hint(monkeypatch) -> None:
    import shutil

    from typer.testing import CliRunner

    from ruhrwatt.cli import app
    from ruhrwatt.spark import session

    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(session.JavaMissingError, match="openjdk-21"):
        session.check_java()
    r = CliRunner().invoke(app, ["spark", "parity"])
    assert r.exit_code == 1 and "openjdk-21" in r.output
