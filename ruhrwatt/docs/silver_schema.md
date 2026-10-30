# Silver layer: energy tables

Built by `make silver` (`ruhrwatt silver build`) from the 25 bronze SMARD series with Polars.
Two files in `data/silver/`, both zstd Parquet:

## `energy_15min.parquet`: one row per quarter-hour

All series are joined onto one continuous 15-minute UTC grid. SMARD's padded future rows
(all null) are trimmed. Rows where only some series are published yet (prices come a day
ahead, load and generation later) are kept with `is_complete = false`.

Units: SMARD quarter-hour values are **MWh per quarter-hour**. Multiply by 4 for average MW.
Prefixes: `de_` = Germany, `amp_` = Amprion control area (proxy for NRW).

| Column | Notes |
|---|---|
| `ts_utc` | timestamp[ms, UTC], start of the quarter-hour, join key |
| `ts_berlin`, `date_berlin`, `hour_berlin` | local time; DST days have 92 or 100 rows |
| `weekday` | 1 = Monday ... 7 = Sunday |
| `is_weekend`, `is_holiday_nrw`, `is_workday` | NRW public holidays (incl. Fronleichnam, Allerheiligen) |
| `is_peak` | EEX peak: Mon-Fri 08:00-20:00 local (holidays not excluded) |
| `is_complete` | every one of the 25 series has a value |
| `price_eur_mwh` | day-ahead DE-LU |
| `price_is_native_15min` | false before 2025-10-01: hourly price repeated four times |
| `price_is_negative`, `price_delta_15min` | |
| `price_mean_24h`, `price_std_24h` | trailing 96 quarter-hours, current row included, never future rows |
| `{de,amp}_load_mwh`, `_residual_load_mwh` | grid load; load minus wind and PV (can be negative) |
| `{de,amp}_lignite_mwh`, `_hard_coal_mwh`, `_gas_mwh` | fossil generation |
| `{de,amp}_wind_onshore_mwh`, `de_wind_offshore_mwh`, `{de,amp}_pv_mwh` | variable renewables (Amprion has no offshore wind) |
| `{de,amp}_biomass_mwh`, `_hydro_mwh` | other renewables |
| `{de,amp}_fc_wind_onshore_mwh`, `de_fc_wind_offshore_mwh`, `{de,amp}_fc_pv_mwh` | TSO day-ahead forecasts for the same quarter-hour |
| `{de,amp}_wind_pv_mwh`, `_wind_pv_share` | wind (onshore + offshore) + PV, and as share of load |
| `{de,amp}_renewables_mwh`, `_renewables_share` | wind + PV + biomass + hydro, and as share of load |
| `{de,amp}_fc_wind_pv_mwh` | forecast wind + PV |
| `{de,amp}_fossil_mwh`, `_fossil_share` | lignite + hard coal + gas, and as share of load |
| `amp_load_share_of_de` | Amprion load / German load |

`renewables_share` covers wind, PV, biomass and hydro; SMARD's "other renewables" (a small
remainder) is not included.

## `energy_daily.parquet`: one row per Berlin day

| Column | Notes |
|---|---|
| `date_berlin`, `weekday`, `is_weekend`, `is_holiday_nrw` | |
| `n_qh`, `expected_qh`, `n_complete`, `is_complete_day` | filter on `is_complete_day` for analysis |
| `price_base`, `price_peak`, `price_offpeak` | means over all, peak and off-peak quarter-hours |
| `price_min`, `price_max`, `price_spread` | spread = max - min (flexibility / storage value) |
| `price_negative_qh` | quarter-hours with a negative price |
| `{de,amp}_load_mwh`, `_residual_load_mwh`, `_wind_pv_mwh`, `_fossil_mwh` | daily sums |
| `{de,amp}_wind_pv_share`, `_fossil_share` | ratios of the daily sums |
| `{de,amp}_load_weighted_price` | what a consumer with that load profile paid on average |

## Quality gates

`silver build` prints every gate and exits 1 if an error-level gate fails (files are still
written so you can inspect them).

| Gate | Level | Checks |
|---|---|---|
| `on_grid` | error | no bronze row lies off the 15-minute UTC grid |
| `continuous_grid` | error | strictly increasing, exactly 15 minutes apart |
| `price_range` | error | prices within -500 .. 5000 EUR/MWh (catches unit/parsing errors) |
| `non_negative` | error | load and generation are never negative |
| `amprion_within_de` | error / warn | Amprion never exceeds Germany (catches swapped regions or ids). A handful of rows (up to max(10, 0.1 %)) is a SMARD inconsistency and only warns, with an example row |
| `{de,amp}_pv_dark_at_night`, `{de,amp}_fc_pv_dark_at_night` | error | PV and PV forecast at 21:00-01:59 UTC < 1 % of their mean (catches a wrong PV id) |
| `*_fc_*_tracks_actual` | error | each forecast correlates >= 0.8 with the series it predicts (catches a wrong forecast id) |
| `internal_nulls` | warn | missing values before the last complete quarter-hour |
