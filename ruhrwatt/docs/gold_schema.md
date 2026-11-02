# Gold layer

Built by `make gold` (`ruhrwatt gold build`) from the silver tables. Three files in
`data/gold/`, plus `backtest_predictions.parquet` from `make backtest`.

## `price_features_15min.parquet`: day-ahead price forecasting

One row per delivery quarter-hour. Target: `price_eur_mwh`.

**Information set.** The DE-LU day-ahead auction for day D closes on D-1 at 12:00 Berlin.
No *actual* value from day D enters a feature: a test changes every price, load and
generation value from day D on and checks that day D's features stay identical. The only
day-D inputs are forecasts.

**Forecast caveat.** TSOs must publish day-ahead wind/PV forecasts by 18:00 on D-1, which can
be after the auction. Traders have their own weather-based forecasts of similar quality
before noon, so the TSO forecast stands in for that information. A live system would use a
forecast vintage from before 12:00. SMARD has no load forecast, so load enters as lags.

| Feature | Known because |
|---|---|
| `hour_berlin`, `qh_of_day` (0-95), `weekday`, `month`, `doy_sin`, `doy_cos` | calendar |
| `is_workday`, `is_holiday_nrw`, `is_peak` | calendar |
| `price_lag_1d`, `price_lag_2d`, `price_lag_7d` | D-1 prices come from the auction on D-2 |
| `price_d1_mean`, `_min`, `_max`, `_spread` | whole of D-1, also known since D-2 |
| `{de,amp}_*_lag_2d`, `_lag_7d` for load, residual load, wind+PV (and DE fossil) | actuals of D-2 are published by D-1 noon; D-1 is not complete yet |
| `de_fc_wind_onshore/offshore/pv/wind_pv_mwh`, `amp_fc_wind_onshore/pv/wind_pv_mwh` | TSO day-ahead forecasts for D (see caveat) |
| `de_fc_wind_pv_day_mean`, `_day_max` | forecast wind + PV over the whole of day D |
| `de_fc_residual_proxy`, `de_fc_wind_pv_share_proxy` | load 7 days earlier minus / divided by forecast wind + PV |

Lags use **Berlin wall-clock time** ("same quarter-hour k days earlier"), so a 08:00 load
peak is compared with yesterday's 08:00 even across a clock change. On the 25-hour autumn
day the repeated 02:00 hour is looked up once (first occurrence); on the 23-hour spring day
the missing 02:00 hour gives null lags (the gradient-boosting model handles nulls natively).

Silver's rolling 24 h price columns include the current row and are deliberately not used.

Not included yet: load forecasts, gas (TTF) and CO2 (EUA) prices, cross-border flows.

### Model comparison (`make backtest`)

Walk-forward: the last 8 weeks (`--test-weeks`) are split into weekly blocks
(`--retrain-days 7`); for each block every model is retrained on all earlier days and
predicts the block. All models are scored on exactly the same rows. `--models a,b,c` picks
a subset; the full run takes a few minutes (CatBoost is the slowest).

| Model | What it is |
|---|---|
| `naive_d1` | same quarter-hour yesterday (the reference; skill = 1 - MAE / MAE of this) |
| `naive_d7` | same quarter-hour last week |
| `ridge` | linear model, one-hot hour and weekday |
| `hgb` | scikit-learn HistGradientBoosting |
| `lightgbm`, `xgboost`, `catboost` | the three common boosting libraries |
| `lightgbm_no_fc` | LightGBM without the day-D forecasts: shows what the forecasts are worth |

Tree models share settings (500 trees, learning rate 0.05, absolute-error loss; CatBoost
uses 0.1 for its symmetric trees) instead of per-model tuning.

**Price range (P10-P90).** LightGBM quantile models (alpha 0.1 / 0.9) are fitted on all
training days except the last 14. The range is then conformalized (CQR) and **recalibrated
every day**: for day D, the misses over the 14 days before D (all known before D's auction)
set the margin. Misses are measured relative to the raw range width, so volatile
quarter-hours get proportionally more room. Raw quantile models are overconfident and prices
shift between seasons (a weekly calibration reached only 69 % in autumn 2026), so the output
reports raw and calibrated coverage side by side.
`backtest_predictions` holds `p10_raw`, `p90_raw`, `p10`, `p90` and the daily `conformal_margin`
(in units of raw range width).

Outputs in `data/gold/`: `backtest_scores`, `backtest_by_hour` (MAE per hour of day),
`backtest_predictions` (all predictions plus `p10`/`p90`), `feature_importance` (LightGBM
gain on the last fold).

## `load_shift_daily.parquet`: load-shifting economics

One row per plant and day with complete day-ahead prices.

**Model.** A plant averages `avg_mw` and must use the same energy each day, but may draw
between `min_frac` and `max_frac` of its average in every quarter-hour (shifting within the
day, e.g. by buffering intermediate products). Minimising the energy bill under those limits
has an exact greedy solution: minimum everywhere, the rest into the cheapest quarter-hours at
maximum. Tests check it against a general LP solver (scipy HiGHS) on normal and DST days.

| Column | Notes |
|---|---|
| `plant`, `date_berlin`, `n_qh` | |
| `energy_mwh` | the day's energy (same for flat and shifted) |
| `cost_flat_eur`, `cost_shifted_eur`, `savings_eur` | day-ahead energy cost only |
| `savings_eur_per_mwh` | savings per MWh consumed |
| `price_flat_eur_mwh`, `price_shifted_eur_mwh` | average price paid |
| `moved_mwh` | energy moved between quarter-hours |
| `negative_price_mwh` | energy bought at negative prices after shifting |

**Presets** are illustrative assumptions for typical NRW process types, not data of real
plants: `generic_10pct` (100 MW, 90-110 %), `chlor_alkali` (60 MW, 60-115 %), `aluminium`
(250 MW, 85-105 %), `steel_eaf` (120 MW, 30-160 %). Use
`ruhrwatt gold shift --avg-mw 40 --min-frac 0.5 --max-frac 1.5` for your own numbers.

**Upper bound.** Not modelled: ramp rates, minimum run times, production schedules beyond
one day, grid fees and levies (which do not change with the hour), intraday markets.
Real savings are lower; the ranking between plants and months is the useful signal.

## `market_monthly.parquet`: monthly market facts

One row per month over complete days. The RAG assistant answers from this table.

`days`, `price_base`, `price_peak`, `price_offpeak`, `price_min`, `price_max`,
`price_daily_spread_avg`, `negative_price_hours`, `days_with_negative_prices`,
`{de,amp}_wind_pv_share`, `{de,amp}_renewables_share`, `{de,amp}_fossil_share`, `amp_load_share_of_de`,
`{de,amp}_load_weighted_price`, and `shift_savings_eur_mwh_<plant>` per preset.
