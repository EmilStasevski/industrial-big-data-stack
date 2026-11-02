"""Walk-forward comparison of day-ahead price forecasting models.

The most recent ``test_weeks`` are cut into blocks of ``retrain_days``. For each block every
model is trained on all days before it (expanding window) and predicts the block, like a
desk that retrains weekly. All models are scored on exactly the same rows.

Models:

* ``naive_d1`` / ``naive_d7``: same quarter-hour yesterday / last week (the bar to beat)
* ``ridge``: linear model, one-hot hour and weekday (how far a linear model gets)
* ``hgb``: scikit-learn HistGradientBoosting
* ``lightgbm``, ``xgboost``, ``catboost``: the three common boosting libraries
* ``lightgbm_no_fc``: LightGBM without the day-D wind/PV forecasts (value of the forecasts)

The tree models use comparable settings (500 trees, learning rate 0.05, absolute-error
loss) rather than per-model tuning, so differences reflect the libraries, not the tuning
effort.

Price ranges: LightGBM quantile models (alpha 0.1 / 0.9) give a raw P10-P90 range. Raw
quantile models are often overconfident, so the range is conformalized (CQR, Romano et al.
2019) and recalibrated every day:

* the quantile models are fitted on the training days except the last ``CALIBRATION_DAYS``;
* for each test day D, the misses over the ``CALIBRATION_DAYS`` before D (calibration days
  plus earlier test days, whose prices are all known before D's auction) set a margin;
* misses are measured relative to the raw range width, so the margin widens a range in
  proportion: volatile quarter-hours (wide raw range) get more room than calm ones.

Recalibrating daily lets the range follow regime changes such as rising autumn volatility.
Both raw and calibrated coverage are reported.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date, timedelta

import numpy as np
import polars as pl

from ruhrwatt.gold.features import TARGET, feature_columns

N_TREES = 500
LEARNING_RATE = 0.05
QUANTILES = (0.1, 0.9)
CALIBRATION_DAYS = 14  # trailing days whose misses calibrate each day's price range
WIDTH_FLOOR = 5.0  # EUR/MWh: smallest raw width used to scale misses (avoids division by ~0)


def scaled_scores(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Misses relative to the raw range width (negative inside the range)."""
    return np.maximum(lo - y, y - hi) / np.maximum(hi - lo, WIDTH_FLOOR)


def daily_margins(
    days: np.ndarray, scores: np.ndarray, test_days: np.ndarray, level: float, window: int
) -> np.ndarray:
    """Margin for every test row: from scores of rows dated in [D - window, D - 1] only.

    ``days``/``scores`` cover the calibration rows and the test rows (whose scores become
    known day by day); a test day never sees its own or later scores.
    """
    out = np.zeros(len(test_days))
    for d in np.unique(test_days):
        past = (days < d) & (days >= d - np.timedelta64(window, "D"))
        if past.sum() < 10:
            continue  # too little history: keep the raw range
        idx = test_days == d
        n = int(past.sum())
        q = min(1.0, np.ceil((n + 1) * level) / n)
        out[idx] = np.quantile(scores[past], q, method="higher")
    return out


def conformal_margin(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, level: float) -> float:
    """CQR margin: widening that makes [lo - m, hi + m] cover ``level`` of the calibration set.

    Conformity score per row is max(lo - y, y - hi): negative inside the range, positive by how
    far it missed. The margin is its finite-sample-corrected ``level`` quantile. Can be
    negative (the range is shrunk) when the raw range was too wide.
    """
    scores = np.maximum(lo - y, y - hi)
    n = len(scores)
    q = min(1.0, np.ceil((n + 1) * level) / n)
    return float(np.quantile(scores, q, method="higher"))


@dataclass(frozen=True)
class Score:
    model: str
    mae: float
    rmse: float
    skill_vs_naive_d1: float  # 1 - MAE / MAE(naive_d1); 0.10 = 10 % better than "yesterday"
    n: int


# --------------------------------------------------------------------------- models


def _ridge(cols: list[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cat = [cols.index(c) for c in ("hour_berlin", "weekday")]
    num = [i for i in range(len(cols)) if i not in cat]
    prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ]
    )
    return make_pipeline(prep, Ridge(alpha=1.0))


def _hgb(seed: int):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=N_TREES,
        learning_rate=LEARNING_RATE,
        max_leaf_nodes=63,
        random_state=seed,
    )


def _lightgbm(seed: int, objective: str = "l1", alpha: float | None = None):
    from lightgbm import LGBMRegressor

    extra = {"alpha": alpha} if alpha is not None else {}
    return LGBMRegressor(
        objective=objective,
        n_estimators=N_TREES,
        learning_rate=LEARNING_RATE,
        num_leaves=63,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        verbose=-1,
        **extra,
    )


def _xgboost(seed: int):
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=N_TREES,
        learning_rate=LEARNING_RATE,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=seed,
    )


def _catboost(seed: int):
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        loss_function="MAE",
        iterations=N_TREES,
        learning_rate=LEARNING_RATE * 2,  # symmetric trees learn more per step at depth 8
        depth=8,
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
    )


@dataclass(frozen=True)
class Spec:
    make: Callable[[int, list[str]], object] | None  # None = naive baseline
    forecasts: bool = True
    naive_col: str | None = None


MODELS: dict[str, Spec] = {
    "naive_d1": Spec(None, naive_col="price_lag_1d"),
    "naive_d7": Spec(None, naive_col="price_lag_7d"),
    "ridge": Spec(lambda seed, cols: _ridge(cols)),
    "hgb": Spec(lambda seed, cols: _hgb(seed)),
    "lightgbm": Spec(lambda seed, cols: _lightgbm(seed)),
    "xgboost": Spec(lambda seed, cols: _xgboost(seed)),
    "catboost": Spec(lambda seed, cols: _catboost(seed)),
    "lightgbm_no_fc": Spec(lambda seed, cols: _lightgbm(seed), forecasts=False),
}
DEFAULT_MODELS = tuple(MODELS)


# --------------------------------------------------------------------------- folds


def folds(
    features: pl.DataFrame, test_weeks: int = 8, retrain_days: int = 7
) -> list[tuple[date, date]]:
    """(start, end) of each test block; end is exclusive."""
    known = features.filter(pl.col(TARGET).is_not_null())
    if known.height == 0:
        raise ValueError("no rows with a known price")
    last = known["date_berlin"].max()
    first_test = last - timedelta(days=7 * test_weeks - 1)
    if known.filter(pl.col("date_berlin") < first_test).height == 0:
        raise ValueError(f"not enough data for a {test_weeks}-week test window")
    out, start = [], first_test
    while start <= last:
        end = min(start + timedelta(days=retrain_days), last + timedelta(days=1))
        out.append((start, end))
        start = end
    return out


def _matrix(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return df.select(pl.col(cols).cast(pl.Float64)).to_numpy()


# --------------------------------------------------------------------------- run


def run(
    features: pl.DataFrame,
    test_weeks: int = 8,
    retrain_days: int = 7,
    models: tuple[str, ...] = DEFAULT_MODELS,
    intervals: bool = True,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Walk-forward backtest. Returns scores, per-hour errors, importance and predictions."""
    unknown = set(models) - set(MODELS)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}; choose from {list(MODELS)}")
    if "naive_d1" not in models:
        models = ("naive_d1", *models)  # needed as the skill reference
    known = features.filter(pl.col(TARGET).is_not_null())
    blocks = folds(features, test_weeks, retrain_days)
    parts: list[pl.DataFrame] = []
    importance = None

    for i, (start, end) in enumerate(blocks, 1):
        train = known.filter(pl.col("date_berlin") < start)
        test = known.filter(pl.col("date_berlin").is_between(start, end, closed="left"))
        if test.height == 0:
            continue
        if progress:
            progress(f"fold {i}/{len(blocks)}: train {train.height} rows, test {start} .. {end}")
        y_train = train[TARGET].to_numpy()
        part = test.select("ts_utc", "ts_berlin", "date_berlin", "hour_berlin", TARGET)
        part = part.with_columns(pl.lit(i).alias("fold"))
        for name in models:
            spec = MODELS[name]
            if spec.make is None:
                pred = test[spec.naive_col].cast(pl.Float64).to_numpy()
            else:
                cols = feature_columns(forecasts=spec.forecasts)
                model = spec.make(seed, cols)
                model.fit(_matrix(train, cols), y_train)
                pred = model.predict(_matrix(test, cols))
                if name == "lightgbm":
                    importance = (cols, model.booster_.feature_importance("gain"))
            part = part.with_columns(pl.Series(f"pred_{name}", pred, dtype=pl.Float64))
        if intervals:
            cols = feature_columns()
            cal_start = start - timedelta(days=CALIBRATION_DAYS)
            fit = train.filter(pl.col("date_berlin") < cal_start)
            cal = train.filter(pl.col("date_berlin") >= cal_start)
            if fit.height == 0 or cal.height == 0:  # too little history: raw range only
                fit, cal = train, train.clear()
            bands = {}
            for q in QUANTILES:
                m = _lightgbm(seed, objective="quantile", alpha=q)
                m.fit(_matrix(fit, cols), fit[TARGET].to_numpy())
                bands[q] = (m.predict(_matrix(cal, cols)), m.predict(_matrix(test, cols)))
            (lo_cal, lo_test), (hi_cal, hi_test) = bands[QUANTILES[0]], bands[QUANTILES[1]]
            level = QUANTILES[1] - QUANTILES[0]
            y_test = test[TARGET].to_numpy()
            days = np.concatenate([cal["date_berlin"].to_numpy(), test["date_berlin"].to_numpy()])
            scores = np.concatenate(
                [
                    scaled_scores(lo_cal, hi_cal, cal[TARGET].to_numpy()),
                    scaled_scores(lo_test, hi_test, y_test),
                ]
            )
            margin = daily_margins(
                days, scores, test["date_berlin"].to_numpy(), level, CALIBRATION_DAYS
            )
            scale = np.maximum(hi_test - lo_test, WIDTH_FLOOR)
            part = part.with_columns(
                pl.Series("p10_raw", lo_test),
                pl.Series("p90_raw", hi_test),
                pl.Series("p10", lo_test - margin * scale),
                pl.Series("p90", hi_test + margin * scale),
                pl.Series("conformal_margin", margin),
            )
        parts.append(part)

    preds = pl.concat(parts)
    pred_cols = [f"pred_{m}" for m in models]
    # Same rows for every model: baselines are null on the spring DST hour and early days.
    common = preds.filter(
        pl.all_horizontal(pl.col(pred_cols).is_not_nan() & pl.col(pred_cols).is_not_null())
    )
    y = common[TARGET].to_numpy()
    raw = {}
    for m in models:
        err = y - common[f"pred_{m}"].to_numpy()
        raw[m] = (float(np.abs(err).mean()), float(np.sqrt((err**2).mean())))
    ref = raw["naive_d1"][0]
    scores = sorted(
        (Score(m, mae, rmse, 1 - mae / ref, len(y)) for m, (mae, rmse) in raw.items()),
        key=lambda s: s.mae,
    )
    by_hour = (
        common.group_by("hour_berlin")
        .agg(
            (pl.col(c) - pl.col(TARGET)).abs().mean().alias(c.removeprefix("pred_"))
            for c in pred_cols
        )
        .sort("hour_berlin")
    )
    result = {
        "scores": scores,
        "scores_df": pl.DataFrame([asdict(s) for s in scores]),
        "by_hour": by_hour,
        "folds": blocks,
        "test_rows": len(y),
        "predictions": preds,
        "importance": None,
        "interval": None,
    }
    if importance is not None:
        cols, gain = importance
        result["importance"] = (
            pl.DataFrame({"feature": cols, "gain": gain.astype(float)})
            .with_columns(share=pl.col("gain") / pl.col("gain").sum())
            .sort("gain", descending=True)
        )
    if intervals:
        y_s = common[TARGET]

        def cover(lo: str, hi: str) -> float:
            return float(((y_s >= common[lo]) & (y_s <= common[hi])).mean())

        result["interval"] = {
            "coverage_raw": cover("p10_raw", "p90_raw"),
            "width_raw": float((common["p90_raw"] - common["p10_raw"]).mean()),
            "coverage_p10_p90": cover("p10", "p90"),
            "mean_width": float((common["p90"] - common["p10"]).mean()),
        }
    return result
