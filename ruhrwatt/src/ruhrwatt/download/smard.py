"""SMARD (Bundesnetzagentur) raw downloader.

Pulls the weekly ``chart_data`` JSON files byte-for-byte into
``<data>/raw/smard/<filter>/<region>/<resolution>/``. No parsing happens here:
the raw layer stays untouched so the Rust parser can be re-run at any time.

Layout of the chart_data API (verified against live files by third-party
loaders, re-check on first run):

* ``{BASE}/{filter}/{region}/index_{resolution}.json`` -> ``{"timestamps": [ms, ...]}``
* ``{BASE}/{filter}/{region}/{filter}_{region}_{resolution}_{ts}.json``
  -> ``{"meta_data": {...}, "series": [[ts_ms, value|null], ...]}``

Files are weekly, start Monday 00:00 Europe/Berlin, and use epoch-ms UTC timestamps.
Data: Bundesnetzagentur | SMARD.de, CC BY 4.0.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

BASE_URL = "https://www.smard.de/app/chart_data"
ATTRIBUTION = "Bundesnetzagentur | SMARD.de (CC BY 4.0)"
USER_AGENT = "ruhrwatt/0.1 (+https://github.com/; research, polite client)"

RESOLUTIONS = ("quarterhour", "hour", "day", "week", "month", "year")

# Step between observations in ms; 0 = no gap/alignment checks (day and longer start at Berlin
# midnight, so their step is not constant in UTC).
STEP_MS = {"quarterhour": 900_000, "hour": 3_600_000}

# The day-ahead market became natively 15-minute on 2025-10-01 00:00 Europe/Berlin
# (= 2025-09-30T22:00:00Z). Before that SMARD repeats each hourly price four times.
DAY_AHEAD_NATIVE_15MIN_FROM_MS = 1_759_269_600_000


@dataclass(frozen=True)
class Dataset:
    name: str
    filter_id: int
    regions: tuple[str, ...]
    description: str
    verified: bool = False  # True only once confirmed against a live file
    native_from_ms: int | None = None  # None = every row is native resolution


# All 25 series downloaded live and passed the silver gates (PV dark at night, Amprion <=
# Germany, every forecast correlates >= 0.93 with its actual), so ids and labels are
# verified. A new series should be added with verified=False until it has done the same.
TWO = ("DE", "Amprion")
DATASETS: dict[str, Dataset] = {
    d.name: d
    for d in (
        Dataset(
            "day_ahead_price",
            4169,
            ("DE-LU",),
            "Day-ahead price, EUR/MWh",
            verified=True,
            native_from_ms=DAY_AHEAD_NATIVE_15MIN_FROM_MS,
        ),
        Dataset("grid_load", 410, TWO, "Grid load, MWh", verified=True),
        Dataset("residual_load", 4359, TWO, "Residual load, MWh", verified=True),
        Dataset("gen_lignite", 1223, TWO, "Generation: lignite", verified=True),
        Dataset("gen_wind_onshore", 4067, TWO, "Generation: wind onshore", verified=True),
        Dataset("gen_pv", 4068, TWO, "Generation: photovoltaics", verified=True),
        Dataset("gen_hard_coal", 4069, TWO, "Generation: hard coal", verified=True),
        Dataset("gen_gas", 4071, TWO, "Generation: fossil gas", verified=True),
        # Offshore wind lands in the TenneT and 50Hertz areas, so Amprion has none.
        Dataset("gen_wind_offshore", 1225, ("DE",), "Generation: wind offshore", verified=True),
        Dataset("gen_biomass", 4066, TWO, "Generation: biomass", verified=True),
        Dataset("gen_hydro", 1226, TWO, "Generation: hydropower", verified=True),
        # TSO day-ahead forecasts for the delivery day.
        Dataset("fc_wind_onshore", 123, TWO, "Forecast generation: wind onshore", verified=True),
        Dataset(
            "fc_wind_offshore", 3791, ("DE",), "Forecast generation: wind offshore", verified=True
        ),
        Dataset("fc_pv", 125, TWO, "Forecast generation: photovoltaics", verified=True),
    )
}


@dataclass
class Stats:
    fetched: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


def make_client(transport: httpx.BaseTransport | None = None) -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(30.0),
        follow_redirects=True,
        transport=transport,
    )


def index_url(filter_id: int, region: str, resolution: str) -> str:
    return f"{BASE_URL}/{filter_id}/{region}/index_{resolution}.json"


def week_url(filter_id: int, region: str, resolution: str, ts_ms: int) -> str:
    return f"{BASE_URL}/{filter_id}/{region}/{filter_id}_{region}_{resolution}_{ts_ms}.json"


def target_dir(root: Path, filter_id: int, region: str, resolution: str) -> Path:
    return root / str(filter_id) / region / resolution


def _get_with_retry(
    client: httpx.Client, url: str, retries: int = 4, backoff: float = 1.0
) -> httpx.Response:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.get(url)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
            resp.raise_for_status()
            return resp
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            last = exc
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 429:
                raise  # 4xx other than 429: retrying will not help
            time.sleep(backoff * 2**attempt)
    assert last is not None
    raise last


def fetch_index(
    client: httpx.Client, filter_id: int, region: str, resolution: str = "quarterhour"
) -> list[int]:
    """All weekly timestamps (epoch ms) SMARD has for this series, ascending."""
    resp = _get_with_retry(client, index_url(filter_id, region, resolution))
    return sorted(int(t) for t in resp.json()["timestamps"])


def _ts_to_date(ts_ms: int) -> date:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).date()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(content)
    tmp.replace(path)


# SMARD keeps filling a week for a while after it ends (load and generation come with a
# lag, values get revised). A file downloaded before then is not final.
SETTLE_DAYS = 3
WEEK_MS = 7 * 86_400_000


def is_final(path: Path, week_ts_ms: int, settle_days: int = SETTLE_DAYS) -> bool:
    """True if ``path`` was downloaded at least ``settle_days`` after its week ended."""
    final_from_s = (week_ts_ms + WEEK_MS) / 1000 + settle_days * 86_400
    return path.stat().st_mtime >= final_from_s


def _validate_payload(content: bytes) -> None:
    payload = json.loads(content)
    if "series" not in payload:
        raise ValueError("payload has no 'series' key")


def download_series(
    client: httpx.Client,
    root: Path,
    filter_id: int,
    region: str,
    resolution: str = "quarterhour",
    since: date | None = None,
    refresh_weeks: int = 2,
    delay: float = 0.2,
) -> Stats:
    """Download all weekly files for one series. Resumable and idempotent.

    Existing files are skipped, except the newest ``refresh_weeks`` weeks and any file that
    was downloaded before its week was final (see ``is_final``): such a file is a partial
    snapshot of a week SMARD was still publishing, and is re-fetched until it is complete.
    """
    stats = Stats()
    timestamps = fetch_index(client, filter_id, region, resolution)
    if since is not None:
        # keep the week that contains `since` (it starts on or before it)
        earlier = [t for t in timestamps if _ts_to_date(t) <= since]
        cutoff = earlier[-1] if earlier else timestamps[0]
        timestamps = [t for t in timestamps if t >= cutoff]

    refresh = set(timestamps[-refresh_weeks:]) if refresh_weeks > 0 else set()
    out_dir = target_dir(root, filter_id, region, resolution)
    manifest = out_dir / "_manifest.jsonl"

    for ts in timestamps:
        path = out_dir / f"{filter_id}_{region}_{resolution}_{ts}.json"
        if path.exists() and ts not in refresh and is_final(path, ts):
            stats.skipped += 1
            continue
        url = week_url(filter_id, region, resolution, ts)
        try:
            content = _get_with_retry(client, url).content
            _validate_payload(content)
        except (httpx.HTTPError, ValueError) as exc:
            stats.failed += 1
            stats.errors.append(f"{url}: {exc}")
            continue
        _atomic_write(path, content)
        record = {
            "file": path.name,
            "url": url,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "attribution": ATTRIBUTION,
        }
        with manifest.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        stats.fetched += 1
        time.sleep(delay)
    return stats
