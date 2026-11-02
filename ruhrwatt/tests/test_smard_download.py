"""Offline tests for the SMARD downloader, using httpx.MockTransport."""

from __future__ import annotations

import json
from datetime import date

import httpx
import pytest

from ruhrwatt.download import smard

# Mondays 00:00 Europe/Berlin, as epoch ms UTC (CEST = UTC+2 in October 2025)
W1 = 1759096800000  # 2025-09-29 00:00 CEST
W2 = 1759701600000  # 2025-10-06
W3 = 1760306400000  # 2025-10-13
W4 = 1760911200000  # 2025-10-20


def _payload(ts: int, gap: bool = False) -> dict:
    middle = None if gap else 7.0
    return {"meta_data": {}, "series": [[ts, 41.5], [ts + 900_000, middle], [ts + 1_800_000, -3.2]]}


class FakeSmard:
    def __init__(self, timestamps: list[int]) -> None:
        self.timestamps = timestamps
        self.calls: list[str] = []
        self.fail_week: int | None = None
        self.gaps = False  # True: every week contains a null

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.calls.append(url)
        if url.endswith("index_quarterhour.json"):
            return httpx.Response(200, json={"timestamps": self.timestamps})
        for ts in self.timestamps:
            if url.endswith(f"_{ts}.json"):
                if ts == self.fail_week:
                    return httpx.Response(404)
                return httpx.Response(200, content=json.dumps(_payload(ts, self.gaps)).encode())
        return httpx.Response(404)


def _client(fake: FakeSmard) -> httpx.Client:
    return smard.make_client(httpx.MockTransport(fake))


def test_urls() -> None:
    assert smard.index_url(4169, "DE-LU", "quarterhour").endswith(
        "/4169/DE-LU/index_quarterhour.json"
    )
    assert smard.week_url(4169, "DE-LU", "quarterhour", W1).endswith(
        f"/4169/DE-LU/4169_DE-LU_quarterhour_{W1}.json"
    )


def test_downloads_all_weeks_and_writes_manifest(tmp_path) -> None:
    fake = FakeSmard([W1, W2, W3])
    with _client(fake) as client:
        stats = smard.download_series(client, tmp_path, 4169, "DE-LU", delay=0)
    assert (stats.fetched, stats.skipped, stats.failed) == (3, 0, 0)
    out = tmp_path / "4169" / "DE-LU" / "quarterhour"
    assert len(list(out.glob("*.json"))) == 3
    lines = (out / "_manifest.jsonl").read_text().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["attribution"] == smard.ATTRIBUTION


def test_resume_skips_old_files_but_refreshes_newest(tmp_path) -> None:
    fake = FakeSmard([W1, W2, W3, W4])
    with _client(fake) as client:
        smard.download_series(client, tmp_path, 4169, "DE-LU", delay=0)
        fake.calls.clear()
        stats = smard.download_series(client, tmp_path, 4169, "DE-LU", refresh_weeks=2, delay=0)
    # W1, W2 skipped; W3, W4 (newest two) re-fetched
    assert (stats.fetched, stats.skipped) == (2, 2)
    week_calls = [c for c in fake.calls if "index_" not in c]
    assert len(week_calls) == 2


def test_since_keeps_the_containing_week(tmp_path) -> None:
    fake = FakeSmard([W1, W2, W3, W4])
    with _client(fake) as client:
        # 2025-10-08 falls inside the week starting W2
        stats = smard.download_series(
            client, tmp_path, 4169, "DE-LU", since=date(2025, 10, 8), delay=0
        )
    assert stats.fetched == 3  # W2, W3, W4
    names = {p.name for p in (tmp_path / "4169" / "DE-LU" / "quarterhour").glob("*.json")}
    assert f"4169_DE-LU_quarterhour_{W1}.json" not in names


def test_failed_week_is_counted_not_fatal(tmp_path) -> None:
    fake = FakeSmard([W1, W2, W3])
    fake.fail_week = W2
    with _client(fake) as client:
        stats = smard.download_series(client, tmp_path, 4169, "DE-LU", delay=0)
    assert (stats.fetched, stats.failed) == (2, 1)
    assert "404" in stats.errors[0] or str(W2) in stats.errors[0]


def test_invalid_payload_is_rejected(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "index_" in str(request.url):
            return httpx.Response(200, json={"timestamps": [W1]})
        return httpx.Response(200, content=b'{"unexpected": true}')

    with smard.make_client(httpx.MockTransport(handler)) as client:
        stats = smard.download_series(client, tmp_path, 4169, "DE-LU", delay=0)
    assert stats.fetched == 0 and stats.failed == 1
    assert not list(tmp_path.rglob("*.json"))


def test_registry_has_verified_price_dataset() -> None:
    price = smard.DATASETS["day_ahead_price"]
    assert price.filter_id == 4169 and price.verified
    assert "DE-LU" in price.regions


@pytest.mark.parametrize("res", ["quarterhour", "hour", "day"])
def test_resolutions_known(res: str) -> None:
    assert res in smard.RESOLUTIONS


def test_partial_week_snapshot_is_refetched_until_final(tmp_path) -> None:
    import os

    fake = FakeSmard([W1, W2, W3, W4])
    with _client(fake) as client:
        smard.download_series(client, tmp_path, 4169, "DE-LU", delay=0)
        out = tmp_path / "4169" / "DE-LU" / "quarterhour"
        # W1 was downloaded while its week was still running (a partial snapshot)
        stale = out / f"4169_DE-LU_quarterhour_{W1}.json"
        mid_week = (W1 + 3 * 86_400_000) / 1000
        os.utime(stale, (mid_week, mid_week))
        assert not smard.is_final(stale, W1)
        fake.calls.clear()
        stats = smard.download_series(client, tmp_path, 4169, "DE-LU", refresh_weeks=1, delay=0)
    # W1 (stale) and W4 (newest) re-fetched; W2, W3 final and skipped
    assert (stats.fetched, stats.skipped) == (2, 2)
    assert any(c.endswith(f"_{W1}.json") for c in fake.calls)
    assert smard.is_final(stale, W1)  # fresh download now counts as final


def test_recent_week_with_gaps_is_rechecked(tmp_path) -> None:
    fake = FakeSmard([W1, W2, W3, W4])
    fake.gaps = True  # every payload contains one null
    with _client(fake) as client:
        smard.download_series(client, tmp_path, 4169, "DE-LU", delay=0)
        out = tmp_path / "4169" / "DE-LU" / "quarterhour"
        # make W1 and W2 gap-free: they must then be skipped
        for ts in (W1, W2):
            p = out / f"4169_DE-LU_quarterhour_{ts}.json"
            p.write_text(json.dumps({"meta_data": {}, "series": [[ts, 1.0], [ts + 900_000, 2.0]]}))
        fake.calls.clear()
        stats = smard.download_series(client, tmp_path, 4169, "DE-LU", refresh_weeks=1, delay=0)
    # W3 has a gap (re-checked), W4 is the newest (refreshed); W1, W2 complete and skipped
    assert (stats.fetched, stats.skipped) == (2, 2)
    assert smard.has_gaps(out / f"4169_DE-LU_quarterhour_{W3}.json")
    assert not smard.has_gaps(out / f"4169_DE-LU_quarterhour_{W1}.json")


def test_old_gaps_are_left_alone(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(smard, "RECHECK_WEEKS", 1)
    fake = FakeSmard([W1, W2, W3, W4])
    fake.gaps = True
    with _client(fake) as client:
        smard.download_series(client, tmp_path, 4169, "DE-LU", delay=0)
        stats = smard.download_series(client, tmp_path, 4169, "DE-LU", refresh_weeks=0, delay=0)
    # only W4 is recent enough to be re-checked
    assert (stats.fetched, stats.skipped) == (1, 3)
