from __future__ import annotations

import datetime as dt
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from dao.lib.da_prices import (
    DaPrices,
    parse_easyenergy_prices,
    parse_entsoe_prices,
    parse_nordpool_prices,
    parse_tibber_prices,
)

AMSTERDAM = ZoneInfo("Europe/Amsterdam")


def test_parse_entsoe_converts_eur_per_mwh_and_keeps_dst_offsets():
    times = [
        dt.datetime(2026, 10, 25, 2, 0, tzinfo=dt.timezone(dt.timedelta(hours=2))),
        dt.datetime(2026, 10, 25, 2, 0, tzinfo=dt.timezone(dt.timedelta(hours=1))),
    ]
    data = pd.Series([123.0, -25.0], index=pd.Index(times, dtype=object))

    result = parse_entsoe_prices(data)

    assert result == [
        (int(times[0].timestamp()), 0.123),
        (int(times[1].timestamp()), -0.025),
    ]


def test_parse_entsoe_empty_response():
    assert parse_entsoe_prices(pd.Series(dtype=float)) == []


def test_parse_easyenergy_naive_local_times_cover_spring_dst_transition():
    data = [
        {"Timestamp": "2026-03-29T01:00:00+01:00", "TariffReturn": 0.21},
        {"Timestamp": "2026-03-29T03:00:00+02:00", "TariffReturn": 0.31},
    ]

    result = parse_easyenergy_prices(data, ZoneInfo("Europe/Amsterdam"))

    expected = [
        int(dt.datetime.fromisoformat(data[0]["Timestamp"]).timestamp()),
        int(dt.datetime.fromisoformat(data[1]["Timestamp"]).timestamp()),
    ]
    assert result == [
        (expected[0], 0.21),
        (expected[1], 0.31),
    ]


def test_parse_easyenergy_preserves_current_invalid_timestamp_error():
    with pytest.raises(ValueError):
        parse_easyenergy_prices([{"Timestamp": "not-a-date", "TariffReturn": 1}], ZoneInfo("Europe/Amsterdam"))

    assert parse_easyenergy_prices([], ZoneInfo("Europe/Amsterdam")) == []


def test_parse_tibber_reads_today_tomorrow_and_range_nodes():
    node = {"startsAt": "2026-10-25T02:00:00.000+0200", "energy": "0.2"}
    node_second_fold = {"startsAt": "2026-10-25T02:00:00.000+0100", "energy": "-0.1"}
    payload = {
        "data": {
            "viewer": {
                "homes": [{
                    "currentSubscription": {
                        "priceInfo": {"today": [node], "tomorrow": []},
                        "priceInfoRange": {"nodes": [node_second_fold]},
                    }
                }]
            }
        }
    }

    result = parse_tibber_prices(payload)

    assert result == [
        (int(dt.datetime.fromisoformat("2026-10-25T02:00:00+02:00").timestamp()), 0.2),
        (int(dt.datetime.fromisoformat("2026-10-25T02:00:00+01:00").timestamp()), -0.1),
    ]


def test_parse_nordpool_converts_units_skips_infinity_and_handles_dst():
    payload = {
        "areas": {"NL": {"values": [
            {"start": dt.datetime(2026, 10, 25, 2, tzinfo=dt.timezone(dt.timedelta(hours=2))), "value": 100.0},
            {"start": dt.datetime(2026, 10, 25, 2, tzinfo=dt.timezone(dt.timedelta(hours=1))), "value": 200.0},
            {"start": dt.datetime(2026, 10, 25, 3, tzinfo=AMSTERDAM), "value": float("inf")},
        ]}}
    }

    result = parse_nordpool_prices(payload, "NL", ZoneInfo("Europe/Amsterdam"))

    assert result == [
        (int(payload["areas"]["NL"]["values"][0]["start"].timestamp()), 0.1),
        (int(payload["areas"]["NL"]["values"][1]["start"].timestamp()), 0.2),
    ]


class FakeDB:
    def __init__(self):
        self.deleted = []
        self.saved = []

    def get_time_border_record(self, code):
        assert code == "da"

    def deleterange(self, *args):
        self.deleted.append(args)

    def savedata(self, dataframe):
        self.saved.append(dataframe.copy())


def test_get_prices_fetches_range_and_saves_provider_results():
    database = FakeDB()
    config = SimpleNamespace(interval="1hour")
    prices = DaPrices(config, database)
    start = dt.datetime(2026, 1, 1, tzinfo=AMSTERDAM)
    end = dt.datetime(2026, 1, 1, 3, tzinfo=AMSTERDAM)
    provider_prices = [
        (int(start.timestamp()), 0.1),
        (int((start + dt.timedelta(hours=1)).timestamp()), -0.2),
    ]
    calls = []

    def fetch(actual_start, actual_end):
        calls.append((actual_start, actual_end))
        return provider_prices

    prices._get_prices_entsoe = fetch

    prices.get_prices("ENTSOE", _start=start, _end=end)

    assert calls == [(start, end)]
    assert database.deleted == [
        ('da', dt.datetime(2025, 12, 31, 23, 0, tzinfo=dt.UTC), dt.datetime(2026, 1, 1, 0, 0, tzinfo=dt.UTC))]
    assert len(database.saved) == 1
    assert database.saved[0].to_dict("records") == [
        {"time": provider_prices[0][0], "code": "da", "value": 0.1},
        {"time": provider_prices[1][0], "code": "da", "value": -0.2},
    ]


def test_get_prices_does_not_write_when_provider_returns_no_prices():
    database = FakeDB()
    prices = DaPrices(SimpleNamespace(interval="1hour"), database)
    prices._get_prices_entsoe = lambda start, end: []

    prices.get_prices(
        "entsoe",
        _start=dt.datetime(2026, 1, 1, tzinfo=AMSTERDAM),
        _end=dt.datetime(2026, 1, 1, 1, tzinfo=AMSTERDAM),
    )

    assert database.deleted == []
    assert database.saved == []
