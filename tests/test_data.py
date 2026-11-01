from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from pandas.testing import assert_frame_equal

from futures_backtester.data import (
    FuturesPanel,
    load_panel_frames,
    make_sample_panel,
    panel_to_frame,
    write_sample_data,
)


def test_files_schema_and_shared_loader(tmp_path) -> None:
    write_sample_data(tmp_path, days=300, contracts=44)
    assert {p.name for p in tmp_path.iterdir()} == {
        "panel.parquet",
        "contracts_resolved.csv",
        "roll_schedule.csv",
        "rates_daily.parquet",
    }
    frame = pd.read_parquet(tmp_path / "panel.parquet")
    assert len(frame.columns) == 16
    assert frame.date.dtype == "datetime64[us]"
    assert str(frame.contract.dtype) == "category"
    assert frame.volume.dtype == "float64"
    assert frame.is_roll_day.dtype == "bool"
    panel = FuturesPanel(**load_panel_frames(tmp_path))
    assert panel.prices.shape == (300, 44)
    assert panel.commission_bps.shape == (300, 44)
    assert len(panel_to_frame(panel)) == len(frame)
    missing = panel.sectors.index[panel.sectors.isin(["metals", "grains", "softs", "livestock"])]
    assert panel.carry[missing].isna().all().all()
    assert panel.carry["BZ"].isna().all()
    assert_frame_equal(panel.prices, make_sample_panel(days=300, contracts=44).prices)
    assert_frame_equal(frame, pd.read_parquet(tmp_path / "panel.parquet"))


@pytest.mark.skipif(not Path("data/real/panel.parquet").exists(), reason="Private panel absent")
def test_real_and_synthetic_parquet_schemas_match(tmp_path) -> None:
    write_sample_data(tmp_path, days=80)
    actual = pq.read_schema(tmp_path / "panel.parquet").remove_metadata()
    expected = pq.read_schema("data/real/panel.parquet").remove_metadata()
    assert actual.equals(expected)


def test_generator_matches_sector_volatility_and_correlation(tmp_path) -> None:
    write_sample_data(tmp_path)
    panel = FuturesPanel(**load_panel_frames(tmp_path))
    returns = panel.prices.pct_change(fill_method=None)
    vols = returns.std() * np.sqrt(252)
    assert 0.04 < vols[panel.sectors == "rates"].median() < 0.08
    assert 0.17 < vols[panel.sectors == "equity_index"].median() < 0.26
    assert 0.32 < vols[panel.sectors == "energy"].median() < 0.47
    for sector in ("rates", "equity_index"):
        corr = returns.loc[:, panel.sectors == sector].corr().to_numpy()
        assert 0.70 < corr[np.triu_indices_from(corr, k=1)].mean() < 0.90
    assert panel.carry.loc["2024-04-08":, panel.sectors == "energy"].isna().all().all()


def test_loader_preserves_late_listing_and_limits_stale_prices(tmp_path) -> None:
    write_sample_data(tmp_path, days=100, contracts=2)
    long = pd.read_parquet(tmp_path / "panel.parquet")
    last_name = long.contract.cat.categories[-1]
    dates = sorted(long.date.unique())
    long = long.loc[~((long.contract == last_name) & (long.date < dates[20]))]
    long = long.loc[~((long.contract == last_name) & long.date.isin(dates[50:60]))]
    long.to_parquet(tmp_path / "panel.parquet", index=False)
    prices = load_panel_frames(tmp_path)["prices"]
    assert prices[last_name].iloc[:20].isna().all()
    assert prices[last_name].iloc[50:55].notna().all()
    assert prices[last_name].iloc[55:60].isna().all()
