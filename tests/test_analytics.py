import numpy as np
import pandas as pd

from futures_backtester.analytics import concentration_summary


def test_concentration_summary_handles_zero_exposure() -> None:
    weights = pd.DataFrame(0.0, index=pd.RangeIndex(3), columns=["ES", "TY"])

    summary = concentration_summary(weights)

    assert np.isfinite(summary.to_numpy()).all()
    assert summary.loc[0, "avg_gross_weight"] == 0.0
    assert summary.loc[0, "effective_contract_count"] == 0.0


def test_concentration_summary_reports_effective_contract_count() -> None:
    weights = pd.DataFrame(
        {
            "ES": [0.5, 0.2],
            "TY": [-0.5, -0.2],
            "CL": [0.0, 0.6],
        }
    )

    summary = concentration_summary(weights)

    assert 2.0 < summary.loc[0, "effective_contract_count"] < 3.0
    assert summary.loc[0, "avg_top_5_gross_share"] == 1.0
