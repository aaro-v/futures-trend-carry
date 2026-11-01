from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FuturesPanel:
    prices: pd.DataFrame
    carry: pd.DataFrame
    roll_cost_bps: pd.DataFrame
    commission_bps: pd.Series | pd.DataFrame
    sectors: pd.Series
