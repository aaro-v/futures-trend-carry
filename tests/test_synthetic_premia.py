from futures_backtester import BacktestSettings, run_backtest
from futures_backtester.data import make_sample_panel


def test_default_synthetic_sleeves_have_moderate_premia() -> None:
    settings = BacktestSettings(bootstrap_samples=5, report_start="2005-01-01")
    result = run_backtest(make_sample_panel(), settings)
    sharpe = result.summary.set_index("strategy").loc[["trend", "carry"], "sharpe"]
    assert sharpe.between(0.3, 0.8).all(), sharpe
