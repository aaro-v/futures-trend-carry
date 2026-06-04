import futures_backtester


def test_package_exports() -> None:
    assert hasattr(futures_backtester, "run_backtest")
