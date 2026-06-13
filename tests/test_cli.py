from typer.testing import CliRunner

from futures_backtester.cli import app


def test_cli_generates_outputs(tmp_path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "reports"

    result = runner.invoke(
        app,
        [
            "run",
            "--output-dir",
            str(out_dir),
            "--days",
            "620",
            "--contracts",
            "8",
            "--bootstrap-samples",
            "25",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "daily_returns.csv").exists()
    assert (out_dir / "weights.csv").exists()
    assert (out_dir / "sector_exposure.csv").exists()
    assert (out_dir / "concentration.csv").exists()
    assert (out_dir / "drawdowns.csv").exists()
    assert (out_dir / "execution_costs.csv").exists()
    assert (out_dir / "walk_forward_summary.csv").exists()
    assert (out_dir / "research_report.md").exists()
    selections = (out_dir / "walk_forward_selections.csv").read_text(encoding="utf-8")
    assert "test_sharpe" in selections
    assert "test_max_drawdown" in selections
    report = (out_dir / "research_report.md").read_text(encoding="utf-8")
    assert "Sortino" in report
    assert "Calmar" in report
    assert "Walk-forward validation" in report
    assert "Parameter switch rate" in report
    assert "Train-to-test Sharpe degradation" in report
    assert "Drawdown diagnostics" in report
