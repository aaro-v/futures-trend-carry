import pandas as pd
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
    assert (out_dir / "rolling_metrics.csv").exists()
    assert (out_dir / "daily_returns.csv").exists()
    assert (out_dir / "weights.csv").exists()
    assert (out_dir / "parameter_grid.csv").exists()
    assert (out_dir / "parameter_grid_summary.csv").exists()
    assert (out_dir / "sector_exposure.csv").exists()
    assert (out_dir / "concentration.csv").exists()
    assert (out_dir / "drawdowns.csv").exists()
    assert (out_dir / "execution_costs.csv").exists()
    assert (out_dir / "execution_costs_by_contract.csv").exists()
    assert (out_dir / "walk_forward_summary.csv").exists()
    assert (out_dir / "bootstrap_reliability.csv").exists()
    assert (out_dir / "output_manifest.csv").exists()
    assert (out_dir / "research_report.md").exists()
    manifest = pd.read_csv(out_dir / "output_manifest.csv")
    assert {
        "artifact",
        "bytes",
        "sha256",
        "rows",
        "columns",
    } == set(manifest.columns)
    assert {
        "summary.csv",
        "rolling_metrics.csv",
        "daily_returns.csv",
        "weights.csv",
        "walk_forward_selections.csv",
        "parameter_grid.csv",
        "parameter_grid_summary.csv",
        "execution_costs_by_contract.csv",
        "equity_curve.png",
        "bootstrap_reliability.csv",
        "research_report.md",
    } <= set(manifest["artifact"])
    assert manifest["artifact"].is_unique
    assert manifest["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    summary_row = manifest.set_index("artifact").loc["summary.csv"]
    assert summary_row["rows"] == 1
    assert summary_row["columns"] >= 8
    rolling_row = manifest.set_index("artifact").loc["rolling_metrics.csv"]
    assert rolling_row["rows"] >= 1
    assert rolling_row["columns"] == 7
    image_row = manifest.set_index("artifact").loc["equity_curve.png"]
    assert pd.isna(image_row["rows"])
    selections = (out_dir / "walk_forward_selections.csv").read_text(encoding="utf-8")
    assert "test_sharpe" in selections
    assert "test_max_drawdown" in selections
    parameter_grid = pd.read_csv(out_dir / "parameter_grid.csv")
    parameter_grid_summary = pd.read_csv(out_dir / "parameter_grid_summary.csv")
    assert "is_selected" in parameter_grid
    assert parameter_grid["is_selected"].sum() >= 1
    assert {
        "selection_rate",
        "avg_test_sharpe",
        "positive_test_sharpe_rate",
    } <= set(parameter_grid_summary.columns)
    report = (out_dir / "research_report.md").read_text(encoding="utf-8")
    assert "Sortino" in report
    assert "Calmar" in report
    assert "rolling Sharpe" in report
    assert "Walk-forward validation" in report
    assert "Parameter switch rate" in report
    assert "Train-to-test Sharpe degradation" in report
    assert "Parameter grid diagnostics" in report
    assert "Best average test Sharpe candidate" in report
    assert "Drawdown diagnostics" in report
    assert "Bootstrap reliability" in report
    assert "Positive Sharpe probability" in report
    assert "Contract cost diagnostics" in report
    assert "Highest annualized contract cost" in report
