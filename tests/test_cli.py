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
    assert (out_dir / "research_report.md").exists()
