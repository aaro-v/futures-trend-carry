import json

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from futures_backtester.cli import app


def test_cli_file_and_generated_panels_give_identical_results(tmp_path, monkeypatch) -> None:
    runner = CliRunner()
    data = tmp_path / "real"
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        ["sample-data", "--out", str(data), "--days", "780", "--contracts", "12", "--seed", "11"],
    )
    assert result.exit_code == 0, result.exception
    output = tmp_path / "reports" / "real"
    result = runner.invoke(
        app,
        ["run", "--data-dir", str(data), "--bootstrap-samples", "10"],
    )
    assert result.exit_code == 0, result.exception
    config = json.loads((output / "run_config.json").read_text())
    assert config["data_source"] == "file_panel"
    assert len(config["input_sha256"]) == 4
    assert len(list(output.iterdir())) == 11
    assert (output / "equity_curve.png").stat().st_size > 1000
    expected = pd.read_csv(output / "summary.csv")
    original_summary = (output / "summary.csv").read_bytes()
    result = runner.invoke(app, ["robustness", "--report-dir", str(output)])
    assert result.exit_code == 0, result.exception
    assert (output / "summary.csv").read_bytes() == original_summary
    assert (output / "robustness_sectors.csv").is_file()
    checks = pd.read_csv(output / "robustness.csv")
    assert checks.scenario.tolist() == [
        "baseline",
        "next_session_close",
        "trading_costs_2x",
        "trading_costs_3x",
        "fixed_blend_252_0.35",
    ]
    result = runner.invoke(
        app,
        [
            "run",
            "--days",
            "780",
            "--contracts",
            "12",
            "--seed",
            "11",
            "--bootstrap-samples",
            "10",
        ],
    )
    assert result.exit_code == 0, result.exception
    assert json.loads((output / "run_config.json").read_text())["data_source"] == "file_panel"
    output = tmp_path / "reports" / "synthetic"
    actual = pd.read_csv(output / "summary.csv")
    np.testing.assert_allclose(actual.select_dtypes("number"), expected.select_dtypes("number"))
    assert json.loads((output / "run_config.json").read_text())["data_source"] == "synthetic"
