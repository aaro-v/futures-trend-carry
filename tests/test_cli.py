import json

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from futures_backtester.cli import app


def test_cli_file_and_generated_panels_give_identical_results(tmp_path) -> None:
    runner = CliRunner()
    data = tmp_path / "data"
    result = runner.invoke(
        app,
        ["sample-data", "--out", str(data), "--days", "780", "--contracts", "12", "--seed", "11"],
    )
    assert result.exit_code == 0, result.exception
    output = tmp_path / "reports"
    result = runner.invoke(
        app,
        ["run", "--data-dir", str(data), "--output-dir", str(output), "--bootstrap-samples", "10"],
    )
    assert result.exit_code == 0, result.exception
    config = json.loads((output / "run_config.json").read_text())
    assert config["data_source"] == "file_panel"
    assert len(config["input_sha256"]) == 4
    assert len(list(output.iterdir())) == 11
    assert (output / "equity_curve.png").stat().st_size > 1000
    expected = pd.read_csv(output / "summary.csv")
    result = runner.invoke(
        app,
        [
            "run",
            "--output-dir",
            str(output),
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
    actual = pd.read_csv(output / "summary.csv")
    np.testing.assert_allclose(actual.select_dtypes("number"), expected.select_dtypes("number"))
    assert json.loads((output / "run_config.json").read_text())["data_source"] == "synthetic"
