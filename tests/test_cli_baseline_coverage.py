import subprocess
import sys
import pytest
pytest.importorskip("pandas")


def test_cli_baseline_coverage():
    result = subprocess.run(
        [sys.executable, "pipeline.py", "config.yaml", "--baseline-coverage"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Naive baseline coverage:" in result.stdout
