import importlib.util
import sys
import types
from pathlib import Path

import pytest


_REPO_DIR = Path(__file__).resolve().parent.parent
try:
    import pandas  # noqa: F401
    _HAS_PANDAS = not str(Path(getattr(pandas, "__file__", ""))).startswith(
        str(_REPO_DIR)
    )
except Exception:
    sys.modules["pandas"] = types.ModuleType("pandas")
    _HAS_PANDAS = False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip all tests when pandas is unavailable."""  # pragma: no cover - setup
    if not _HAS_PANDAS:
        skip = pytest.mark.skip(reason="pandas not installed")
        for item in items:
            item.add_marker(skip)


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool:  # pragma: no cover - setup
    if not _HAS_PANDAS:
        return True
    return False

# Provide stub numpy module if the real one is missing
if "numpy" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "numpy", Path(__file__).resolve().parent.parent / "numpy_stub.py"
    )
    numpy_stub = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(numpy_stub)
    sys.modules["numpy"] = numpy_stub


def pytest_runtest_setup(item: pytest.Item) -> None:  # pragma: no cover - setup
    """Skip tests individually when pandas is unavailable."""
    if not _HAS_PANDAS:
        pytest.skip("pandas not installed")
