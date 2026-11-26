import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile  # noqa: E402

# tests never read or write delays_<proposal>.csv / Sources_*.yaml in the user's folder
os.environ["XTG_MID_START_DIR"] = tempfile.mkdtemp(prefix="xtg_mid_tests_")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _own_start_dir(tmp_path, monkeypatch):
    """Each test its own start folder (delays / masks files do not leak between tests)."""
    monkeypatch.setenv("XTG_MID_START_DIR", str(tmp_path))
