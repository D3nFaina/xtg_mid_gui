import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile  # noqa: E402

# tests never read or write delays_<proposal>.csv / Sources_*.yaml in the user's folder
os.environ["XTG_MID_START_DIR"] = tempfile.mkdtemp(prefix="xtg_mid_tests_")
