"""Reduced-data .h5 export, auto-save with overwrite question, detector choice and names."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("h5py")
pytest.importorskip("pyqtgraph")
from pyqtgraph.Qt import QtWidgets  # noqa: E402

from xtg_mid_gui import export  # noqa: E402
from xtg_mid_gui.config import SourceConfig  # noqa: E402
from xtg_mid_gui.gui import XTGMidWindow  # noqa: E402


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_detector_names():
    cfg = SourceConfig()
    assert [d.label for d in cfg.detectors.values()] == ["Jung_diffraction", "Jung_transmission"]
    old = SourceConfig.from_dict({"detectors": {"jf1": {"label": "Jungfrau 1", "data": {"source": "X", "key": "k"}}}})
    assert old.detectors["jf1"].label == "Jung_diffraction"


def test_export_file(app, tmp_path):
    import h5py
    w = XTGMidWindow(sim_kw=dict(n_shots=15))
    w.load([101, 102, 103, 104])
    p = w.export_h5(tmp_path / "red.h5") or tmp_path / "red.h5"
    assert export.is_reduced(p) and export.detectors(p) == ["Jung_diffraction", "Jung_transmission"]
    tr = export.traces(p, "Jung_diffraction")
    assert set(tr) == {"TG +1", "Bragg", "TG -1"} and len(tr["Bragg"][0]) == 4
    with h5py.File(p) as f:
        r = f["runs/run_0102"]
        assert r["Jung_diffraction/Bragg"].shape == (3,) and r["pump"].shape == (3,) and "spec/spec1" in r
        assert r["Jung_diffraction/Bragg"][0] == w.scans[102].delay
        assert f["Jung_diffraction/Bragg/per_run"].shape == (4, 4)
        assert np.isfinite(r["probe"][1])
    w.close()


def test_autosave_asks_once_then_overwrites(app, tmp_path):
    w = XTGMidWindow(sim_kw=dict(n_shots=10))
    w.ed_export.setText(str(tmp_path / "auto.h5"))
    asked = []
    w.ask_yes_no = lambda *a: asked.append(a) or True
    (tmp_path / "auto.h5").write_text("old")
    w.load([101, 102])
    w.cb_autosave.setChecked(True)
    w.export_now(ask=False, auto=True)
    assert len(asked) == 1 and export.is_reduced(tmp_path / "auto.h5")
    w.export_now(ask=False, auto=True)                    # same name: no second question
    assert len(asked) == 1
    w.ed_export.setText(str(tmp_path / "other.h5"))
    w.ask_yes_no = lambda *a: asked.append(a) or False
    (tmp_path / "other.h5").write_text("keep me")
    assert w.export_now(ask=False, auto=True) is None and not w.cb_autosave.isChecked()
    assert (tmp_path / "other.h5").read_text() == "keep me"
    w.close()


def test_only_one_detector(app):
    w = XTGMidWindow(sim_kw=dict(n_shots=10))
    w.load([101, 102, 103])
    w.set_detector_used("jf2", False)
    assert w.cfg.use_detectors == ["jf1"] and all(v["det"] == "jf1" for v in w.series.values())
    assert "jf2" not in w.scans[102].stats and "jf2" not in w.scans[101].images
    w.set_detector_used("jf1", False)                     # the last one stays on
    assert w.cfg.use_detectors == ["jf1"]
    w.set_detector_used("jf2", True)
    assert w.cfg.use_detectors == [] and {v["det"] for v in w.series.values()} == {"jf1", "jf2"}
    w.close()
