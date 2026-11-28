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
    assert set(tr) == {"TG +1", "Bragg", "TG -1", "bkg: Background"} and len(tr["Bragg"][0]) == 4
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


def test_fit_page_reads_saved_file_and_compare(app, tmp_path):
    from xtg_mid_gui.analysis import normalise_trace
    w = XTGMidWindow(sim_kw=dict(n_shots=12))
    w.load(w.backend.scans()[:20])
    p = tmp_path / "a.h5"
    w.export_h5(p)
    fp = w.fit_page
    assert fp.add_file(p) and fp.from_file()
    assert [fp.cmb_det.itemText(i) for i in range(fp.cmb_det.count())] == ["Jung_diffraction", "Jung_transmission"]
    fp.cmb_det.setCurrentText("Jung_transmission")
    fp.cmb_roi.setCurrentText("Bragg 2")
    t, y, e = fp.data(in_range=False)
    s = w.series["Bragg 2"]["norm"]
    assert len(t) == len(s[0]) and np.allclose(t, np.asarray(s[0]) / 1000)
    assert fp.run_fit(True).success
    w.load([101, 102])                                     # the file stays the fit source
    assert fp.from_file() and len(fp.data(in_range=False)[0]) == len(s[0])

    cp = fp.compare
    assert cp.add_current(all_rois=True) == 4              # session, Jung_diffraction (3 ROIs + background)
    cp.add_file(p)
    cp.cmb_det.setCurrentText("Jung_diffraction")
    cp.cmb_roi.setCurrentText("Bragg")
    assert cp.add_current() == 1 and len(cp.traces) == 5
    cp.cmb_mode.setCurrentIndex(1)
    for _, t, y, e, _ in cp.curves():
        assert np.nanmin(y) == pytest.approx(0) and np.nanmax(y) == pytest.approx(1)
    cp.cmb_mode.setCurrentIndex(2)
    cp.sp_t1.setValue(0)
    cp.sp_t2.setValue(400)
    for _, t, y, e, _ in cp.curves():
        m = (t >= 0) & (t <= 400)
        assert np.nanmax(y[m]) == pytest.approx(1)
    cp.cmb_mode.setCurrentIndex(0)
    raw = cp.curves()[-1][2]
    assert np.allclose(raw, cp.traces[-1]["norm"][1])
    cp.lst.item(0).setSelected(True)
    cp.remove_selected()
    assert len(cp.traces) == 4
    y, e, k = normalise_trace([0, 1], [1, 2], [0.1, 0.1], 2, (5, 6))
    assert np.isnan(y).all()
    w.close()


def test_hidden_trace_hides_its_error_bars():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pg = pytest.importorskip("pyqtgraph")
    from pyqtgraph.Qt import QtWidgets
    from xtg_mid_gui.gui import XTGMidWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
    w = XTGMidWindow(sim_kw=dict(n_shots=10))
    w.load([101, 102, 103])
    cp = w.fit_page.compare
    assert cp.add_current() == 1
    for plot in (cp.plot, w.fit_page.p_fit):
        curves = [c for c in plot.listDataItems() if getattr(c, "_eb", None) is not None]
        assert curves
        curves[0].setVisible(False)                                   # = click on its legend entry
        assert not curves[0]._eb.isVisible()
        curves[0].setVisible(True)
        assert curves[0]._eb.isVisible()
    w.close()


def test_hiding_a_trace_rescales_fit_and_compare():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("pyqtgraph")
    from pyqtgraph.Qt import QtWidgets
    from xtg_mid_gui.gui import XTGMidWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
    w = XTGMidWindow(sim_kw=dict(n_shots=10))
    w.load([101, 102, 103])
    cp = w.fit_page.compare
    cp.cmb_series.setCurrentText("raw")
    assert cp.add_current(all_rois=True) >= 2
    curves = [c for c in cp.plot.listDataItems() if getattr(c, "_eb", None) is not None]
    big = max(curves, key=lambda c: np.nanmax(c._ylims[1]))
    big.setVisible(False)
    top = cp.plot.getViewBox().viewRange()[1][1]
    rest = max(np.nanmax(c._ylims[1]) for c in curves if c is not big)
    assert top < np.nanmax(big._ylims[1]) and top == pytest.approx(rest, rel=0.2)
    fit_items = [c for c in w.fit_page.p_fit.listDataItems() if c.name() == "data"]
    assert fit_items and getattr(fit_items[0], "_rescales", False)
    w.close()
