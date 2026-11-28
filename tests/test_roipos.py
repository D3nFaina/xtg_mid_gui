"""ROI position check (centre of mass / max) and the 11456 upstream spectrometer 2 x 2 page."""
import os

import numpy as np
import pytest

from xtg_mid_gui import analysis as an


def test_roi_position_com_and_max():
    im = np.zeros((3, 40, 60))
    im[0, 10, 20] = 5                          # one bright pixel
    im[1, 10, 20] = im[1, 10, 24] = 1          # two equal pixels: COM between them
    im[2, 30, 50] = 9                          # outside the ROI below
    geom = (22, 12, 10, 8, 0)                  # ellipse centred (22, 12)
    x, y = an.roi_position(im, geom, "com")
    assert np.allclose([x[0], y[0]], [20.5, 10.5]) and np.allclose([x[1], y[1]], [22.5, 10.5])
    assert np.isnan(x[2])                      # nothing in the ROI
    x, y = an.roi_position(im, geom, "max")
    assert np.allclose([x[0], y[0]], [20.5, 10.5])
    flat = np.ones((1, 40, 60))
    flat[0, 12, 25] = 3
    xc, yc = an.roi_position(flat, geom, "com", an.roi_bkg_level(flat, [(50, 30, 5, 5, 0)]))
    assert np.allclose([xc[0], yc[0]], [25.5, 12.5])   # background level removed first


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("pyqtgraph")
    from pyqtgraph.Qt import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_roi_position_window(app):
    from xtg_mid_gui.gui import XTGMidWindow
    w = XTGMidWindow(sim_kw=dict(n_shots=15))
    w.load([101, 102, 103])
    w.b_pos.setChecked(True)
    pw = w.pos_win
    assert pw.isVisible() and pw.cmb_roi.count() == len(w.signal_keys())
    assert pw.cmb_per.currentText() == "mean of each run"            # default: one point per run
    assert sum(len(i.points()) for i in pw.p_x.items if hasattr(i, "points")) == 3
    pw.cmb_per.setCurrentText("shot by shot")
    n_kept = sum(int(w.keep[s].sum()) for s in w.sorted_scans())
    assert sum(len(i.points()) for i in pw.p_x.items if hasattr(i, "points")) == n_kept
    pw.cmb_per.setCurrentText("mean of each run")
    pw.cmb_mode.setCurrentText("max position")
    assert sum(len(i.points()) for i in pw.p_x.items if hasattr(i, "points")) == 3
    det = next(iter(w.views))
    roi = w.views[det].panel.signal_rois()[0]
    x, y = pw.positions(det, 101, roi, "com", "shot", True)
    cx, cy = roi.geometry()[:2]
    assert np.nanmax(np.hypot(x - cx, y - cy)) < 5          # simulated peak sits in the default ROI
    pw.close()
    assert not w.b_pos.isChecked()
    w.close()


def test_upstream_spectrometer_page_is_2x2(app):
    from xtg_mid_gui.config import config_for
    from xtg_mid_gui.gui import XTGMidWindow
    w = XTGMidWindow(config_for("11456"), sim_kw=dict(n_shots=10))
    layout = w.spec_plots[next(iter(w.spec_plots))][0].parentItem().layout
    cells = set()
    for p in w.spec_plots[next(iter(w.spec_plots))]:
        for r in range(layout.rowCount()):
            for c in range(layout.columnCount()):
                if layout.itemAt(r, c) is p:
                    cells.add((r, c))
    assert cells == {(0, 0), (0, 1), (1, 0), (1, 1)}
    w.close()
