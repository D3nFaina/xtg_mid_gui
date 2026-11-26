"""Offscreen GUI smoke test: load, ROIs, theme switch, CRL toggle, export."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pg = pytest.importorskip("pyqtgraph")
from pyqtgraph.Qt import QtCore, QtWidgets  # noqa: E402

from xtg_mid_gui import theme  # noqa: E402
from xtg_mid_gui.gui import XTGMidWindow  # noqa: E402


@pytest.fixture(scope="module")
def win():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.apply(app, "dark")
    w = XTGMidWindow(sim_kw=dict(n_shots=20))
    w.load([101, 102, 103, 104])
    yield w
    w.close()


def test_default_rois(win):
    for det in ("jf1", "jf2"):
        panel = win.views[det].panel
        assert len(panel.rois) == 4
        assert sum(r.background for r in panel.rois) == 1
    assert len(win.series) == 6                     # 3 signal ROIs x 2 detectors
    assert all(len(s["norm"][0]) == 4 for s in win.series.values())


def test_add_rename_move_delete_roi(win):
    panel = win.views["jf1"].panel
    spec = panel.spawn()
    assert spec.name == "ROI 5" and len(panel.rois) == 5
    panel.rename(spec, "my peak")
    spec.item.setPos((5, 5))
    win.recompute()
    label = f"{win.cfg.detectors['jf1'].label} · my peak"
    assert label in win.series
    panel.remove(spec)
    win.recompute()
    assert label not in win.series and len(panel.rois) == 4


def test_add_scans_and_typed_delay(win):
    win.load([105], append=True)
    assert len(win.scans) == 5
    win.scans[105].delay = win.scans[101].delay     # same delay -> merged point
    win.recompute()
    s = next(iter(win.series.values()))
    assert len(s["norm"][0]) == 4 and s["norm"][3].sum() > 0


def test_theme_and_crl_toggle(win):
    win.toggle_theme()
    assert theme.mode() == "light"
    win.toggle_theme()
    assert theme.mode() == "dark"
    win.set_crl_visible(False)
    assert not win.tabs.isTabVisible(win.crl_tab)
    win.set_crl_visible(True)
    assert win.tabs.isTabVisible(win.crl_tab)


def test_export(win, tmp_path):
    p = tmp_path / "out.csv"
    win.export_csv(p)
    rows = [r for r in p.read_text().splitlines() if not r.startswith("#")]
    assert rows[0].startswith("roi,delay_fs") and len(rows) > 10
    assert np.isfinite(float(rows[1].split(",")[2]))


def test_spectrometer_page(win):
    win.set_spec_visible(False)
    assert not win.tabs.isTabVisible(win.spec_tab)
    win.set_spec_visible(True)
    assert win.tabs.isTabVisible(win.spec_tab)
    full = win.spec_integrals("spec1")[0]
    win.spec_regions["spec1"].setRegion((190, 212))
    win._spec_region_changed("spec1")
    assert win.spec_windows["spec1"] == (190, 213)
    seed = win.spec_integrals("spec1")[0]
    assert seed.shape == full.shape and np.all(seed < full)


def test_roi_shape_toggle_is_per_roi(win):
    panel = win.views["jf1"].panel
    a, b = panel.rois[0], panel.rois[1]
    g0 = a.geometry()
    panel.set_shape(a, "rectangle")
    assert a.shape == "rectangle" and b.shape == "ellipse"
    assert a.geometry()[:5] == pytest.approx(g0[:5], abs=0.05)   # same centre/size/angle
    assert a.item in panel.view.addedItems
    win.recompute()
    assert panel.to_json().count('"rectangle"') == 1
    panel.set_shape(a, "ellipse")
    assert a.shape == "ellipse"


def test_colormaps(win):
    for name in theme.COLORMAPS:
        win.set_cmap(name)
    win.set_cmap(theme.DEFAULT_CMAP["dark"])
    win.toggle_theme()                                  # default follows the theme
    assert win.cmb_cmap.currentText() == theme.DEFAULT_CMAP["light"]
    win.set_cmap("magma")
    win.toggle_theme()                                  # an explicit choice is kept
    assert win.cmb_cmap.currentText() == "magma"


def test_full_page_keeps_edits(win):
    v1, v2 = win.views["jf1"], win.views["jf2"]
    r2 = v2.plot.getViewBox().viewRange()
    win.toggle_full("jf1")
    assert win.full_page == "jf1" and v2.isHidden() and win.sidebar.isHidden()
    spec = v1.panel.spawn("added in full page")
    win.toggle_full("jf1")
    assert win.full_page is None and not v2.isHidden() and not win.sidebar.isHidden()
    assert spec in v1.panel.rois
    assert np.ravel(v2.plot.getViewBox().viewRange()) == pytest.approx(np.ravel(r2), abs=2)
    v1.panel.remove(spec)


def test_diode_trace_toggle(win):
    win.cmb_diode_view.setCurrentText("vs shot number")
    n = sum(win.scans[s].n for s in win.scans if win.use[s])
    xs = [it.getData()[0] for it in win.c_hpu.listDataItems()]
    assert sum(len(x) for x in xs) == n
    win.cmb_diode_view.setCurrentText("distribution")


def test_log_scale_is_display_only(win):
    view = win.views["jf1"]
    before = dict(next(iter(win.series.values())))["raw"][1].copy()
    lin = view.img.image.copy()
    win.set_log(True)
    assert view.log_scale and np.nanmax(view.img.image) == pytest.approx(np.log10(np.nanmax(lin)), rel=1e-4)
    win.recompute()
    assert np.allclose(next(iter(win.series.values()))["raw"][1], before)    # ROI sums unchanged
    win.set_log(False)
    assert np.allclose(view.img.image, lin)


def test_rotated_roi_keeps_centre_on_shape_switch_and_json(win):
    from xtg_mid_gui.rois import RoiSpec
    panel = win.views["jf1"].panel
    spec = panel.add(RoiSpec("rot", (60, 40), (20, 8), (255, 0, 0), angle=30))
    assert spec.geometry()[:5] == (60, 40, 20, 8, 30)
    panel.set_shape(spec, "rectangle")
    assert spec.geometry()[:2] == (60, 40)
    again = RoiSpec.from_dict(spec.to_dict())
    assert again.geometry()[:5] == (60, 40, 20, 8, 30)
    panel.remove(spec)


def test_duplicate_roi_names_keep_both_curves(win):
    panel = win.views["jf1"].panel
    spec = panel.spawn()
    panel.rename(spec, panel.rois[0].name)
    win.recompute()
    labels = [k[0] for k in win.signal_keys()]
    assert len(labels) == len(set(labels)) and len(win.series) == len(labels)
    panel.remove(spec)
    win.recompute()


def test_spectrometer_band_on_falling_ev_axis(win):
    name = next(iter(win.cfg.spectrometers))
    s0 = win.sorted_scans()[0]
    axis = 9000.0 - 0.1 * np.arange(len(win.scans[s0].spec_axis[name]))      # dE < 0
    for s in win.scans.values():
        s.spec_axis[name] = axis
    win.set_spec_visible(True)
    win._draw_spec()
    lo, hi = win.spec_regions[name].getRegion()
    assert np.isclose(lo, axis.min()) and np.isclose(hi, axis.max())
    win.spec_regions[name].setRegion((axis[200], axis[100]))
    win._spec_region_changed(name)
    assert win.spec_windows[name] == (100, 201)


@pytest.fixture(scope="module")
def win7016():
    from xtg_mid_gui.config import config_for
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.apply(app, "dark")
    w = XTGMidWindow(config_for("7016"), sim_kw=dict(n_shots=12))
    w.load([101, 102, 103, 104, 105])
    yield w
    w.close()


def test_7016_pages(win7016):
    w = win7016
    tabs = [w.tabs.tabText(i) for i in range(w.tabs.count()) if w.tabs.isTabVisible(i)]
    assert "Spectrometers && XGM" in tabs and "CRL pointing" not in tabs
    assert w.mosaic is not None and list(w.views) == ["agipd_sub"]
    from xtg_mid_gui import agipd
    assert w.mosaic._raw.shape == agipd.canvas_shape()
    assert len(w.series) == 3 and all(len(s["norm"][0]) == 5 for s in w.series.values())
    assert w.xgm_plots is not None and len(w.spec_plots) == 2
    assert w.cmb_preset.currentData() == "7016"


def test_7016_tile_choice_keeps_rois_on_their_tiles(win7016):
    w = win7016
    panel = w.views["agipd_sub"].panel
    bragg = next(r for r in panel.rois if "Bragg" in r.name)
    y0 = bragg.geometry()[1]
    w.mosaic.toggle_tile(3)                      # click on tile 3 in the all-tiles view
    assert w.mosaic.chosen() == [3, 8, 10, 15]
    w.mosaic.b_apply.click()
    assert w.cfg.detectors["agipd_sub"].tiles == [3, 8, 10, 15]
    assert w.views["agipd_sub"]._raw.shape[0] == 4 * 128 + 3 * 8
    assert bragg.geometry()[1] == pytest.approx(y0 + 136)      # tile 15 moved down one slot
    w.mosaic.toggle_tile(3)
    w.mosaic.b_apply.click()
    assert bragg.geometry()[1] == pytest.approx(y0)


def test_7016_spectrometer_band_sets_pump_intensity(win7016):
    w = win7016
    s = w.sorted_scans()[0]
    full = w.inten(s, "pump").copy()
    w.spec_regions["hirex"].setRegion((1990, 2010))
    w._spec_region_changed("hirex")
    part = w.inten(s, "pump")
    assert w.spec_windows["hirex"] is not None and np.all(part < full)
    w.spec_regions["hirex"].setRegion((1750, 2249))
    w._spec_region_changed("hirex")
    assert np.allclose(w.inten(s, "pump"), full)


def test_switch_proposal_rebuilds_window(win7016, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    new = win7016.switch_preset("11456")
    try:
        assert new.cfg.proposal == 11456 and new.mosaic is None
        assert list(new.cfg.spectrometers) == ["spec1"] and len(new.spec_plots) == 1
        assert set(new.views) == {"jf1", "jf2"} and not new.scans      # nothing loaded automatically
        back = new.switch_preset("7016")
        assert back.mosaic is not None
        back.close()
    finally:
        new.close()


def test_typed_delays_are_saved_and_filled_in(tmp_path, monkeypatch):
    from xtg_mid_gui import delays
    monkeypatch.setenv("XTG_MID_START_DIR", str(tmp_path))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = XTGMidWindow(sim_kw=dict(n_shots=8, delays=[100, 200, 300, 400]))
    w.load([104, 101, 103])                      # any order, with a gap
    path = w.delays_path()
    assert path.parent == tmp_path and not path.exists()
    row = {w.tbl.item(i, 0).data(QtCore.Qt.UserRole): i for i in range(w.tbl.rowCount())}
    w.tbl.item(row[103], 2).setText("-10")      # type a delay in the table
    assert delays.load(path) == {103: -10.0} and w.scans[103].delay == -10.0
    w.cmb_order.setCurrentText("run number")
    assert [w.tbl.item(i, 1).text() for i in range(3)] == ["101", "103", "104"]
    w.cmb_order.setCurrentText("delay")
    assert w.tbl.item(0, 1).text() == "103"     # -10 fs comes first
    w.cmb_order.setCurrentText("as typed")
    assert [w.tbl.item(i, 1).text() for i in range(3)] == ["104", "101", "103"]
    w.load([103, 102])                           # a new session: 103 gets its delay from the file
    assert w.scans[103].delay == -10.0 and w.scans[103].delay_from == "file"
    assert w.scans[102].delay_from == "data"
    w.set_delay(103, "")                          # cleared: back to the data, removed from the file
    assert delays.load(path) == {} and w.scans[103].delay == w.scans[103].delay_data
    w.close()
