"""Reduction engine (compute.py), per-run cache (store.py) and ROI database (roidb.py)."""
import os
import sys

import numpy as np
import pytest

from xtg_mid_gui import analysis as an
from xtg_mid_gui import compute, roidb, store
from xtg_mid_gui.backends import SimulatedBackend
from xtg_mid_gui.config import SourceConfig

pytest.importorskip("h5py")

G = (40.0, 50.0, 6.0, 6.0, 0.0, "ellipse")
G2 = (60.0, 50.0, 6.0, 4.0, 10.0, "rect")
B = (10.0, 10.0, 5.0, 5.0, 0.0, "ellipse")


def _be(**kw):
    return SimulatedBackend(SourceConfig(), n_shots=15, **kw)


def test_cache_is_reused_and_only_new_rois_are_read():
    be = _be()
    rois = {"jf1": ([G, B], [B])}
    assert compute.ensure_run(be, 101, rois)["did"] == "read"
    assert compute.ensure_run(be, 101, rois)["did"] == "cached"
    assert compute.ensure_run(be, 101, {"jf1": ([G], [B])})["did"] == "cached"   # subset: nothing to do
    info = compute.ensure_run(be, 101, {"jf1": ([G, G2, B], [B])})
    assert info["did"] == "rois"
    d = store.load_light(info["path"])
    full = be.load_scan(101)
    for g in (G, G2):
        assert np.allclose(d.stats["jf1"].signal(g, [B]), an.roi_signal(full.images["jf1"], g, [B]))
        assert d.stats["jf1"].position(g, "com", [B]) is not None
    assert np.allclose(d.pump, full.pump) and np.allclose(d.mean_image("jf2"), full.mean_image("jf2"), atol=1e-5)
    assert d.images == {} and getattr(d.spectra, "lazy", False)
    assert np.allclose(d.spectra["spec1"], full.spectra["spec1"])


def test_settings_change_reads_again():
    be = _be()
    compute.ensure_run(be, 102, {"jf1": ([G], [])})
    be.cfg.apply_mask = not be.cfg.apply_mask
    assert compute.ensure_run(be, 102, {"jf1": ([G], [])})["did"] == "read"
    assert compute.ensure_run(be, 102, {"jf1": ([G], [])}, force=True)["did"] == "read"
    other = _be(seed=8)                                    # other simulated data, other fingerprint
    assert compute.ensure_run(other, 102, {"jf1": ([G], [])})["did"] == "read"


def test_new_background_roi_adds_positions():
    be = _be()
    compute.ensure_run(be, 103, {"jf1": ([G], [])})
    info = compute.ensure_run(be, 103, {"jf1": ([G, B], [B])})
    assert info["did"] == "rois"
    st = store.load_light(info["path"]).stats["jf1"]
    assert st.position(G, "max", [B]) is not None and st.signal(G, [B]) is not None


def test_ensure_many_in_parallel():
    sys.path.insert(0, os.path.dirname(__file__))
    from _par_backend import ParBackend
    be = ParBackend(SourceConfig())
    out = dict(compute.ensure_many(be, [101, 102, 999], {"jf1": ([G], [])}, workers=3))
    assert isinstance(out[999], RuntimeError)
    assert out[101]["did"] == "read" and os.path.isfile(out[101]["path"])
    again = dict(compute.ensure_many(be, [101, 102], {"jf1": ([G], [])}, workers=2))
    assert {v["did"] for v in again.values()} == {"cached"}


def test_command_line_uses_roi_database(capsys):
    cfg = SourceConfig()
    roi = dict(name="peak", shape="ellipse", center=[40.0, 50.0], radii=[6.0, 6.0], angle=0.0,
               background=False, enabled=True, color=[1, 2, 3])
    roidb.save(cfg.proposal, {"jf1": [roi]}, runs=[105])
    assert roidb.for_run(cfg.proposal, 105)["jf1"][0]["name"] == "peak"
    assert roidb.spec_from_dicts(roidb.last(cfg.proposal)) == {"jf1": ([G], [])}
    assert compute.main(["--backend", "simulated", "--preset", "custom", "--runs", "105-106", "--roi-run", "105"]) == 0
    assert "read" in capsys.readouterr().out
    assert compute.main(["--backend", "simulated", "--preset", "custom", "--runs", "105-106"]) == 0
    assert capsys.readouterr().out.count("cached") == 2


def test_gui_uses_cache_and_remembers_rois():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("pyqtgraph")
    from pyqtgraph.Qt import QtWidgets
    from xtg_mid_gui.gui import XTGMidWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
    w = XTGMidWindow(sim_kw=dict(n_shots=12))
    w.load([101, 102, 103])
    assert "2 read" in w.log_box.toPlainText()
    ref_series = {k: v["raw"][1].copy() for k, v in w.series.items()}
    w.load([101, 102, 103])
    assert "2 from the cache" in w.log_box.toPlainText()
    for k, v in w.series.items():
        assert np.allclose(v["raw"][1], ref_series[k])
    assert w.scans[102].images == {} and w.scans[101].images           # only the reference keeps images
    w.set_reference(102)                                               # 101 leaves: from its own images
    assert w.scans[101].images == {} and w.scans[102].images
    for k, v in w.series.items():
        assert np.allclose(v["raw"][1], ref_series[k])
    roi = w.views["jf1"].panel.rois[0]
    roi.item.setPos(roi.item.pos() + pg_point(3, 2))
    w.recompute()
    assert w._stale
    w.recalc_stale()
    assert "new ROIs only" in w.log_box.toPlainText()
    moved = roi.geometry()
    w.close()                                                          # saves the ROIs
    w2 = XTGMidWindow(sim_kw=dict(n_shots=12))
    assert w2.views["jf1"].panel.rois[0].geometry() == moved          # last ROIs come back
    assert w2.cmb_roirun.count() == 3
    w2.views["jf1"].panel.rois[0].item.setPos(pg_point(1, 1))
    w2.use_rois_of_run(102)                                            # 102: last analysed in the first session
    assert w2.views["jf1"].panel.rois[0].geometry() == moved
    w2.close()


def pg_point(x, y):
    from pyqtgraph.Qt import QtCore
    return QtCore.QPointF(x, y)


def test_mean_image_follows_a_file_rewritten_by_a_worker():
    """Real data: worker processes rewrite the cache files; the GUI process must not keep
    showing the mean images it had read before (new AGIPD tiles showed the old ones)."""
    sys.path.insert(0, os.path.dirname(__file__))
    from _par_backend import ParBackend
    from xtg_mid_gui.config import config_for
    cfg = config_for("7016")
    det = next(n for n, d in cfg.detectors.items() if d.kind == "agipd_tiles")
    be = ParBackend(cfg)
    for tiles in ([8, 10, 15], [9, 8, 15]):
        cfg.detectors[det].tiles = tiles
        out = dict(compute.ensure_many(be, [101, 102], {}, workers=2))
        for s in (101, 102):
            got = store.load_light(out[s]["path"]).mean_image(det)
            ref = be.load_scan(s, rois=compute.full_rois(cfg)).mean_image(det)      # read here
            assert np.allclose(got, ref, atol=1e-4, equal_nan=True), (tiles, s)


def test_gui_remembers_agipd_tiles_with_the_rois(tmp_path, monkeypatch):
    """p7016: the ROIs sit on stacked tiles, so the tiles come back with them (closing the
    GUI, 'ROIs of run', and 'From file…' on an exported .h5)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("pyqtgraph")
    monkeypatch.setenv("XTG_MID_START_DIR", str(tmp_path))
    from pyqtgraph.Qt import QtWidgets
    from xtg_mid_gui.config import config_for
    from xtg_mid_gui.gui import XTGMidWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
    w = XTGMidWindow(config_for("7016"), sim_kw=dict(n_shots=10))
    w.load([101, 102])
    w.set_tiles([9, 8, 15])
    roi = w.views["agipd_sub"].panel.rois[0]
    roi.item.setPos(roi.item.pos() + pg_point(4, 3))
    moved = roi.geometry()
    out = tmp_path / "exp.h5"
    w.export_h5(out)
    w.close()
    w2 = XTGMidWindow(config_for("7016"), sim_kw=dict(n_shots=10))
    assert w2.cfg.detectors["agipd_sub"].tiles == [9, 8, 15]
    assert w2.mosaic.chosen() == [9, 8, 15]
    assert w2.views["agipd_sub"].panel.rois[0].geometry() == moved
    w2.load([101])
    assert w2.views["agipd_sub"].panel.rois[0].geometry() == moved
    w2.set_tiles([8, 10, 15])
    w2.views["agipd_sub"].panel.rois[0].item.setPos(pg_point(1, 1))
    w2.rois_from_file(str(out))                                         # from the exported file
    assert w2.cfg.detectors["agipd_sub"].tiles == [9, 8, 15]
    assert w2.views["agipd_sub"].panel.rois[0].geometry() == moved
    w2.set_tiles([8, 10, 15])
    w2.use_rois_of_run(102)                                            # 102: last analysed in the first session
    assert w2.cfg.detectors["agipd_sub"].tiles == [9, 8, 15]
    assert w2.views["agipd_sub"].panel.rois[0].geometry() == moved
    w2.close()
