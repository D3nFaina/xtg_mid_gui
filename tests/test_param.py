"""Parameter scanned inside a run or across runs (sample rotation, temperature ...)."""
import os
import sys
import types

import numpy as np
import pytest

from xtg_mid_gui import analysis as an
from xtg_mid_gui.backends import ExtraDataBackend, SimulatedBackend, match_trains
from xtg_mid_gui.config import PARAM_PRESETS, Source, SourceConfig, config_for


def test_param_series_steps_and_bins():
    rng = np.random.default_rng(1)
    x = np.repeat(np.linspace(0, 2, 5), 40) + rng.normal(0, 1e-4, 200)
    y = 1 + x
    c, v, e, n = an.param_series(x, y)
    assert len(c) == 5 and np.allclose(v, 1 + np.linspace(0, 2, 5), atol=1e-3) and np.all(n == 40)
    xc = rng.uniform(0, 10, 2000)                                   # continuous drift -> bins
    assert an.param_steps(xc) is None
    c, v, e, n = an.param_series(xc, 2 * xc, mode="auto", nbins=20)
    assert len(c) == 20 and np.allclose(v, 2 * c, rtol=0.05)
    c, v, e, n = an.param_series(x, y, np.full_like(y, 2.0), mode="bins", nbins=10)
    assert np.allclose(v, (1 + c) / 2, atol=0.01)                  # normalised: ratio of means
    assert an.param_series([np.nan], [1.0])[0].size == 0


def test_simulated_param_only_when_enabled():
    cfg = SourceConfig()
    assert SimulatedBackend(cfg, n_shots=18).load_scan(101).param is None
    cfg.param_enabled, cfg.param = True, Source(*PARAM_PRESETS["phi, sample rotation (motor T3)"][:2])
    d = SimulatedBackend(cfg, n_shots=18).load_scan(101)
    assert d.param.shape == (18,) and len(an.param_steps(d.param)) == 9


def test_match_trains():
    out = match_trains(np.array([10, 11, 13]), np.array([1.0, 2.0, 3.0]), np.array([11, 12, 13]))
    assert np.allclose(out, [2.0, np.nan, 3.0], equal_nan=True)


def test_extra_data_reads_param(monkeypatch):
    from test_presets import _ByIndex, _ById, _KD, _fake_run
    run = _fake_run()
    run.keys[("MID_EXP_UPP/MDL/DATA_SELECTOR_1", "midExpUppMotorT3.actualPosition")] = _KD(
        np.arange(1000, 1006), np.array([0.0, 0, 0.5, 0.5, 1.0, 1.0]))
    monkeypatch.setitem(sys.modules, "extra_data",
                        types.SimpleNamespace(open_run=lambda *a, **k: run, by_index=_ByIndex(), by_id=_ById()))
    cfg = config_for("7016")
    assert cfg.param.key == "midExpUppMotorT3.actualPosition" and not cfg.param_enabled
    be = ExtraDataBackend(cfg)
    assert be.load_scan(42).param is None                      # off: not read
    cfg.param_enabled = True
    d = be.load_scan(42)
    assert np.allclose(d.param, [0, 0, 0.5, 0.5, 1, 1])
    assert np.allclose(be.load_param(42, d.train), d.param)    # fast path: parameter only
    cfg.param = Source("WRONG/NAME", "x")
    assert be.load_scan(42).param is None                      # a wrong name does not lose the run


# ---------------------------------------------------------------- GUI
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_param_page_pools_runs_and_finds_the_peak():
    pytest.importorskip("pyqtgraph")
    from pyqtgraph.Qt import QtWidgets
    from xtg_mid_gui.gui import XTGMidWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = XTGMidWindow(sim_kw=dict(n_shots=45))
    w.load([101])
    assert not w.tabs.isTabVisible(w.param_tab)
    w.b_param.setChecked(True)
    assert w.tabs.isTabVisible(w.param_tab) and "Read" in w.lbl_pinfo.text()
    w.cmb_ppreset.setCurrentText("phi, sample rotation (motor T3)")
    assert w.ed_pkey.text() == "midExpUppMotorT3.actualPosition" and w.ed_punit.text() == "deg"
    w.apply_param()                                             # single run
    curves = {c.name(): c.getData() for c in w.pp_norm.listDataItems()}
    x, y = curves["Jungfrau 1 · TG +1"]
    assert len(x) == 9 and abs(x[np.nanargmax(y)] - 0.25) < 0.2
    w.load([102, 103], append=True)                             # more runs: read on load, pooled
    assert all(w.scans[s].param is not None for s in (101, 102, 103))
    assert "3 scans" in w.lbl_pinfo.text()
    w.cmb_pgroup.setCurrentText("one curve per scan")
    assert len(w.pp_norm.listDataItems()) == 3
    w.cmb_pgroup.setCurrentText("one curve per delay")
    w.cmb_pbin.setCurrentText("equal-width bins")
    w.sp_pbins.setValue(5)
    assert all(len(c.getData()[0]) <= 5 for c in w.pp_norm.listDataItems())
    w.b_param.setChecked(False)
    assert not w.tabs.isTabVisible(w.param_tab) and not w.cfg.param_enabled
    w.close()
