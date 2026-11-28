"""Fit models ported from GaO_Analysis_mac.ipynb, FFT on uneven delays, and the Fitting page."""
import os

import numpy as np
import pytest

pytest.importorskip("scipy")
from xtg_mid_gui import fitting as F  # noqa: E402

TRUE = dict(t0=0.02, sigma=0.05, a_exp=0.8, tau=0.7, a1=0.12, nu1=2.5, phi1=0.4, a2=0.07, nu2=4.1,
            phi2=-0.5, tau2=0.6, y0=0.35)


def test_model_names_match_notebook_variants():
    assert F.ModelSpec(2, [False, True, True]).names() == \
        ["t0", "sigma", "a_exp", "tau", "a1", "nu1", "phi1", "a2", "nu2", "phi2", "tau2", "y0"]   # freq2_dump1
    assert len(F.ModelSpec(3, [True, True, True]).names()) == 17                                 # freq3_dump3
    assert F.ModelSpec(2, [False, True, False], harmonic=True).names()[-2:] == ["a_h", "y0"]       # 2nd harm


def test_thermal_is_model_without_oscillations():
    t = np.linspace(-0.5, 3, 200)
    spec = F.ModelSpec(2, [False, True, True])
    v = dict(TRUE, a1=0.0, a2=0.0)
    assert np.allclose(F.model(t, v, spec), F.thermal(t, v))
    assert F.thermal(np.array([-5.0]), TRUE)[0] == pytest.approx(0, abs=1e-12)


def test_fit_recovers_parameters_on_uneven_grid():
    spec = F.ModelSpec(2, [False, True, True])
    t = np.r_[np.arange(-0.3, 0, 0.06), np.arange(0, 1.24, 0.03), np.arange(1.3, 3.0, 0.1)]
    y = F.model(t, TRUE, spec) + np.random.default_rng(0).normal(0, 0.003, len(t))
    r = F.fit(t, y, spec, F.default_params(spec, t))
    assert r.success
    for k in ("nu1", "nu2", "tau", "a_exp"):
        assert r.values[k] == pytest.approx(TRUE[k], rel=0.05)


def test_fixed_parameter_is_held():
    spec = F.ModelSpec(1, [True, False, False])
    t = np.linspace(-0.3, 3, 120)
    y = F.model(t, dict(TRUE, tau1=0.8), spec)
    p = F.default_params(spec, t)
    p["sigma"].fixed, p["sigma"].value = True, 0.05
    r = F.fit(t, y, spec, p)
    assert r.values["sigma"] == 0.05 and r.errors["sigma"] == 0.0


def test_fft_on_uneven_delays_uses_even_grid():
    t = np.r_[np.arange(0, 1, 0.02), np.arange(1, 4, 0.05)]
    even, dmin, dmax, med = F.spacing_report(t)
    assert not even and dmin == pytest.approx(0.02) and dmax == pytest.approx(0.05)
    y = np.sin(2 * np.pi * 3.0 * t)
    tg, yg, f, a = F.fft_with_grid(t, y, step=0.02)
    assert np.allclose(np.diff(tg), 0.02)                       # even grid
    assert f[np.argmax(a)] == pytest.approx(3.0, abs=0.3)
    assert F.guess_frequencies(f, a, 1) == [pytest.approx(f[np.argmax(a)])]
    assert F.spacing_report(np.arange(0, 1, 0.1))[0]


def test_fit_page(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pyqtgraph.Qt import QtWidgets
    from xtg_mid_gui import theme
    from xtg_mid_gui.gui import XTGMidWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.apply(app, "dark")
    w = XTGMidWindow(sim_kw=dict(n_shots=20))
    w.load(w.backend.scans())
    fp = w.fit_page
    assert fp.cmb_roi.count() == 8 and "NOT evenly spaced" in fp.lbl_warn.text()
    assert fp.run_fit(True).success                            # notebook order: thermal first ...
    nus = fp.guess_nu()                                        # ... then ν from FFT of data - thermal
    assert len(nus) == 2 and min(abs(np.array(nus) - 2.38)) < 0.45     # 2.38 THz mode, within one FFT bin (0.41 THz)
    r = fp.run_fit(False)
    assert r.success and r.least_squares < 0.2
    fp.cmb_n.setCurrentText("3")
    assert "nu3" in fp.params and fp.result is None
    fp.table.item(5, 2).setText("1000")                        # edit a bound (nu1 min, GHz)
    assert fp.params["nu1"].lo == pytest.approx(1.0)
    w.close()


def test_even_grid_stays_inside_measured_delays():
    t = np.array([0.0, 0.1, 0.25, 0.3, 0.7, 1.05])
    tu, yu = F.uniform(t, t ** 2)
    assert tu[0] == 0.0 and tu[-1] <= t[-1] + 1e-12
    assert np.all(np.diff(tu) > 0)
    tu, _ = F.uniform(np.array([0.0, 0.0, 0.0, 0.5, 1.0]), np.arange(5.0))   # repeated delays: no crash
    assert len(tu) >= 2


def test_thermal_only_fit_reports_only_thermal_parameters():
    spec = F.ModelSpec(2, [False, True, True])
    t = np.linspace(-0.5, 3, 60)
    p = F.default_params(spec, t)
    y = F.thermal(t, {n: q.value for n, q in p.items()})
    r = F.fit(t, y, spec, p, thermal_only=True)
    assert set(r.fitted) == {"t0", "sigma", "a_exp", "tau", "y0"}


def test_fit_page_damped_boxes_and_oscillation_choice():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pyqtgraph.Qt import QtWidgets
    from xtg_mid_gui.gui import XTGMidWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = XTGMidWindow(sim_kw=dict(n_shots=10))
    w.load([101, 102, 103, 104, 105, 106])
    fp = w.fit_page
    assert [fp.cmb_n.itemText(i) for i in range(fp.cmb_n.count())] == ["1", "2", "3"]
    assert fp.cmb_n.minimumWidth() >= 60
    assert type(fp.cmb_n.view()) is QtWidgets.QListView
    fp.cmb_n.setCurrentText("3")
    ch3 = fp.chk_damp[2]
    assert not ch3.isHidden() and ch3.isEnabled() and ch3.isChecked()
    ch3.setChecked(False)                                       # nu3 can be un-ticked
    assert fp.spec.damped[2] is False and "tau3" not in fp.params
    fp.cmb_n.setCurrentText("2")
    assert ch3.isHidden()                                        # unused nu: no box at all
    fp.cmb_n.setCurrentText("1")
    assert fp.chk_damp[1].isHidden() and not fp.chk_damp[0].isHidden()
    w.close()
