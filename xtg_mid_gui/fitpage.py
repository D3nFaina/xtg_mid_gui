"""Fitting page: fit a delay-series ROI curve with Den's XTG models (1-3
oscillations on a thermal decay), with editable start values and bounds.

Panels: data + full fit + thermal part | thermal part alone
        data - thermal (measured + interpolated points) | FFT of data - thermal
        residual (measured + interpolated points) | FFT of residual

The delays are usually NOT evenly spaced, so every FFT is computed on an even
grid obtained by linear interpolation; the page warns about it and plots the
measured points together with the interpolated ones.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from . import analysis as an
from . import export
from . import fitting as F
from . import theme

SESSION = "this session (live)"

COLS = ["parameter", "start", "min", "max", "fix", "result ± error"]
OSC_COLORS = [(18, 170, 130), (230, 73, 128), (132, 94, 247)]


def _num(text, default):
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _narrow(*combos):
    """Combo boxes that may get narrower than their longest item (long file / mode names)."""
    for c in combos:
        c.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        c.setMinimumContentsLength(8)


class FitPage(QtWidgets.QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.spec = F.ModelSpec(2, [False, True, True], False)
        self.params = F.default_params(self.spec)
        self.result: F.FitResult | None = None
        self._range_set = False
        self._file_traces = {}            # (path, detector, normalised) -> {roi: (t, y, e, n)}
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.inner = QtWidgets.QTabWidget()
        self.inner.addTab(split, "Fit")
        self.compare = ComparePanel(self)
        self.inner.addTab(self.compare, "Compare traces")
        lay.addWidget(self.inner)
        split.addWidget(self._build_controls())
        split.addWidget(self._build_plots())
        split.setSizes([300, 1060])

    # ============================================================ controls
    def _build_controls(self):
        from .gui import Card, _label, _spin
        side = QtWidgets.QWidget(objectName="sidebar")
        v = QtWidgets.QVBoxLayout(side)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(10)

        c = Card("Data to fit")
        f = c.form()
        self.cmb_roi = QtWidgets.QComboBox()
        self.cmb_curve = QtWidgets.QComboBox()
        self.cmb_curve.addItems(["normalised", "raw"])
        self.chk_scale = QtWidgets.QCheckBox("scale to max = 1 (as in the notebook)")
        self.chk_scale.setChecked(True)
        self.sp_tmin = _spin(-1e4, 1e4, -1.0, 0.05, 3)
        self.sp_tmax = _spin(-1e4, 1e4, 10.0, 0.05, 3)
        b_full = QtWidgets.QPushButton("full range")
        b_full.clicked.connect(self._full_range)

        self.cmb_src = QtWidgets.QComboBox()
        self.cmb_src.addItem(SESSION)
        self.cmb_src.setToolTip("fit the delay series computed now, or one saved with Export (.h5)")
        b_open = QtWidgets.QPushButton("Open file…")
        b_open.clicked.connect(self.choose_file)
        self.cmb_det = QtWidgets.QComboBox()
        self.cmb_det.setToolTip("detector of the saved file")
        _narrow(self.cmb_src, self.cmb_det, self.cmb_roi, self.cmb_curve)
        f.addRow("data from", self.cmb_src)
        f.addRow("", b_open)
        f.addRow("detector", self.cmb_det)
        self.cmb_src.currentIndexChanged.connect(lambda *_: self._source_changed())
        self.cmb_det.currentIndexChanged.connect(lambda *_: (self._fill_rois(), self._data_changed()))
        f.addRow("ROI curve", self.cmb_roi)
        f.addRow("series", self.cmb_curve)
        f.addRow("fit from (ps)", self.sp_tmin)
        f.addRow("fit to (ps)", self.sp_tmax)
        f.addRow("", b_full)
        c.body.addWidget(self.chk_scale)
        for w in (self.cmb_roi, self.cmb_curve):
            w.currentTextChanged.connect(lambda *_: self._data_changed())
        self.chk_scale.toggled.connect(lambda *_: self._data_changed())
        for w in (self.sp_tmin, self.sp_tmax):
            w.valueChanged.connect(lambda *_: self.draw())
        v.addWidget(c)

        c = Card("Model")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(_label("oscillations", "muted", wrap=False))
        self.cmb_n = QtWidgets.QComboBox()
        self.cmb_n.setView(QtWidgets.QListView())      # macOS: styled popups need a list view to show the items
        self.cmb_n.addItems(["1", "2", "3"])
        self.cmb_n.setMinimumWidth(70)                  # macOS: room for the number next to the arrow
        self.cmb_n.setMinimumContentsLength(2)
        self.cmb_n.setCurrentText(str(self.spec.n_osc))
        row.addWidget(self.cmb_n)
        row.addStretch(1)
        c.body.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.chk_damp = []
        for i in range(3):
            ch = QtWidgets.QCheckBox(f"ν{i + 1} damped")
            ch.setChecked(self.spec.damped[i])
            ch.setToolTip(f"multiply oscillation {i + 1} by exp(-(t-t0)/τ{i + 1})")
            self.chk_damp.append(ch)
            row.addWidget(ch)
        row.addStretch(1)
        c.body.addLayout(row)
        self._show_damp_boxes(self.spec.n_osc)
        self.chk_harm = QtWidgets.QCheckBox("+ 2nd harmonic of ν1 (damped, τ/2)")
        self.chk_harm.setToolTip("2nd harmonic of ν1, damped with τ/2 (notebook: freq3_dump2_2nd_harm)")
        c.body.addWidget(self.chk_harm)
        c.body.addWidget(_label(
            "S(t) = | ½[1+erf((t−t0)/σ)] · ( a_exp e^(−(t−t0)/τ) − Σ aᵢ sin(2πνᵢ(t−t0)+φᵢ) [e^(−(t−t0)/τᵢ)] + y0 ) |²",
            "muted"))
        self.cmb_n.currentTextChanged.connect(self._model_changed)
        for ch in self.chk_damp + [self.chk_harm]:
            ch.toggled.connect(self._model_changed)
        v.addWidget(c)

        c = Card("Parameters")
        self.table = QtWidgets.QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(5, QtWidgets.QHeaderView.Stretch)
        self.table.setMinimumHeight(330)
        self.table.itemChanged.connect(self._table_edited)
        c.body.addWidget(self.table)
        self.b_fit = QtWidgets.QPushButton("Fit", objectName="primary")
        self.b_th = QtWidgets.QPushButton("Fit thermal only")
        self.b_guess = QtWidgets.QPushButton("Guess ν from FFT")
        self.b_use = QtWidgets.QPushButton("Results → start")
        self.b_reset = QtWidgets.QPushButton("Reset")
        for bs in ((self.b_fit, self.b_th), (self.b_guess, self.b_use), (self.b_reset,)):
            row = QtWidgets.QHBoxLayout()
            for b in bs:
                row.addWidget(b)
            c.body.addLayout(row)
        self.b_fit.clicked.connect(lambda: self.run_fit(False))
        self.b_th.clicked.connect(lambda: self.run_fit(True))
        self.b_guess.clicked.connect(self.guess_nu)
        self.b_use.clicked.connect(self.results_to_start)
        self.b_reset.clicked.connect(self.reset)
        self.lbl_result = _label("not fitted yet: curves show the start values", "muted")
        c.body.addWidget(self.lbl_result)
        c.body.addWidget(_label("ν in GHz, t0 / σ / τ in ps, φ in rad. Tick 'fix' to hold a parameter.", "muted"))
        v.addWidget(c)

        c = Card("FFT")
        f = c.form()
        self.sp_fft_tmin = _spin(-1e4, 1e4, 0.05, 0.01, 3)
        self.sp_fft_tmin.setToolTip("only points with t ≥ this go into the FFTs (the notebook's 'cut')")
        self.sp_step = _spin(0, 1e5, 0, 5, 1)
        self.sp_step.setSpecialValueText("auto (median step)")
        self.sp_step.setToolTip("even grid step for the interpolation before the FFT (fs); 0 = median step")
        f.addRow("FFT from (ps)", self.sp_fft_tmin)
        f.addRow("grid step (fs)", self.sp_step)
        for w in (self.sp_fft_tmin, self.sp_step):
            w.valueChanged.connect(lambda *_: self.draw())
        v.addWidget(c)
        v.addStretch(1)
        sc = QtWidgets.QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(side)
        sc.setMinimumWidth(292)
        sc.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._rebuild_table()
        return sc

    def _build_plots(self):
        from .gui import _label
        box = QtWidgets.QWidget()
        bv = QtWidgets.QVBoxLayout(box)
        bv.setContentsMargins(4, 4, 4, 4)
        self.lbl_warn = _label("", "tbc")            # FFT / uneven-delay warning, always visible
        bv.addWidget(self.lbl_warn)
        g = theme.register_widget(pg.GraphicsLayoutWidget())
        bv.addWidget(g, 1)
        S = theme.style_plot
        self.p_fit = S(g.addPlot(row=0, col=0), "Data, full fit and thermal part", "Δt (ps)", "TG signal")
        self.p_th = S(g.addPlot(row=0, col=1), "Thermal part alone", "Δt (ps)", "TG signal")
        self.p_osc = S(g.addPlot(row=1, col=0), "Data − thermal (● measured, ✕ interpolated)", "Δt (ps)", "signal")
        self.p_fosc = S(g.addPlot(row=1, col=1), "FFT of data − thermal", "ν (GHz)", "FFT amplitude")
        self.p_res = S(g.addPlot(row=2, col=0), "Residual (● measured, ✕ interpolated)", "Δt (ps)", "data − fit")
        self.p_fres = S(g.addPlot(row=2, col=1), "FFT of residual", "ν (GHz)", "FFT amplitude")
        for p in (self.p_fit, self.p_osc, self.p_res):
            p.addLegend(offset=(-10, 10))
            S(p, p._xtg["title"], p._xtg["x"], p._xtg["y"])
        self.p_th.setXLink(self.p_fit)
        self.p_osc.setXLink(self.p_fit)
        self.p_res.setXLink(self.p_fit)
        self.p_fres.setXLink(self.p_fosc)
        return box

    # ============================================================ params table
    def _rebuild_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.params))
        r = self.result
        for i, p in enumerate(self.params.values()):
            unit = f" [{p.unit}]" if p.unit else ""
            it = QtWidgets.QTableWidgetItem(p.name.replace("nu", "ν").replace("phi", "φ").replace("tau", "τ")
                                            .replace("sigma", "σ") + unit)
            it.setFlags(QtCore.Qt.ItemIsEnabled)
            it.setData(QtCore.Qt.UserRole, p.name)
            self.table.setItem(i, 0, it)
            for j, val in ((1, p.value), (2, p.lo), (3, p.hi)):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(f"{val * p.display:.6g}"))
            fx = QtWidgets.QTableWidgetItem()
            fx.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            fx.setCheckState(QtCore.Qt.Checked if p.fixed else QtCore.Qt.Unchecked)
            self.table.setItem(i, 4, fx)
            txt = ""
            if r is not None and p.name in r.values and (not r.fitted or p.name in r.fitted):
                e = r.errors.get(p.name, np.nan)
                txt = f"{r.values[p.name] * p.display:.5g}" + ("" if not np.isfinite(e) else f" ± {e * p.display:.2g}")
                lo, hi = p.lo, p.hi
                if not p.fixed and (np.isclose(r.values[p.name], lo, rtol=1e-4) or np.isclose(r.values[p.name], hi, rtol=1e-4)):
                    txt += "  ⚠ at bound"
            res = QtWidgets.QTableWidgetItem(txt)
            res.setFlags(QtCore.Qt.ItemIsEnabled)
            self.table.setItem(i, 5, res)
        self.table.blockSignals(False)

    def _table_edited(self, item):
        name = self.table.item(item.row(), 0).data(QtCore.Qt.UserRole)
        p = self.params[name]
        col = item.column()
        if col == 4:
            p.fixed = item.checkState() == QtCore.Qt.Checked
            return
        attr = {1: "value", 2: "lo", 3: "hi"}.get(col)
        if attr:
            setattr(p, attr, _num(item.text(), getattr(p, attr) * p.display) / p.display)
            if col == 1:
                self.draw()

    # ============================================================ actions
    def _model_changed(self, *_):
        spec = F.ModelSpec(int(self.cmb_n.currentText()), [c.isChecked() for c in self.chk_damp],
                           self.chk_harm.isChecked())
        self._show_damp_boxes(spec.n_osc)
        t, _, _ = self.data()
        new = F.default_params(spec, t)
        for n in new:                                   # keep what the user already set
            if n in self.params:
                new[n] = self.params[n]
        self.spec, self.params, self.result = spec, new, None
        self._rebuild_table()
        self.draw()

    def _show_damp_boxes(self, n):
        """Only the oscillations in the model get a 'damped' box: a greyed-out ticked box
        for an unused ν looked selected and could not be unticked."""
        for i, ch in enumerate(self.chk_damp):
            ch.setVisible(i < n)
            ch.setEnabled(True)

    def reset(self):
        t, _, _ = self.data()
        self.params = F.default_params(self.spec, t)
        self.result = None
        self._rebuild_table()
        self.draw()

    def results_to_start(self):
        if self.result is None:
            return
        for n, val in self.result.values.items():
            if n in self.params:
                p = self.params[n]
                p.value = float(np.clip(val, p.lo, p.hi))
        self._rebuild_table()

    def run_fit(self, thermal_only=False):
        t, y, _ = self.data(in_range=True)
        if len(t) < 5:
            self.lbl_result.setText("not enough points in the fit range")
            return None
        self.result = F.fit(t, y, self.spec, self.params, thermal_only=thermal_only)
        r = self.result
        what = "thermal only" if thermal_only else self.spec.label()
        self.lbl_result.setText(
            (f"{what}: least squares = {r.least_squares:.4g}" if r.success else f"fit failed: {r.message}")
            + f"   ({len(t)} points)")
        if thermal_only and r.success:                 # thermal values become the start values
            for n in ("t0", "sigma", "a_exp", "tau", "y0"):
                self.params[n].value = float(np.clip(r.values[n], self.params[n].lo, self.params[n].hi))
        self._rebuild_table()
        self.draw()
        return r

    def guess_nu(self):
        """Fill ν start values (and ±20 % bounds) from the strongest peaks of the
        FFT of data − thermal."""
        t, y, _ = self.data(in_range=True)
        v = self.values()
        f, a = F.fft_amplitude(t, y - F.thermal(t, v), self.sp_fft_tmin.value(), self.step())
        nus = F.guess_frequencies(f, a, self.spec.n_osc, fmin=2.5 * f[1] if len(f) > 1 else None)
        for i, nu in enumerate(nus, 1):
            p = self.params[f"nu{i}"]
            p.value, p.lo, p.hi = nu, 0.8 * nu, 1.2 * nu
        self._rebuild_table()
        self.draw()
        return nus

    def _full_range(self):
        t, _, _ = self.data(in_range=False)
        if len(t):
            self.sp_tmin.setValue(np.floor(t.min() * 1000) / 1000)    # 3 decimals: do not cut the end points
            self.sp_tmax.setValue(np.ceil(t.max() * 1000) / 1000)

    def _data_changed(self):
        self.result = None
        self._rebuild_table()
        self.draw()

    # ============================================================ data
    def step(self):
        return self.sp_step.value() / 1000.0 or None     # fs -> ps, 0 = auto

    # ------------------------------------------------------------ saved files
    def from_file(self):
        return self.cmb_src.currentIndex() > 0

    def add_file(self, path):
        """Offer a saved reduced-data file as data source (and select it)."""
        path = str(path)
        if not export.is_reduced(path):
            self.win.log(f"{path}: not a reduced-data file from Export (.h5)")
            return False
        i = self.cmb_src.findData(path)
        if i < 0:
            import os
            self.cmb_src.addItem(os.path.basename(path), path)
            self.cmb_src.setItemData(self.cmb_src.count() - 1, path, QtCore.Qt.ToolTipRole)
            i = self.cmb_src.count() - 1
        self.cmb_src.setCurrentIndex(i)
        return True

    def choose_file(self):
        from .delays import start_dir
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Reduced data file", str(start_dir()),
                                                        "HDF5 (*.h5 *.hdf5);;All files (*)")
        if path:
            self.add_file(path)

    def file_traces(self, path, det, normalised=True):
        key = (path, det, normalised)
        if key not in self._file_traces:
            self._file_traces[key] = export.traces(path, det, normalised)
        return self._file_traces[key]

    def _source_changed(self):
        self.cmb_det.blockSignals(True)
        self.cmb_det.clear()
        if self.from_file():
            self._file_traces = {k: v for k, v in self._file_traces.items() if k[0] != self.cmb_src.currentData()}
            self.cmb_det.addItems(export.detectors(self.cmb_src.currentData()))
        self.cmb_det.setEnabled(self.from_file())
        self.cmb_det.blockSignals(False)
        self._range_set = False
        self._fill_rois()
        self.refresh()

    def curve_names(self):
        if self.from_file():
            if not self.cmb_det.currentText():
                return []
            return list(self.file_traces(self.cmb_src.currentData(), self.cmb_det.currentText()))
        return list(self.win.series)

    def _fill_rois(self):
        cur = self.cmb_roi.currentText()
        self.cmb_roi.blockSignals(True)
        self.cmb_roi.clear()
        self.cmb_roi.addItems(self.curve_names())
        i = self.cmb_roi.findText(cur)
        self.cmb_roi.setCurrentIndex(i if i >= 0 else 0)
        self.cmb_roi.blockSignals(False)

    def curve(self):
        """(t fs, y, e, n) of the chosen curve, or None."""
        norm = self.cmb_curve.currentText() == "normalised"
        if self.from_file():
            if not self.cmb_det.currentText():
                return None
            return self.file_traces(self.cmb_src.currentData(), self.cmb_det.currentText(), norm).get(
                self.cmb_roi.currentText())
        s = self.win.series.get(self.cmb_roi.currentText())
        return None if s is None else s["norm" if norm else "raw"]

    def data(self, in_range=True):
        """(t [ps], y, err) of the chosen delay-series curve."""
        c = self.curve()
        if c is None:
            z = np.array([])
            return z, z, z
        t, y, e, _ = c
        t, y, e = np.asarray(t) / 1000.0, np.asarray(y, float), np.asarray(e, float)
        if self.chk_scale.isChecked() and len(y) and np.nanmax(np.abs(y)) > 0:
            k = np.nanmax(np.abs(y))
            y, e = y / k, e / k
        if in_range:
            m = (t >= self.sp_tmin.value()) & (t <= self.sp_tmax.value())
            t, y, e = t[m], y[m], e[m]
        return t, y, e

    def values(self):
        return self.result.values if self.result is not None else {n: p.value for n, p in self.params.items()}

    def refresh(self):
        """Called after every recompute of the main window."""
        self._fill_rois()
        self.compare.refresh()
        if not self._range_set:
            t, _, _ = self.data(in_range=False)
            if len(t):
                self._range_set = True
                self._full_range()
                self.params = F.default_params(self.spec, t)
                self._rebuild_table()
        self.draw()

    # ============================================================ drawing
    def _warn(self, t):
        even, dmin, dmax, med = F.spacing_report(t)
        step = (self.step() or med) * 1000
        if even:
            self.lbl_warn.setText(f"Delays are evenly spaced ({med * 1000:.4g} fs): the FFT uses the points directly.")
            self.lbl_warn.setObjectName("muted")
        else:
            self.lbl_warn.setText(
                f"⚠ Delays are NOT evenly spaced (steps {dmin * 1000:.4g} to {dmax * 1000:.4g} fs). Before every FFT "
                f"the points are linearly interpolated onto an even {step:.4g} fs grid; the plots show measured (●) and "
                f"interpolated (✕) points. Features finer than the coarsest step are not real.")
            self.lbl_warn.setObjectName("tbc")
        self.lbl_warn.style().unpolish(self.lbl_warn)
        self.lbl_warn.style().polish(self.lbl_warn)
        return even

    def _points_and_grid(self, p, t, y, tg, yg, color, name):
        p.plot(tg, yg, pen=pg.mkPen(color + (120,), width=1), symbol="x", symbolSize=6,
               symbolPen=pg.mkPen(color + (170,)), name=f"{name} interpolated")
        p.plot(t, y, pen=None, symbol="o", symbolSize=7, symbolBrush=color, symbolPen=None, name=f"{name} measured")
        p.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen(theme.neutral(110))))
        p.addItem(pg.InfiniteLine(self.sp_fft_tmin.value(), angle=90,
                                  pen=pg.mkPen(theme.neutral(140), style=QtCore.Qt.DashLine)))

    def draw(self):
        for p in (self.p_fit, self.p_th, self.p_osc, self.p_fosc, self.p_res, self.p_fres):
            p.clear()
        t_all, y_all, e_all = self.data(in_range=False)
        t, y, e = self.data(in_range=True)
        if len(t) < 2:
            return
        self._warn(t)
        v = self.values()
        fitted = self.result is not None and self.result.success
        txt = theme.pal()["text"]
        sig_col = theme.PROBE
        out = ~np.isin(t_all, t)
        if out.any():                                           # all points joined, grey outside the range
            k = np.argsort(t_all)
            self.p_fit.plot(t_all[k], y_all[k], pen=pg.mkPen(theme.neutral(110), width=1), symbol="o", symbolSize=5,
                            symbolBrush=theme.neutral(110), symbolPen=None, name="outside fit range")
        self.p_fit.addItem(theme.error_bars(t, y, e, sig_col))
        k = np.argsort(t)
        self.p_fit.plot(t[k], y[k], pen=pg.mkPen(sig_col, width=1.5), symbol="o", symbolSize=6, symbolBrush=sig_col,
                        symbolPen=None, name="data")
        tt = np.linspace(t.min(), t.max(), 1500)
        full, th = F.model(tt, v, self.spec), F.thermal(tt, v)
        lab = "full fit" if fitted else "start values"
        self.p_fit.plot(tt, full, pen=pg.mkPen(txt, width=2, style=QtCore.Qt.SolidLine if fitted else QtCore.Qt.DotLine),
                        name=lab)
        self.p_fit.plot(tt, th, pen=pg.mkPen(theme.PUMP, width=1.5, style=QtCore.Qt.DashLine), name="thermal part")
        theme.set_title(self.p_fit, f"Data, {'full fit' if fitted else 'start values'} and thermal part")

        self.p_th.plot(t, y, pen=None, symbol="o", symbolSize=5, symbolBrush=sig_col + (90,), symbolPen=None)
        self.p_th.plot(tt, th, pen=pg.mkPen(theme.PUMP, width=2.5))
        tau_e = self.result.errors.get("tau", np.nan) if fitted else np.nan
        theme.set_title(self.p_th, f"Thermal part alone: τ = {v['tau']:.4g}"
                        + (f" ± {tau_e:.2g}" if np.isfinite(tau_e) else "") + " ps")

        # data - thermal and its FFT (on the interpolated even grid)
        osc = y - F.thermal(t, v)
        tmin, step = self.sp_fft_tmin.value(), self.step()
        tg, og, f1, a1 = F.fft_with_grid(t, osc, tmin, step)
        keep = t >= tmin
        self._points_and_grid(self.p_osc, t[keep], osc[keep], tg, og, sig_col, "data − thermal")
        if len(f1):
            self.p_fosc.plot(f1 * 1000, a1, pen=pg.mkPen(sig_col, width=2), symbol="o", symbolSize=4,
                             symbolBrush=sig_col, symbolPen=None, fillLevel=0, brush=sig_col + (40,))
        # full fit residual and its FFT
        res = y - F.model(t, v, self.spec)
        tg2, rg, f2, a2 = F.fft_with_grid(t, res, tmin, step)
        self._points_and_grid(self.p_res, t[keep], res[keep], tg2, rg, theme.PROBE if not fitted else (225, 160, 0),
                              "residual")
        if len(f2):
            c = (225, 160, 0)
            self.p_fres.plot(f2 * 1000, a2, pen=pg.mkPen(c, width=2), symbol="o", symbolSize=4,
                             symbolBrush=c, symbolPen=None, fillLevel=0, brush=c + (40,))
        theme.set_title(self.p_res, f"Residual{'' if fitted else ' (start values)'}: Σres² = {np.nansum(res ** 2):.3g}")
        # mark the model frequencies on both FFTs
        for i in range(1, self.spec.n_osc + 1):
            nu = v.get(f"nu{i}")
            if nu is None:
                continue
            for p in (self.p_fosc, self.p_fres):
                p.addItem(pg.InfiniteLine(nu * 1000, angle=90, label=f"ν{i} {nu * 1000:.0f}",
                                          labelOpts=dict(position=0.93 - 0.13 * (i - 1), color=OSC_COLORS[i - 1]),
                                          pen=pg.mkPen(OSC_COLORS[i - 1], width=1.5, style=QtCore.Qt.DashLine)))
        if self.spec.harmonic and "nu1" in v:
            self.p_fosc.addItem(pg.InfiniteLine(2 * v["nu1"] * 1000, angle=90, label="2ν1",
                                                labelOpts=dict(position=0.8, color=OSC_COLORS[0]),
                                                pen=pg.mkPen(OSC_COLORS[0], style=QtCore.Qt.DotLine)))


# ================================================================ comparison panel
class ComparePanel(QtWidgets.QWidget):
    """Overlay delay traces from this session and from saved files: plain, 0-to-1, or each
    divided by its maximum inside a delay range."""

    def __init__(self, page):
        super().__init__()
        from .gui import Card, _label, _spin
        self.page = page
        self.traces = []                  # dicts: label, t, y, e (raw and norm), color, on
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)
        side = QtWidgets.QWidget(objectName="sidebar")
        v = QtWidgets.QVBoxLayout(side)
        v.setContentsMargins(6, 6, 6, 6)

        c = Card("Add traces")
        f = c.form()
        self.cmb_src = QtWidgets.QComboBox()
        self.cmb_src.addItem(SESSION)
        b_open = QtWidgets.QPushButton("Open file…")
        b_open.clicked.connect(self.choose_file)
        f.addRow("from", self.cmb_src)
        f.addRow("", b_open)
        self.cmb_det = QtWidgets.QComboBox()
        f.addRow("detector", self.cmb_det)
        self.cmb_roi = QtWidgets.QComboBox()
        f.addRow("ROI", self.cmb_roi)
        b_add = QtWidgets.QPushButton("Add", objectName="primary")
        b_add.clicked.connect(lambda: self.add_current())
        b_all = QtWidgets.QPushButton("Add all ROIs")
        b_all.clicked.connect(lambda: self.add_current(all_rois=True))
        arow = QtWidgets.QHBoxLayout()
        arow.addWidget(b_add)
        arow.addWidget(b_all)
        c.body.addLayout(arow)
        c.body.addWidget(_label("Traces from this session are a snapshot: add them again after changing "
                                "ROIs, cuts or normalisation to compare before / after.", "muted"))
        self.cmb_src.currentIndexChanged.connect(lambda *_: self._src_changed())
        self.cmb_det.currentIndexChanged.connect(lambda *_: self._fill_rois())
        v.addWidget(c)

        c = Card("Traces")
        self.lst = QtWidgets.QListWidget()
        self.lst.setMinimumHeight(180)
        self.lst.setStyleSheet("QListWidget { background: transparent; }")
        self.lst.itemChanged.connect(lambda it: self._toggled(it))
        c.body.addWidget(self.lst)
        rrow = QtWidgets.QHBoxLayout()
        b_rm = QtWidgets.QPushButton("Remove")
        b_rm.clicked.connect(self.remove_selected)
        b_clr = QtWidgets.QPushButton("Clear")
        b_clr.clicked.connect(self.clear)
        rrow.addWidget(b_rm)
        rrow.addWidget(b_clr)
        c.body.addLayout(rrow)
        v.addWidget(c)

        c = Card("Comparison")
        f = c.form()
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(list(an.COMPARE_MODES))
        self.cmb_series = QtWidgets.QComboBox()
        self.cmb_series.addItems(["normalised", "raw"])
        self.cmb_series.setToolTip("normalised = divided by the pump / probe normaliser chosen when computed")
        self.sp_t1 = _spin(-1e6, 1e6, 0, 10, 1)
        self.sp_t2 = _spin(-1e6, 1e6, 500, 10, 1)
        _narrow(self.cmb_src, self.cmb_det, self.cmb_roi, self.cmb_mode, self.cmb_series)
        self.cmb_mode.setToolTip("\n".join(an.COMPARE_MODES))
        f.addRow("mode", self.cmb_mode)
        f.addRow("series", self.cmb_series)
        f.addRow("peak from (fs)", self.sp_t1)
        f.addRow("peak to (fs)", self.sp_t2)
        self.lbl_info = _label("", "muted")
        f.addRow(self.lbl_info)
        for w in (self.cmb_mode, self.cmb_series):
            w.currentIndexChanged.connect(lambda *_: self.draw())
        for w in (self.sp_t1, self.sp_t2):
            w.valueChanged.connect(lambda *_: self.draw())
        v.addWidget(c)
        v.addStretch(1)
        sc = QtWidgets.QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(side)
        sc.setMinimumWidth(260)
        sc.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        split.addWidget(sc)
        g = theme.register_widget(pg.GraphicsLayoutWidget())
        self.plot = theme.style_plot(g.addPlot(), "Compared traces", "delay (fs)", "signal")
        self.plot.addLegend(offset=(-10, 10))
        theme.style_plot(self.plot, "Compared traces", "delay (fs)", "signal")
        self.region = pg.LinearRegionItem(values=(0, 500), movable=True, brush=pg.mkBrush(128, 128, 128, 45))
        self.region.setZValue(-10)
        self.region.sigRegionChangeFinished.connect(self._region_moved)
        split.addWidget(g)
        split.setSizes([280, 1080])
        self._src_changed()

    # ------------------------------------------------------------ sources
    def choose_file(self):
        from .delays import start_dir
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Reduced data file", str(start_dir()),
                                                        "HDF5 (*.h5 *.hdf5);;All files (*)")
        if path:
            self.add_file(path)

    def add_file(self, path):
        import os
        path = str(path)
        if not export.is_reduced(path):
            self.page.win.log(f"{path}: not a reduced-data file from Export (.h5)")
            return False
        i = self.cmb_src.findData(path)
        if i < 0:
            self.cmb_src.addItem(os.path.basename(path), path)
            i = self.cmb_src.count() - 1
        self.cmb_src.setCurrentIndex(i)
        return True

    def _src_changed(self):
        self.cmb_det.blockSignals(True)
        self.cmb_det.clear()
        if self.cmb_src.currentIndex() > 0:
            self.cmb_det.addItems(export.detectors(self.cmb_src.currentData()))
        else:
            win = self.page.win
            dets = dict.fromkeys(win.cfg.detectors[s["det"]].label for s in win.series.values())
            self.cmb_det.addItems(list(dets))
        self.cmb_det.blockSignals(False)
        self._fill_rois()

    def _session_curves(self, det_label):
        win = self.page.win
        return {s["name"]: s for s in win.series.values() if win.cfg.detectors[s["det"]].label == det_label}

    def _fill_rois(self):
        cur = self.cmb_roi.currentText()
        self.cmb_roi.clear()
        det = self.cmb_det.currentText()
        if not det:
            return
        if self.cmb_src.currentIndex() > 0:
            names = list(export.traces(self.cmb_src.currentData(), det))
        else:
            names = list(self._session_curves(det))
        self.cmb_roi.addItems(names)
        i = self.cmb_roi.findText(cur)
        self.cmb_roi.setCurrentIndex(max(i, 0))

    def refresh(self):
        """Main window recomputed: update the session's detector / ROI lists."""
        if self.cmb_src.currentIndex() == 0:
            cur_det = self.cmb_det.currentText()
            self._src_changed()
            i = self.cmb_det.findText(cur_det)
            if i >= 0:
                self.cmb_det.setCurrentIndex(i)

    # ------------------------------------------------------------ trace list
    def add_current(self, all_rois=False):
        det = self.cmb_det.currentText()
        if not det:
            return 0
        names = [self.cmb_roi.itemText(i) for i in range(self.cmb_roi.count())] if all_rois \
            else [self.cmb_roi.currentText()]
        n = 0
        for roi in names:
            if not roi:
                continue
            if self.cmb_src.currentIndex() > 0:
                path = self.cmb_src.currentData()
                raw, norm = export.traces(path, det, False).get(roi), export.traces(path, det, True).get(roi)
                where = self.cmb_src.currentText()
            else:
                s = self._session_curves(det).get(roi)
                if s is None:
                    continue
                raw, norm = s["raw"], s["norm"]
                where = f"session {QtCore.QTime.currentTime().toString('HH:mm')}"
            if raw is None or norm is None:
                continue
            self.add_trace(f"{roi} · {det} · {where}", raw, norm)
            n += 1
        return n

    def add_trace(self, label, raw, norm):
        copy = lambda c: tuple(np.array(a, float) for a in c[:3])      # noqa: E731
        color = theme.ROI_COLORS[len(self.traces) % len(theme.ROI_COLORS)]
        self.traces.append(dict(label=label, raw=copy(raw), norm=copy(norm), color=color, on=True))
        it = QtWidgets.QListWidgetItem(label)
        it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
        it.setCheckState(QtCore.Qt.Checked)
        it.setForeground(pg.mkColor(color))
        self.lst.blockSignals(True)
        self.lst.addItem(it)
        self.lst.blockSignals(False)
        self.draw()

    def _toggled(self, it):
        i = self.lst.row(it)
        if 0 <= i < len(self.traces):
            self.traces[i]["on"] = it.checkState() == QtCore.Qt.Checked
            self.draw()

    def remove_selected(self):
        for it in sorted(self.lst.selectedItems(), key=self.lst.row, reverse=True):
            i = self.lst.row(it)
            self.lst.takeItem(i)
            del self.traces[i]
        self.draw()

    def clear(self):
        self.lst.clear()
        self.traces.clear()
        self.draw()

    # ------------------------------------------------------------ drawing
    def _region_moved(self):
        a, b = self.region.getRegion()
        for sp, val in ((self.sp_t1, a), (self.sp_t2, b)):
            sp.blockSignals(True)
            sp.setValue(val)
            sp.blockSignals(False)
        self.draw()

    def curves(self):
        """[(label, t, y, e, color)] as plotted (after the chosen scaling)."""
        mode = self.cmb_mode.currentIndex()
        kind = "norm" if self.cmb_series.currentText() == "normalised" else "raw"
        rng = (self.sp_t1.value(), self.sp_t2.value())
        out = []
        for tr in self.traces:
            if not tr["on"]:
                continue
            t, y, e = tr[kind]
            y2, e2, _ = an.normalise_trace(t, y, e, mode, rng)
            out.append((tr["label"], t, y2, e2, tr["color"]))
        return out

    def draw(self):
        p = self.plot
        p.clear()
        mode = self.cmb_mode.currentIndex()
        bad = []
        for label, t, y, e, color in self.curves():
            if len(t) == 0:
                continue
            if not np.isfinite(y).any():
                bad.append(label)
                continue
            p.addItem(theme.error_bars(t, y, e, color))
            k = np.argsort(t)
            p.plot(np.asarray(t)[k], np.asarray(y)[k], pen=pg.mkPen(color, width=1.5), symbol="o", symbolSize=6,
                   symbolBrush=color, symbolPen=None, name=label)
        if mode == 2:
            self.region.blockSignals(True)
            self.region.setRegion((self.sp_t1.value(), self.sp_t2.value()))
            self.region.blockSignals(False)
            p.addItem(self.region)
        ylab = ["signal", "signal (0 to 1)", "signal / peak in range"][mode]
        p.setLabel("left", ylab)
        self.lbl_info.setText(("no point with a positive maximum in the peak range: " + "; ".join(bad))
                              if bad else ("drag the shaded band or type the range" if mode == 2 else ""))
