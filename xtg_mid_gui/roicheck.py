"""ROI position check: centre of mass or brightest pixel inside a ROI, shot by shot
or on the mean image of each run.  A separate window, opened from the header."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from . import analysis as an
from . import theme
from .widgets import ShrinkCombo, fit_to_screen


class RoiPositionWindow(QtWidgets.QWidget):
    closed = QtCore.Signal()

    def __init__(self, main):
        super().__init__(main, QtCore.Qt.Window)
        self.main = main
        self.setObjectName("root")                # dark / light theme as the main window
        self.setWindowTitle("XTG @ MID - ROI position check")
        fit_to_screen(self, 1250, 780)
        self._cache = {}
        v = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()

        def lab(t):
            w = QtWidgets.QLabel(t, objectName="muted")
            row.addWidget(w)

        lab("ROI:")
        self.cmb_roi = ShrinkCombo(220, 120)
        row.addWidget(self.cmb_roi)
        lab("   position:")
        self.cmb_mode = QtWidgets.QComboBox()
        self.cmb_mode.addItems(["centre of mass", "max position"])
        self.cmb_mode.setToolTip("centre of mass of (counts - background) inside the ROI, or the brightest pixel")
        row.addWidget(self.cmb_mode)
        lab("   on:")
        self.cmb_per = QtWidgets.QComboBox()
        self.cmb_per.addItems(["shot by shot", "mean of each run"])
        self.cmb_per.setToolTip("every kept shot, or the mean image of each run's kept shots (one point per run)")
        row.addWidget(self.cmb_per)
        self.cb_bkg = QtWidgets.QCheckBox("subtract background ROI")
        self.cb_bkg.setChecked(True)
        row.addWidget(self.cb_bkg)
        row.addStretch(1)
        v.addLayout(row)
        self.lbl = QtWidgets.QLabel("", objectName="muted")
        v.addWidget(self.lbl)
        for w in self.findChildren(QtWidgets.QComboBox):
            w.setView(QtWidgets.QListView())
        g = theme.register_widget(pg.GraphicsLayoutWidget())
        self.p_x = theme.style_plot(g.addPlot(row=0, col=0), "x (column)", "shot", "x (px)")
        self.p_y = theme.style_plot(g.addPlot(row=1, col=0), "y (row)", "shot", "y (px)")
        self.p_xy = theme.style_plot(g.addPlot(row=0, col=1, rowspan=2), "y vs x (colour = delay)", "x (px)", "y (px)")
        self.p_xy.invertY(True)                   # as on the detector image
        self.p_y.setXLink(self.p_x)
        g.ci.layout.setColumnStretchFactor(0, 3)
        g.ci.layout.setColumnStretchFactor(1, 2)
        self.glw = g
        v.addWidget(g, 1)
        for w in (self.cmb_roi, self.cmb_mode, self.cmb_per):
            w.currentTextChanged.connect(lambda *_: self.refresh())
        self.cb_bkg.toggled.connect(lambda *_: self.refresh())

    def closeEvent(self, ev):
        self.closed.emit()
        super().closeEvent(ev)

    # ------------------------------------------------------------ data
    def keys(self):
        """All enabled ROIs (signal ones), as on the other pages."""
        return self.main.signal_keys()

    def positions(self, det, s, roi, mode, per, bkg_on):
        m = self.main
        bkg = tuple(m.views[det].panel.background_geoms()) if bkg_on else ()
        key = (det, s, roi.geometry(), bkg, mode, per, m.keep[s].tobytes())
        if key not in self._cache:
            if len(self._cache) > 500:
                self._cache.clear()
            imgs = m.scans[s].images[det][m.keep[s]]
            if per == "run":
                with np.errstate(invalid="ignore"):
                    imgs = np.nanmean(imgs, axis=0)[None] if len(imgs) else imgs
            lvl = an.roi_bkg_level(imgs, bkg) if bkg else None
            self._cache[key] = an.roi_position(imgs, roi.geometry(), mode, lvl)
        return self._cache[key]

    # ------------------------------------------------------------ drawing
    def refresh(self):
        if not self.isVisible():
            return
        m = self.main
        keys = self.keys()
        cur = self.cmb_roi.currentText()
        self.cmb_roi.blockSignals(True)
        self.cmb_roi.clear()
        self.cmb_roi.addItems([k[0] for k in keys])
        self.cmb_roi.setCurrentIndex(max(self.cmb_roi.findText(cur), 0))
        self.cmb_roi.blockSignals(False)
        for p in (self.p_x, self.p_y, self.p_xy):
            p.clear()
        scans = m.sorted_scans()
        sel = next((k for k in keys if k[0] == self.cmb_roi.currentText()), None)
        if sel is None or not scans:
            self.lbl.setText("Load scans and switch on a ROI.")
            return
        label, det, roi = sel
        mode = "max" if self.cmb_mode.currentText().startswith("max") else "com"
        per = "run" if self.cmb_per.currentText().startswith("mean") else "shot"
        xs, ys, ts, bounds = [], [], [], []
        for s in scans:
            x, y = self.positions(det, s, roi, mode, per, self.cb_bkg.isChecked())
            xs.append(x)
            ys.append(y)
            ts.append(np.full(len(x), m.scans[s].delay))
            bounds.append(len(x))
        x, y, t = np.concatenate(xs), np.concatenate(ys), np.concatenate(ts)
        fin = np.isfinite(t)
        lo, hi = (t[fin].min(), t[fin].max()) if fin.any() else (0, 1)
        u = np.where(fin, (t - lo) / max(hi - lo, 1e-9), 0.5)
        cols = pg.colormap.get("viridis").map(u, mode="byte")
        cache = {}
        brushes = [cache.setdefault(c, pg.mkBrush(*c)) for c in map(tuple, cols.tolist())]
        n = np.arange(len(x))
        cx, cy = roi.geometry()[:2]
        what = "centre of mass" if mode == "com" else "max position"
        xl = "shot (kept, runs in delay order)" if per == "shot" else "run (delay order)"
        for p, val, ref, name in ((self.p_x, x, cx, "x"), (self.p_y, y, cy, "y")):
            p._xtg["x"] = xl
            ok = np.isfinite(val)
            p.addItem(pg.ScatterPlotItem(n[ok], val[ok], size=5 if per == "shot" else 9, pen=None,
                                         brush=[b for b, k in zip(brushes, ok) if k]))
            if per == "run" and ok.sum() > 1:
                p.plot(n[ok], val[ok], pen=pg.mkPen(theme.neutral(120), width=1))
            p.addItem(pg.InfiniteLine(ref, angle=0, pen=pg.mkPen(roi.color, width=1.5, style=QtCore.Qt.DashLine)))
            theme.set_title(p, f"{name} of the {what}" + (f"  ·  {np.nanmean(val):.2f} ± {np.nanstd(val):.2f} px"
                                                         if ok.any() else ""))
        if per == "run":
            ticks = [(i, f"{s}\n{m.scans[s].delay:g} fs") for i, s in enumerate(scans)]
            for p in (self.p_x, self.p_y):
                p.getAxis("bottom").setTicks([ticks])
        else:
            for p in (self.p_x, self.p_y):
                p.getAxis("bottom").setTicks(None)
            start = 0
            for b in bounds[:-1]:
                start += b
                for p in (self.p_x, self.p_y):
                    p.addItem(pg.InfiniteLine(start - 0.5, angle=90,
                                              pen=pg.mkPen(theme.neutral(90), style=QtCore.Qt.DashLine)))
        ok = np.isfinite(x) & np.isfinite(y)
        self.p_xy.addItem(pg.ScatterPlotItem(x[ok], y[ok], size=5 if per == "shot" else 9, pen=None,
                                             brush=[b for b, k in zip(brushes, ok) if k]))
        self.p_xy.plot([cx], [cy], pen=None, symbol="+", symbolSize=16, symbolPen=pg.mkPen(roi.color, width=2))
        self.lbl.setText(f"{label}: {what} of the ROI pixels{' minus the background ROI level' if self.cb_bkg.isChecked() else ''}"
                         f", {'every kept shot' if per == 'shot' else 'the mean image of each run'}; "
                         f"{len(scans)} runs, {int(ok.sum())} points. Dashed: ROI centre "
                         f"({cx:.1f}, {cy:.1f}). Colour = delay (viridis).")
        for p in (self.p_x, self.p_y, self.p_xy):
            p.enableAutoRange()
        if per == "run":                          # room for the first / last run label
            self.p_x.setXRange(-0.5, len(scans) - 0.5, padding=0.02)
