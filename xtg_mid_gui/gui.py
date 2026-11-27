"""XTG analysis GUI for MID: two Jungfraus, pump/probe diamond diodes, one delay
per scan, upstream spectrometers, CRL pointing.  PyQt5 + pyqtgraph.

    python -m xtg_mid_gui --backend simulated --scans 101-149   # fake data for testing
    python -m xtg_mid_gui --proposal 11456 --scans 20-40       # real runs (default backend: extra_data)
"""
from __future__ import annotations

import csv
import os
import sys
import time
import warnings

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from . import agipd
from . import delays as delay_file
from . import analysis as an
from . import theme
from .backends import ScanData, load_many, make_backend, parse_scans
from .config import PARAM_PRESETS, PRESET_DIR, PRESETS, Source, SourceConfig, config_for, mask_key_for
from .fitpage import FitPage
from .roicheck import RoiPositionWindow
from .rois import RoiPanel
from .widgets import ShrinkCombo, elastic, fit_to_screen

ALL_SCANS = "average of all used scans"


# ---------------------------------------------------------------- small widgets
class Card(QtWidgets.QFrame):
    def __init__(self, title):
        super().__init__()
        self.setObjectName("card")
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(12, 10, 12, 12)
        v.setSpacing(8)
        t = QtWidgets.QLabel(title.upper())
        t.setObjectName("cardTitle")
        v.addWidget(t)
        self.body = v

    def form(self):
        f = QtWidgets.QFormLayout()
        f.setHorizontalSpacing(10)
        f.setVerticalSpacing(6)
        self.body.addLayout(f)
        return f


def _label(text="", name=None, wrap=True):
    lab = QtWidgets.QLabel(text)
    if name:
        lab.setObjectName(name)
    lab.setWordWrap(wrap)
    return lab


def _spin(lo, hi, val, step, dec=2):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    return s


def _glw():
    return theme.register_widget(pg.GraphicsLayoutWidget())


# ---------------------------------------------------------------- detector view
class ImagePane(QtWidgets.QWidget):
    """Title bar + image + colour bar, with colour map, log scale and full-page button."""

    def __init__(self, label, full_tip="Show only this detector, using the whole window (ROIs and settings are kept)"):
        super().__init__()
        v = self.vbox = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        top = self.top = QtWidgets.QHBoxLayout()
        t = _label(label, "appTitle", wrap=False)
        t.setStyleSheet("font-size: 14px;")
        top.addWidget(t)
        top.addStretch(1)
        top.addWidget(_label("image:", "muted", wrap=False))
        self.source = ShrinkCombo(220, 100)
        top.addWidget(self.source)
        self.b_full = QtWidgets.QPushButton("⤢  Full page")
        self.b_full.setToolTip(full_tip)
        top.addWidget(self.b_full)
        v.addLayout(top)
        self.glw = _glw()
        self.plot = theme.style_plot(self.glw.addPlot(), x="column (px)", y="row (px)")
        self.plot.invertY(True)
        self.plot.setAspectLocked(True)
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        self.lut = pg.HistogramLUTItem(self.img)
        self.set_cmap(theme.DEFAULT_CMAP[theme.mode()])
        self.glw.addItem(self.lut)
        v.addWidget(self.glw, 3)
        self._first = True
        self._raw = None
        self._labels = []

    def set_cmap(self, name):
        self.lut.gradient.setColorMap(theme.colormap(name))   # also recolours the image

    log_scale = False

    def set_log(self, on):
        """Display only: log10 of the image (ROI sums always use linear counts)."""
        if bool(on) != self.log_scale:
            self.log_scale = bool(on)
            self._relevel = True
            if self._raw is not None:
                self.show_image(self._raw)

    @staticmethod
    def log_image(im):
        """log10(counts); values <= 0 are clipped to 1e-3 x the image maximum."""
        im = np.asarray(im, np.float32)
        top = np.nanmax(np.abs(im)) if np.isfinite(im).any() else 1.0
        return np.log10(np.clip(im, max(top, 1e-12) * 1e-3, None))

    def show_image(self, im):
        if im is None:
            self._raw = None
            self.img.clear()
            return
        self._raw = im
        shown = self.log_image(im) if self.log_scale else im
        relevel = self._first or getattr(self, "_relevel", False)
        self.img.setImage(shown, autoLevels=False)
        if relevel and np.isfinite(shown).any():
            lo, hi = np.nanpercentile(shown, [1, 99.7])
            self.img.setLevels((lo, hi))
            self.lut.setLevels(lo, hi)
            self._relevel = False
        if self._first:
            self.plot.autoRange()
        self._after_image(im)
        self._first = False

    def _after_image(self, im):
        pass

    def set_labels(self, items):
        """items = [(col, row, text)]: tile numbers drawn on the image."""
        for it in self._labels:
            self.plot.removeItem(it)
        self._labels = []
        for x, y, txt in items:
            it = pg.TextItem(txt, color=(255, 255, 255), anchor=(0, 0), fill=pg.mkBrush(0, 0, 0, 140))
            it.setPos(x + 2, y + 2)
            it.setZValue(20)
            self.plot.addItem(it)
            self._labels.append(it)


class DetectorView(ImagePane):
    """Scan-average image of one detector (Jungfrau, or the stacked AGIPD tiles) with
    its editable ROIs."""

    def __init__(self, det, label, on_change):
        super().__init__(label)
        self.det = det
        self.vbox.addWidget(_label("Drag a ROI to move it, drag its handles to resize / rotate. Double-click a "
                                   "name to rename it; click 'shape' to switch that ROI between ellipse and "
                                   "rectangle.", "muted"))
        self.panel = RoiPanel(det, label, self.plot.getViewBox())
        self.panel.changed.connect(on_change)
        self.vbox.addWidget(self.panel, 1)

    def _after_image(self, im):
        if self.panel.shape != im.shape or self._first:
            self.panel.shape = im.shape
            if not self.panel.rois:
                self.panel.defaults()


class MosaicView(ImagePane):
    """All AGIPD tiles (mean image per scan), laid out as on the detector and numbered.
    Click tiles to choose the ones analysed (with ROIs) in the second view."""
    tiles_applied = QtCore.Signal(list)

    def __init__(self, label, selected):
        super().__init__(label, "Show only the all-tiles view, using the whole window")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(_label("tiles to analyse:", "muted", wrap=False))
        self.ed_tiles = QtWidgets.QLineEdit(" ".join(map(str, selected)))
        self.ed_tiles.setToolTip("AGIPD tile numbers, e.g. '8 10 15' or '8-11'. Click a tile on the image to "
                                 "add / remove it, then Apply (re-reads the loaded scans).")
        self.ed_tiles.textChanged.connect(lambda *_: self._mark())
        row.addWidget(self.ed_tiles, 1)
        self.b_apply = QtWidgets.QPushButton("Apply")
        self.b_apply.clicked.connect(lambda: self.tiles_applied.emit(self.chosen()))
        row.addWidget(self.b_apply)
        self.vbox.addLayout(row)
        self.vbox.addWidget(_label("Mean of each tile per scan (no ROIs here). Click a tile to add or remove it from "
                                   "the analysed tiles, then Apply.", "muted"))
        self._rects = []
        self.plot.scene().sigMouseClicked.connect(self._clicked)
        self.set_labels([(c0, r0, str(t)) for t, (r0, c0) in agipd.module_origins().items()])
        self._mark()

    def chosen(self):
        try:
            return agipd.parse_tiles(self.ed_tiles.text())
        except ValueError:
            return []

    def toggle_tile(self, t):
        cur = self.chosen()
        cur = sorted(set(cur) ^ {t})
        self.ed_tiles.setText(" ".join(map(str, cur)))

    def _clicked(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            return
        p = self.plot.getViewBox().mapSceneToView(ev.scenePos())
        t = agipd.module_at(p.y(), p.x())
        if t is not None:
            self.toggle_tile(t)

    def _mark(self):
        for r in self._rects:
            self.plot.removeItem(r)
        self._rects = []
        mh, mw = agipd.module_size()
        for t in self.chosen():
            r0, c0 = agipd.module_origins()[t]
            rect = QtWidgets.QGraphicsRectItem(c0 - 1, r0 - 1, mw + 2, mh + 2)
            rect.setPen(pg.mkPen(theme.PUMP, width=2.5))
            rect.setZValue(15)
            self.plot.addItem(rect)
            self._rects.append(rect)


# ---------------------------------------------------------------- main window
class XTGMidWindow(QtWidgets.QMainWindow):
    last = None          # the window after a proposal switch (keeps it alive)

    def __init__(self, cfg: SourceConfig | None = None, backend="simulated", sim_kw=None):
        super().__init__()
        self.cfg = cfg or SourceConfig()
        self.backend_kind = backend
        self.sim_kw = sim_kw or {}
        self.backend = make_backend(backend, self.cfg, **(self.sim_kw if backend == "simulated" else {}))
        self.scans: dict[int, ScanData] = {}
        self._stale: set = set()
        self.use: dict[int, bool] = {}
        self.keep: dict[int, np.ndarray] = {}
        self._cache: dict = {}
        self.series: dict = {}
        self.setWindowTitle("XTG @ MID - scan analysis")
        fit_to_screen(self, 1560, 960)
        self._timer = QtCore.QTimer(self, singleShot=True, interval=120)
        self._timer.timeout.connect(self.recompute)
        self._build()
        for cb in self.findChildren(QtWidgets.QComboBox):   # macOS: styled popups need a list view to show items
            cb.setView(QtWidgets.QListView())
        self._update_placeholder_pill()

    # ============================================================ layout
    def _build(self):
        root = QtWidgets.QWidget(objectName="root")
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(10, 8, 10, 6)
        outer.addLayout(self._build_header())
        split = self.main_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.sidebar = self._build_sidebar()
        split.addWidget(self.sidebar)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_detectors(), "Detectors && ROIs")
        self.tabs.addTab(self._build_delay(), "Delay series")
        self.tabs.addTab(self._build_corr(), "Correlations")
        self.tabs.addTab(self._build_spec(), "Spectrometers && XGM" if self.spec_mode() == "monitors" else
                         ("Upstream spectrometer" if len(self.cfg.spectrometers) == 1 else "Upstream spectrometers"))
        self.spec_tab = 3
        self.tabs.addTab(self._build_crl(), "CRL pointing")
        self.crl_tab = 4
        self.fit_page = FitPage(self)
        self.tabs.addTab(self.fit_page, "Fitting")
        self.fit_tab = 5
        self.tabs.addTab(self._build_param(), "Parameter scan")
        self.param_tab = 6
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        split.setSizes([280, 1280])
        outer.addWidget(split, 1)
        self.setStatusBar(QtWidgets.QStatusBar())
        self.set_crl_visible(self.cfg.crl_enabled)
        self.set_spec_visible(self.cfg.spec_enabled)
        self.set_param_visible(self.cfg.param_enabled)

    def _build_header(self):
        h = QtWidgets.QHBoxLayout()
        h.addWidget(_label("XTG @ MID", "appTitle", wrap=False))
        if self.cfg.overview_detector()[1] is not None:
            what = f"p{self.cfg.proposal} · AGIPD tiles · HIREX / DES · XGM"
        else:
            nd, ns = len(self.cfg.image_detectors()), len(self.cfg.spectrometers)
            what = f"p{self.cfg.proposal or '?'} · {nd} detectors · diamond diodes · {ns} spectrometer{'s' * (ns != 1)}"
        h.addWidget(elastic(_label("  " + what, "muted", wrap=False)), 1)
        self.pills = {k: _label("", "pill", wrap=False) for k in ("scans", "delays", "shots", "memory", "backend")}
        for p in self.pills.values():
            h.addWidget(p)
        self.pill_ph = _label("", "tbc", wrap=False)
        h.addWidget(self.pill_ph)
        self.b_crl = QtWidgets.QPushButton("CRL page")
        self.b_crl.setCheckable(True)
        self.b_crl.setToolTip("Show / hide the CRL beam-pointing page")
        self.b_crl.toggled.connect(self.set_crl_visible)
        self.b_spec = QtWidgets.QPushButton("Spectrometers")
        self.b_spec.setCheckable(True)
        self.b_spec.setToolTip("Show / hide the upstream spectrometer page")
        self.b_spec.toggled.connect(self.set_spec_visible)
        self.b_param = QtWidgets.QPushButton("Param scan")
        self.b_param.setCheckable(True)
        self.b_param.setToolTip("Show / hide the page that plots the ROI signal against a scanned parameter\n"
                                "(sample rotation, grating-sample distance, temperature ...), read from the\n"
                                "source you type there; works within one run and across several runs")
        self.b_param.toggled.connect(self.set_param_visible)
        h.addWidget(self.b_param)
        self.b_pos = QtWidgets.QPushButton("ROI position")
        self.b_pos.setCheckable(True)
        self.b_pos.setToolTip("Open / close a window that checks where the intensity sits inside a ROI:\n"
                              "centre of mass or brightest pixel (x and y), shot by shot or per run")
        self.b_pos.toggled.connect(self.set_pos_window)
        h.addWidget(self.b_pos)
        h.addWidget(self.b_spec)
        h.addWidget(self.b_crl)
        self.b_theme = QtWidgets.QPushButton()
        self.b_theme.clicked.connect(self.toggle_theme)
        h.addWidget(self.b_theme)
        self._theme_text()
        return h

    def _build_sidebar(self):
        side = QtWidgets.QWidget(objectName="sidebar")
        v = QtWidgets.QVBoxLayout(side)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(10)

        c = Card("Data")
        f = c.form()
        self.cmb_backend = ShrinkCombo(120, 70)
        self.cmb_backend.addItems(["simulated", "extra_data"])
        self.cmb_backend.setCurrentText(self.backend_kind)
        self.cmb_backend.currentTextChanged.connect(self._backend_changed)
        f.addRow("backend", self.cmb_backend)
        prow = QtWidgets.QHBoxLayout()
        self.cmb_preset = ShrinkCombo(140, 70)
        for k, txt in PRESETS.items():
            self.cmb_preset.addItem(txt, k)
        self.cmb_preset.setCurrentIndex(max(self.cmb_preset.findData(self.cfg.preset), 0))
        self.cmb_preset.setToolTip("7016: AGIPD tiles, HIREX / DES spectrometers, XGM (Sources_7016.yaml)\n"
                                   "11456: Jungfraus, diamond diodes, one upstream spectrometer (Sources_11456.yaml)\n"
                                   "custom: any proposal number, sources from a YAML file you choose")
        self.cmb_preset.activated.connect(self._preset_chosen)
        prow.addWidget(self.cmb_preset, 1)
        self.sp_prop = QtWidgets.QSpinBox()
        self.sp_prop.setRange(0, 999999)
        self.sp_prop.setValue(self.cfg.proposal)
        self.sp_prop.setEnabled(self.cfg.preset == "custom")
        self.sp_prop.setToolTip("proposal number (editable for 'custom')")
        prow.addWidget(self.sp_prop)
        f.addRow("proposal", prow)
        src = getattr(self.cfg, "_source_file", "") or "built-in defaults"
        self.lbl_cfgfile = elastic(_label(f"sources: {os.path.basename(str(src)) or src}", "muted", wrap=False))
        self.lbl_cfgfile.setToolTip(str(src))
        f.addRow(self.lbl_cfgfile)
        sims = getattr(self.backend, "scans", lambda: [])()
        self.ed_scans = QtWidgets.QLineEdit(f"{min(sims)}-{max(sims)}" if sims else "")
        self.ed_scans.setToolTip("run numbers in any order, with gaps and ranges, e.g. '92, 96, 94, 95' or '95 96 98-99'")
        f.addRow("scans", self.ed_scans)
        row = QtWidgets.QHBoxLayout()
        b_load = QtWidgets.QPushButton("Load", objectName="primary")
        b_load.setToolTip("replace the loaded scans")
        b_add = QtWidgets.QPushButton("Add")
        b_add.setToolTip("append these scans to the delay series")
        b_src = QtWidgets.QPushButton("Sources")
        b_load.clicked.connect(lambda: self.load(parse_scans(self.ed_scans.text()), append=False))
        b_add.clicked.connect(lambda: self.load(parse_scans(self.ed_scans.text()), append=True))
        b_src.clicked.connect(self._show_sources)
        for b in (b_load, b_add, b_src):
            row.addWidget(b)
        c.body.addLayout(row)
        self.cb_mask = QtWidgets.QCheckBox("bad-pixel masks")
        self.cb_mask.setChecked(bool(getattr(self.cfg, "apply_mask", True)))
        self.cb_mask.setToolTip("Set pixels flagged in the detector / spectrometer mask (e.g. AGIPD image.mask,\n"
                                "HIREX data.mask; mask > 0) to NaN so they are left out of every sum and mean.\n"
                                "Config: apply_mask, and mask_key per detector / spectrometer.\n"
                                "Changing it reads the loaded scans again.")
        self.cb_mask.toggled.connect(self._mask_toggled)
        c.body.addWidget(self.cb_mask)
        self.cb_mask_pt = QtWidgets.QCheckBox("mask every frame (slow)")
        self.cb_mask_pt.setChecked(bool(getattr(self.cfg, "mask_per_train", False)))
        self.cb_mask_pt.setToolTip("Off (fast): the mask is read once per run, from its first trains, and a pixel\n"
                                   "flagged in more than half of those frames is left out everywhere.\n"
                                   "On: the mask of every frame is read and applied (about twice the reading time).\n"
                                   "Changing it reads the loaded scans again.")
        self.cb_mask_pt.toggled.connect(self._mask_pt_toggled)
        c.body.addWidget(self.cb_mask_pt)
        b_mask = QtWidgets.QPushButton("Re-read masks")
        b_mask.setToolTip("The masks are read once for the experiment and kept in masks_<proposal>.npz in\n"
                          "the start folder. This deletes that file and reads the loaded scans again.")
        b_mask.clicked.connect(self._mask_reread)
        c.body.addWidget(b_mask)
        self.cb_phot = QtWidgets.QCheckBox("photonize Jungfraus")
        self.cb_phot.setChecked(bool(getattr(self.cfg, "photonize", False)))
        self.cb_phot.setToolTip("Per frame, before averaging: values < 0 -> NaN, the others -> round(value / photon energy),\n"
                                "so the Jungfrau images and ROI sums are in photons (corrected data must be in keV).\n"
                                "Photon energy: the undulator setting (config photon_energy, default\n"
                                "MID_XTD1_UND/DOOCS/ENERGY / actualPosition.value), or photon_energy_keV if > 0.\n"
                                "Changing it reads the loaded scans again.")
        self.cb_phot.toggled.connect(self._photonize_toggled)
        c.body.addWidget(self.cb_phot)
        v.addWidget(c)

        c = Card("Scans")
        orow = QtWidgets.QHBoxLayout()
        orow.addWidget(_label("order by", "muted", wrap=False))
        self.cmb_order = ShrinkCombo(120, 70)
        self.cmb_order.addItems(["delay", "run number", "as typed"])
        self.cmb_order.setToolTip("order of the rows in this table (the delay series is always sorted by delay)")
        self.cmb_order.currentTextChanged.connect(lambda *_: self._fill_scan_table())
        orow.addWidget(self.cmb_order, 1)
        c.body.addLayout(orow)
        rrow = QtWidgets.QHBoxLayout()
        rrow.addWidget(_label("reference", "muted", wrap=False))
        self.cmb_ref = ShrinkCombo(120, 70)
        self.cmb_ref.setToolTip("The only run whose images are all kept in memory: place the ROIs on it.\n"
                                "The other runs are read in pieces and only their numbers are kept\n"
                                "(ROI signals, peak positions, run average image), a few MB per run.")
        self.cmb_ref.activated.connect(lambda *_: self.set_reference(self.cmb_ref.currentData()))
        rrow.addWidget(self.cmb_ref, 1)
        c.body.addLayout(rrow)
        self.b_recalc = QtWidgets.QPushButton("", objectName="primary")
        self.b_recalc.setToolTip("The ROIs changed: read the other runs again to get their numbers for the new ROIs")
        self.b_recalc.clicked.connect(self.recalc_stale)
        self.b_recalc.setVisible(False)
        c.body.addWidget(self.b_recalc)
        self.tbl = QtWidgets.QTableWidget(0, 5)
        self.tbl.setHorizontalHeaderLabels(["", "run", "delay fs", "shots", "kept"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.tbl.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.tbl.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.tbl.setMinimumHeight(230)
        self.tbl.itemChanged.connect(self._scan_item_changed)
        c.body.addWidget(self.tbl)
        self.lbl_delays = _label("", "muted")
        c.body.addWidget(self.lbl_delays)
        self._update_delay_label()
        v.addWidget(c)

        c = Card("Shots & normalisation")
        f = c.form()
        self.sp_mad = _spin(0, 50, 5, 0.5, 1)
        self.sp_mad.setToolTip("per-scan median/MAD cut on both diodes (0 = off)")
        self.sp_min = _spin(0, 1, 0.1, 0.05, 2)
        self.sp_min.setToolTip("drop shots with a diode below this fraction of its median (X-rays off)")
        self.cmb_norm = ShrinkCombo(140, 80)
        self.cmb_norm.addItems(list(an.NORMALISERS))
        self.cmb_norm.setCurrentText("pump^2 x probe")
        self.cb_pos = QtWidgets.QCheckBox("drop diode ≤ 0")
        self.cb_pos.setChecked(True)
        self.cb_pos.setToolTip("Untick for monitors that can be negative (offset, negative-going signal,\n"
                               "spectrometer integral after baseline subtraction)")
        self.cb_pos.toggled.connect(self._schedule)
        f.addRow("MAD cut σ", self.sp_mad)
        f.addRow("min / median", self.sp_min)
        f.addRow("", self.cb_pos)
        f.addRow("normalise", self.cmb_norm)
        self.lbl_cuts = _label("", "muted")
        self.lbl_cuts.setToolTip("A cut that would remove every shot of a scan is skipped for that scan")
        f.addRow(self.lbl_cuts)
        for w in (self.sp_mad, self.sp_min):
            w.valueChanged.connect(self._schedule)
        self.cmb_norm.currentTextChanged.connect(self._schedule)
        b_csv = QtWidgets.QPushButton("Export CSV")
        b_csv.clicked.connect(self._export)
        c.body.addWidget(b_csv)
        v.addWidget(c)

        c = Card("Log")
        self.log_box = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log_box.setMaximumHeight(130)
        c.body.addWidget(self.log_box)
        v.addWidget(c)
        v.addStretch(1)
        sc = QtWidgets.QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(side)
        sc.setMinimumWidth(262)
        sc.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        return sc

    def _build_detectors(self):
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(4, 4, 4, 0)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(_label("colour map:", "muted", wrap=False))
        self.cmb_cmap = QtWidgets.QComboBox()
        self.cmb_cmap.addItems(list(theme.COLORMAPS))
        self.cmb_cmap.setCurrentText(theme.DEFAULT_CMAP[theme.mode()])
        self.cmb_cmap.setToolTip("'white …' maps put low counts at white, handy in light mode")
        self.cmb_cmap.currentTextChanged.connect(self.set_cmap)
        bar.addWidget(self.cmb_cmap)
        self.chk_log = QtWidgets.QCheckBox("log scale")
        self.chk_log.setToolTip("show log10(counts) on both detectors (display only: ROI sums stay linear)")
        self.chk_log.toggled.connect(self.set_log)
        bar.addSpacing(16)
        bar.addWidget(self.chk_log)
        bar.addStretch(1)
        v.addLayout(bar)
        w = self.det_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.views = {}
        self._full = None
        self.mosaic = None
        ov_name, ov = self.cfg.overview_detector()
        if ov is not None:                      # p7016: all AGIPD tiles + the analysed subset
            sub = next((d for d in self.cfg.detectors.values() if d.kind == "agipd_tiles"), None)
            self.mosaic = MosaicView(ov.label, sub.tiles if sub else [])
            self.mosaic.source.currentIndexChanged.connect(lambda *_: self._draw_mosaic())
            self.mosaic.b_full.clicked.connect(lambda *_: self.toggle_full(ov_name))
            self.mosaic.tiles_applied.connect(self.set_tiles)
            w.addWidget(self.mosaic)
        for det, d in self.cfg.image_detectors().items():
            view = DetectorView(det, d.label, self._schedule)
            view.source.currentIndexChanged.connect(lambda *_, det=det: self._draw_image(det))
            view.b_full.clicked.connect(lambda *_, det=det: self.toggle_full(det))
            self.views[det] = view
            w.addWidget(view)
        v.addWidget(w, 1)
        return page

    def set_log(self, on):
        if self.chk_log.isChecked() != bool(on):
            self.chk_log.setChecked(bool(on))
            return
        for view in self.panes().values():
            view.set_log(on)

    def set_cmap(self, name):
        if self.cmb_cmap.currentText() != name:
            self.cmb_cmap.setCurrentText(name)
            return
        for view in self.panes().values():
            view.set_cmap(name)

    def panes(self) -> dict:
        """Every image pane on the Detectors page (the AGIPD all-tiles view included)."""
        out = {}
        if self.mosaic is not None:
            out[self.cfg.overview_detector()[0]] = self.mosaic
        out.update(self.views)
        return out

    def set_tiles(self, tiles):
        """New AGIPD tile subset: re-read the loaded scans with these tiles."""
        tiles = sorted(set(tiles))
        if not tiles:
            self.log("choose at least one AGIPD tile")
            return
        for name, d in self.cfg.detectors.items():
            if d.kind == "agipd_tiles":
                self._move_rois_with_tiles(name, d.tiles, tiles)
                d.tiles = list(tiles)
        self.log(f"AGIPD tiles to analyse: {' '.join(map(str, tiles))}")
        if self.scans:
            self.reload_scans()

    def _move_rois_with_tiles(self, det, old_tiles, new_tiles):
        """Keep each ROI on the same AGIPD tile when the stack of analysed tiles changes."""
        if det not in self.views:
            return
        old_o, new_o = agipd.stack_origins(old_tiles), agipd.stack_origins(new_tiles)
        mh = agipd.TILE_SHAPE[1]
        for roi in self.views[det].panel.rois:
            cy = roi.geometry()[1]
            t = next((t for t, (r0, _) in old_o.items() if r0 <= cy < r0 + mh), None)
            if t is not None and t in new_o:
                dy = new_o[t][0] - old_o[t][0]
                if dy:
                    roi.item.setPos(roi.item.pos() + QtCore.QPointF(0, dy), finish=False)

    def toggle_full(self, det):
        """Full page for one Jungfrau.  Going back restores the layout and the zoom of
        every detector; ROI edits, levels and colour map made meanwhile are kept."""
        if self._full is None:
            self._full = dict(det=det, main=self.main_split.sizes(), dets=self.det_split.sizes(),
                              ranges={d: v.plot.getViewBox().viewRange() for d, v in self.panes().items()})
            self.sidebar.hide()
            for d, view in self.panes().items():
                view.setVisible(d == det)
            self.panes()[det].b_full.setText("⤡  Back")
        else:
            st, self._full = self._full, None
            self.sidebar.show()
            for d, view in self.panes().items():
                view.show()
                view.b_full.setText("⤢  Full page")
            self.main_split.setSizes(st["main"])
            self.det_split.setSizes(st["dets"])
            for d, (xr, yr) in st["ranges"].items():
                self.panes()[d].plot.getViewBox().setRange(xRange=xr, yRange=yr, padding=0)

    @property
    def full_page(self):
        return None if self._full is None else self._full["det"]

    def _build_delay(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(_label("background:", "muted", wrap=False))
        self.cmb_bkg = ShrinkCombo(230, 120)
        self.cmb_bkg.setToolTip("background ROI: signal = ROI sum minus the ROI's pixels x the mean background-ROI pixel.\n"
                                "none: plain ROI sum.\n"
                                "run …: every shot's plain ROI sum minus the mean ROI sum (kept shots) of that run,\n"
                                "e.g. a run before time zero. Normalisation is applied after the subtraction.\n"
                                "Applies to every page (delay series, correlations, parameter scan, fit).")
        self.cmb_bkg.activated.connect(lambda *_: self._bkg_changed())
        row.addWidget(self.cmb_bkg)
        row.addWidget(_label("   ROI value:", "muted", wrap=False))
        self.cmb_value = QtWidgets.QComboBox()
        self.cmb_value.addItems(["sum", "mean per pixel"])
        self.cmb_value.setToolTip("sum: all ROI pixels added up (bigger ROI = bigger number).\n"
                                  "mean per pixel: that sum divided by the ROI's pixel count (masked pixels\n"
                                  "not counted), so ROIs of different size can be compared directly.\n"
                                  "Applies to every page.")
        self.cmb_value.activated.connect(lambda *_: (self.log(f"ROI value: {self.cmb_value.currentText()}"),
                                                     self.recompute()))
        row.addWidget(self.cmb_value)
        self.lbl_bkgwarn = _label("", wrap=False)
        self.lbl_bkgwarn.setStyleSheet("color:#e0a030; font-weight:600")
        row.addWidget(self.lbl_bkgwarn)
        row.addWidget(_label("     x axis:", "muted", wrap=False))
        self.cmb_xaxis = QtWidgets.QComboBox()
        self.cmb_xaxis.addItems(["delay (one per run)", "another variable…"])
        self.cmb_xaxis.setToolTip("another variable: plot the ROI signal against any recorded value\n"
                                  "(a motor such as z, a temperature …), within one run or over several runs.\n"
                                  "Opens the 'Parameter scan' page, where you paste its source and key.")
        self.cmb_xaxis.activated.connect(self._xaxis_chosen)
        row.addWidget(self.cmb_xaxis)
        row.addStretch(1)
        v.addLayout(row)
        g = _glw()
        self.p_raw = theme.style_plot(g.addPlot(row=0, col=0), "ROI signal (background subtracted)",
                                      "delay (fs)", "mean S (counts)")
        self.p_norm = theme.style_plot(g.addPlot(row=1, col=0), "Normalised", "delay (fs)", "mean S / mean N")
        self.p_diode = theme.style_plot(g.addPlot(row=2, col=0), "Diodes per scan", "delay (fs)", "mean diode")
        for p in (self.p_raw, self.p_diode):
            p.addLegend(offset=(-10, 10), colCount=3 if p is self.p_raw else 1)
            theme.style_plot(p, p._xtg["title"], p._xtg["x"], p._xtg["y"])
        self.p_norm.setXLink(self.p_raw)
        self.p_diode.setXLink(self.p_raw)
        g.ci.layout.setRowStretchFactor(0, 3)
        g.ci.layout.setRowStretchFactor(1, 3)
        g.ci.layout.setRowStretchFactor(2, 2)
        v.addWidget(g)
        return w

    def _build_corr(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(_label("signal ROI:", "muted", wrap=False))
        self.cmb_sig = ShrinkCombo(200, 110)
        self.cmb_sig.currentTextChanged.connect(lambda *_: self._draw_corr())
        row.addWidget(self.cmb_sig)
        row.addWidget(_label("   diodes:", "muted", wrap=False))
        self.cmb_diode_view = QtWidgets.QComboBox()
        self.cmb_diode_view.addItems(["distribution", "vs shot number"])
        self.cmb_diode_view.setToolTip("histograms, or each diode vs shot number in acquisition (scan) order")
        self.cmb_diode_view.currentTextChanged.connect(lambda *_: self._draw_corr())
        row.addWidget(self.cmb_diode_view)
        row.addStretch(1)
        v.addLayout(row)
        v.addWidget(_label("points coloured by delay (viridis: early = dark, late = yellow)", "muted"))
        g = _glw()
        self.c_pp = theme.style_plot(g.addPlot(row=0, col=0), "Pump vs probe", "probe diode", "pump diode")
        self.c_hpu = theme.style_plot(g.addPlot(row=0, col=1), "Pump diode", "pump", "shots")
        self.c_hpr = theme.style_plot(g.addPlot(row=0, col=2), "Probe diode", "probe", "shots")
        self.c_spu = theme.style_plot(g.addPlot(row=1, col=0), "Signal vs pump", "pump diode", "signal")
        self.c_spr = theme.style_plot(g.addPlot(row=1, col=1), "Signal vs probe", "probe diode", "signal")
        self.c_sn = theme.style_plot(g.addPlot(row=1, col=2), "Signal vs normaliser", "normaliser", "signal")
        v.addWidget(g)
        return w

    # ------------------------------------------------------------ parameter scan page
    def _build_param(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.addWidget(_label(
            "ROI signal against a parameter recorded per train: a motor stepped inside one run (sample rotation, "
            "grating-sample distance), a slow drift (temperature), or one value per run across several runs. "
            "Type the source and key (or pick a preset from the previous experiments) and press Read. "
            "Shots from all used scans in the table are pooled; cleaning and normalisation are as on the other pages.",
            "muted"))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(_label("preset:", "muted", wrap=False))
        self.cmb_ppreset = ShrinkCombo(230, 120)
        self.cmb_ppreset.addItems(["custom"] + list(PARAM_PRESETS))
        self.cmb_ppreset.setToolTip("Parameters read in the p7016 notebooks (rocking_MID.ipynb)")
        self.cmb_ppreset.currentTextChanged.connect(self._param_preset_chosen)
        row.addWidget(self.cmb_ppreset)
        self.ed_psrc = QtWidgets.QLineEdit(self.cfg.param.source)
        self.ed_psrc.setPlaceholderText("source, e.g. MID_EXP_UPP/MDL/DATA_SELECTOR_1")
        self.ed_pkey = QtWidgets.QLineEdit(self.cfg.param.key)
        self.ed_pkey.setPlaceholderText("key, e.g. midExpUppMotorT3.actualPosition")
        for ed, wd in ((self.ed_psrc, 3), (self.ed_pkey, 3)):
            ed.setMinimumWidth(110)
            row.addWidget(ed, wd)
        self.ed_plabel = QtWidgets.QLineEdit(self.cfg.param_label)
        self.ed_plabel.setPlaceholderText("name")
        self.ed_punit = QtWidgets.QLineEdit(self.cfg.param_unit)
        self.ed_punit.setPlaceholderText("unit")
        self.ed_punit.setMaximumWidth(60)
        row.addWidget(self.ed_plabel, 2)
        row.addWidget(self.ed_punit)
        self.b_pread = QtWidgets.QPushButton("Read", objectName="primary")
        self.b_pread.setToolTip("Read this parameter for all loaded scans (and for scans loaded later)")
        self.b_pread.clicked.connect(self.apply_param)
        row.addWidget(self.b_pread)
        v.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(_label("points:", "muted", wrap=False))
        self.cmb_pbin = ShrinkCombo(0, 120)
        self.cmb_pbin.addItems(["auto (one per motor step)", "equal-width bins"])
        self.cmb_pbin.setToolTip("auto: one point per motor position if the parameter is stepped (up to 60 "
                                 "positions),\notherwise equal-width bins (continuous drift, e.g. temperature)")
        self.sp_pbins = QtWidgets.QSpinBox()
        self.sp_pbins.setRange(2, 500)
        self.sp_pbins.setValue(30)
        self.sp_pbins.setPrefix("bins ")
        self.cmb_pgroup = ShrinkCombo(0, 110)
        self.cmb_pgroup.addItems(["all scans pooled", "one curve per scan", "one curve per delay"])
        self.cmb_psig = ShrinkCombo(200, 110)
        self.cmb_psig.setToolTip("ROI for the per-scan / per-delay curves and the shot scatter")
        for wdg in (self.cmb_pbin, self.sp_pbins, self.cmb_pgroup):
            row.addWidget(wdg)
        row.addWidget(_label("   ROI:", "muted", wrap=False))
        row.addWidget(self.cmb_psig)
        self.lbl_pinfo = elastic(_label("", "muted", wrap=False), tip=False)
        row.addWidget(self.lbl_pinfo, 1)
        v.addLayout(row)
        self.cmb_pbin.currentTextChanged.connect(lambda *_: self._draw_param())
        self.sp_pbins.valueChanged.connect(lambda *_: self._draw_param())
        self.cmb_pgroup.currentTextChanged.connect(lambda *_: self._draw_param())
        self.cmb_psig.currentTextChanged.connect(lambda *_: self._draw_param())
        g = _glw()
        self.pp_norm = theme.style_plot(g.addPlot(row=0, col=0, colspan=2), "Normalised ROI signal", "parameter",
                                        "mean S / mean N")
        self.pp_raw = theme.style_plot(g.addPlot(row=0, col=2), "ROI signal (background subtracted)", "parameter",
                                       "mean S")
        self.pp_shot = theme.style_plot(g.addPlot(row=1, col=0), "Shots", "parameter", "signal")
        self.pp_int = theme.style_plot(g.addPlot(row=1, col=1), "Pump / probe", "parameter", "mean")
        self.pp_trace = theme.style_plot(g.addPlot(row=1, col=2), "Parameter vs shot", "shot (scan order)",
                                         "parameter")
        for p in (self.pp_norm, self.pp_int):
            p.addLegend(offset=(-10, 10), colCount=2)
            theme.style_plot(p, p._xtg["title"], p._xtg["x"], p._xtg["y"])
        g.ci.layout.setRowStretchFactor(0, 3)
        g.ci.layout.setRowStretchFactor(1, 2)
        v.addWidget(g)
        return w

    def _param_preset_chosen(self, name):
        if name in PARAM_PRESETS:
            src, key, unit = PARAM_PRESETS[name]
            self.ed_psrc.setText(src)
            self.ed_pkey.setText(key)
            self.ed_plabel.setText(name.split(" (")[0])
            self.ed_punit.setText(unit)

    def bkg_mode(self):
        """("roi", None), ("none", None) or ("run", run number)."""
        d = self.cmb_bkg.currentData() if hasattr(self, "cmb_bkg") else None
        if isinstance(d, int) and d in self.scans:
            return "run", d
        return (d if d in ("roi", "none") else "roi"), None

    def bkg_geoms(self, det):
        """Background ROIs of a detector: only in 'background ROI' mode."""
        if self.bkg_mode()[0] != "roi":
            return []
        return self.views[det].panel.background_geoms()

    def _fill_bkg_combo(self):
        cur = self.cmb_bkg.currentData() if self.cmb_bkg.count() else "roi"
        self.cmb_bkg.blockSignals(True)
        self.cmb_bkg.clear()
        self.cmb_bkg.addItem("background ROI", "roi")
        self.cmb_bkg.addItem("none", "none")
        for s in sorted(self.scans, key=lambda k: (np.nan_to_num(self.scans[k].delay, nan=np.inf), k)):
            dl = self.scans[s].delay
            self.cmb_bkg.addItem(f"run {s}  ({'?' if np.isnan(dl) else f'{dl:g}'} fs)", s)
        i = self.cmb_bkg.findData(cur)
        self.cmb_bkg.setCurrentIndex(max(i, 0))
        self.cmb_bkg.blockSignals(False)

    def _update_bkg_flag(self):
        mode, run = self.bkg_mode()
        txt = ""
        if mode == "run":
            dl = self.scans[run].delay
            if not np.isfinite(dl):
                txt = f"⚠ run {run} has no delay: is it really before time zero?"
            elif dl >= 0:
                txt = f"⚠ run {run} is at delay {dl:g} fs (≥ 0): not a background run"
        if txt and txt != getattr(self, "_last_bkg_flag", ""):
            self.log(txt)
        self._last_bkg_flag = txt
        self.lbl_bkgwarn.setText("  " + txt if txt else "")
        titles = {"roi": "ROI signal (background ROI subtracted)", "none": "ROI signal (no background subtraction)",
                  "run": f"ROI signal minus run {run}"}
        theme.set_title(self.p_raw, titles[mode] + (" · per pixel" if self.per_pixel() else ""))
        self.p_raw.setLabel("left", "mean S per pixel (counts)" if self.per_pixel() else "mean S (counts)")

    def _bkg_changed(self):
        mode, run = self.bkg_mode()
        self.log("background: " + {"roi": "background ROI", "none": "none"}.get(mode, f"run {run}"))
        self.recompute()

    def _xaxis_chosen(self, i):
        if i == 1:
            self.set_param_visible(True)
            self.ed_psrc.setFocus()
            self.log("another variable: paste its source and key on the 'Parameter scan' page, then press Read")
        self.cmb_xaxis.setCurrentIndex(0)

    def set_param_visible(self, on):
        on = bool(on)
        self.cfg.param_enabled = on
        if self.b_param.isChecked() != on:
            self.b_param.blockSignals(True)
            self.b_param.setChecked(on)
            self.b_param.blockSignals(False)
        self.tabs.setTabVisible(self.param_tab, on)
        if on:
            self.tabs.setCurrentIndex(self.param_tab)
            self._draw_param()

    def set_pos_window(self, on):
        """ROI position check (centre of mass / max) in its own window."""
        if getattr(self, "pos_win", None) is None:
            self.pos_win = RoiPositionWindow(self)
            self.pos_win.closed.connect(lambda: self.b_pos.setChecked(False))
        if self.b_pos.isChecked() != bool(on):
            self.b_pos.blockSignals(True)
            self.b_pos.setChecked(bool(on))
            self.b_pos.blockSignals(False)
        if on:
            self.pos_win.show()
            self.pos_win.raise_()
            self.pos_win.refresh()
        elif self.pos_win.isVisible():
            self.pos_win.hide()

    def param_axis_label(self):
        u = self.cfg.param_unit.strip()
        return f"{self.cfg.param_label or 'parameter'}" + (f" ({u})" if u else "")

    def apply_param(self):
        """Take the source typed on the page and read it for every loaded scan."""
        src, key = self.ed_psrc.text().strip(), self.ed_pkey.text().strip()
        self.cfg.param = Source(src, key)
        self.cfg.param_label = self.ed_plabel.text().strip() or key.split(".")[0] or "parameter"
        self.cfg.param_unit = self.ed_punit.text().strip()
        self.cfg.param_enabled = True
        if not src or not key:
            self.log("parameter scan: type a source and a key")
            self._draw_param()
            return
        self.log(f"parameter: {src} / {key}")
        if not self.scans:
            self._draw_param()
            return
        need_reload = []
        for s, d in self.scans.items():
            try:
                x = self.backend.load_param(s, d.train)
            except Exception as e:
                self.log(f"scan {s}: parameter not read: {type(e).__name__}: {e}")
                d.param = None
                continue
            if x is None:
                need_reload.append(s)
            else:
                d.param = np.asarray(x, float)
        if need_reload:
            self.reload_scans()
        else:
            self.recompute()

    def _mask_toggled(self, on):
        self.cfg.apply_mask = bool(on)
        self.log(f"bad-pixel masks {'on' if on else 'off'}")
        if self.scans:
            self.reload_scans()

    def _mask_reread(self):
        from . import maskstore
        maskstore.clear(self.cfg.proposal)
        self.log(f"{maskstore.path_for(self.cfg.proposal).name} deleted: bad-pixel masks are read again")
        if self.scans:
            self.reload_scans()

    def _mask_pt_toggled(self, on):
        self.cfg.mask_per_train = bool(on)
        self.log(f"bad-pixel mask {'of every frame' if on else 'read once per run'}")
        if self.scans:
            self.reload_scans()

    def _photonize_toggled(self, on):
        self.cfg.photonize = bool(on)
        self.log(f"photonizing the Jungfraus {'on' if on else 'off'}")
        if self.backend_kind == "simulated":
            self.log("(simulated data are not in keV: photonizing applies to real data only)")
        if self.scans:
            self.reload_scans()

    def reload_scans(self):
        """Read the loaded scans again (new sources), keeping 'use' ticks and typed delays."""
        keep_use = dict(self.use)
        delays = {s: (d.delay, d.delay_from) for s, d in self.scans.items()}
        order = list(getattr(self, "_typed_order", [])) or sorted(self.scans)
        self.load([s for s in order if s in self.scans] + [s for s in self.scans if s not in order], append=False)
        for s in self.scans:
            self.use[s] = keep_use.get(s, True)
            old, how = delays.get(s, (np.nan, "data"))
            if how == "typed":
                self.scans[s].delay, self.scans[s].delay_from = old, how
        self._fill_scan_table()
        self._fill_sources()
        self.recompute()

    def param_values(self, s):
        x = self.scans[s].param
        return np.full(self.scans[s].n, np.nan) if x is None else np.asarray(x, float)

    def _param_points(self, x, num, den):
        mode = "auto" if self.cmb_pbin.currentText().startswith("auto") else "bins"
        return an.param_series(x, num, den, mode=mode, nbins=self.sp_pbins.value())

    def _draw_param(self):
        if not hasattr(self, "pp_norm") or not self.cfg.param_enabled:
            return
        plots = (self.pp_norm, self.pp_raw, self.pp_shot, self.pp_int, self.pp_trace)
        for p in plots:
            p.clear()
        xl = self.param_axis_label()
        for p in plots[:4]:
            p._xtg["x"] = xl
        self.pp_trace._xtg["y"] = xl
        for p in plots:
            theme.set_title(p, p._xtg["title"])
        cur = self.cmb_psig.currentText()
        keys = self.signal_keys()
        self.cmb_psig.blockSignals(True)
        self.cmb_psig.clear()
        self.cmb_psig.addItems([k[0] for k in keys])
        self.cmb_psig.setCurrentIndex(max(self.cmb_psig.findText(cur), 0))
        self.cmb_psig.blockSignals(False)
        scans = self.sorted_scans()
        missing = [s for s in scans if self.scans[s].param is None]
        if not self.cfg.param.source:
            self._param_info("   type a source and key, then Read")
            return
        if not scans or len(missing) == len(scans):
            self._param_info("   no parameter values yet: press Read" if scans else "   load scans first")
            return
        use = [s for s in scans if s not in missing]
        x = self._concat_over(use, self.param_values)
        steps = an.param_steps(x)
        what = (f"{len(steps)} positions" if steps is not None and self.cmb_pbin.currentText().startswith("auto")
                else f"{self.sp_pbins.value()} bins")
        fin = x[np.isfinite(x)]
        rng = f"{fin.min():.4g} … {fin.max():.4g}" if fin.size else "no finite values"
        info = f"   {len(use)} scans · {fin.size} shots · {rng} · {what}"
        if missing:
            info += f" · no parameter in scan {', '.join(map(str, missing))}"
        self._param_info(info)
        norm = self.cmb_norm.currentText()
        nrm = self._concat_over(use, lambda s: an.normaliser(norm, self.inten(s, 'pump'), self.inten(s, 'probe')))
        theme.set_title(self.pp_norm, f"Normalised by {norm}")
        group = self.cmb_pgroup.currentText()
        sel = next((k for k in keys if k[0] == self.cmb_psig.currentText()), keys[0] if keys else None)
        first_det = next(iter(self.views), None)
        if group == "all scans pooled":
            for label, det, roi in keys:
                bkg = self.bkg_geoms(det)
                sig = self._concat_over(use, lambda s: self.roi_values(det, s, roi, bkg))
                dash = det != first_det
                self._curve(self.pp_norm, *self._param_points(x, sig, nrm)[:3], roi.color, label, dash)
                self._curve(self.pp_raw, *self._param_points(x, sig, None)[:3], roi.color, label, dash)
        elif sel is not None:
            _, det, roi = sel
            bkg = self.bkg_geoms(det)
            if group == "one curve per scan":
                groups = [(f"scan {s}", [s]) for s in use]
            else:
                dl = sorted({self.scans[s].delay for s in use}, key=lambda t: (np.isnan(t), t))
                groups = [(f"{t:g} fs", [s for s in use if self.scans[s].delay == t or
                                         (np.isnan(t) and np.isnan(self.scans[s].delay))]) for t in dl]
            cmap = pg.colormap.get("viridis")
            for i, (name, ss) in enumerate(groups):
                c = tuple(int(v) for v in cmap.map(i / max(len(groups) - 1, 1), mode="byte")[:3])
                xs = self._concat_over(ss, self.param_values)
                sig = self._concat_over(ss, lambda s: self.roi_values(det, s, roi, bkg))
                nn = self._concat_over(ss, lambda s: an.normaliser(norm, self.inten(s, 'pump'), self.inten(s, 'probe')))
                self._curve(self.pp_norm, *self._param_points(xs, sig, nn)[:3], c, name)
                self._curve(self.pp_raw, *self._param_points(xs, sig, None)[:3], c, name)
            theme.set_title(self.pp_norm, f"{sel[0]} · normalised by {norm}")
        for arm, color in (("pump", theme.PUMP), ("probe", theme.PROBE)):
            val = self._concat_over(use, lambda s: self.inten(s, arm))
            self._curve(self.pp_int, *self._param_points(x, val, None)[:3], color, arm)
        if sel is not None:
            _, det, roi = sel
            bkg = self.bkg_geoms(det)
            sig = self._concat_over(use, lambda s: self.roi_values(det, s, roi, bkg))
            self._scatter(self.pp_shot, x, sig, self._delay_colors_over(use), f"{roi.name}: kept shots",
                          with_r=False)
        for p in plots:
            p.enableAutoRange()
        n = np.arange(x.size)
        self.pp_trace.plot(n, x, pen=None, symbol="o", symbolSize=3, symbolPen=None,
                           symbolBrush=pg.mkBrush(theme.neutral(160)))

    def _concat_over(self, scans, fn):
        if not scans:
            return np.array([])
        return np.concatenate([np.asarray(fn(s), float)[self.keep[s]] for s in scans])

    def _delay_colors_over(self, scans):
        t = self._concat_over(scans, lambda s: np.full(self.scans[s].n, self.scans[s].delay))
        fin = np.isfinite(t)
        lo, hi = (np.min(t[fin]), np.max(t[fin])) if fin.any() else (0, 1)
        u = np.where(fin, (t - lo) / max(hi - lo, 1e-9), 0.5)
        cols = pg.colormap.get("viridis").map(u, mode="byte")
        cols[:, 3] = 150
        return cols

    def _param_info(self, text):
        self.lbl_pinfo.setText(text)
        self.lbl_pinfo.setToolTip(text.strip())   # the label may be cut short in a narrow window

    def _build_crl(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        self.crl_banner = _label(
            "Beam pointing of each arm on the CRLs before the split-and-delay line. A feedback loop keeps the beams "
            "centred, so positions should stay near 0: a drift, a wide spread, or a diode that drops off-centre "
            "(white line = binned mean) means the loop is not keeping up.", "muted")
        v.addWidget(self.crl_banner)
        g = _glw()
        self.crl_plots = {}
        for r, arm in enumerate(("pump", "probe")):
            self.crl_plots[arm] = (
                theme.style_plot(g.addPlot(row=r, col=0), f"{arm.capitalize()} arm pointing", "x (µm)", "y (µm)"),
                theme.style_plot(g.addPlot(row=r, col=1), f"{arm} diode vs x", "x (µm)", f"{arm} diode"),
                theme.style_plot(g.addPlot(row=r, col=2), f"{arm} diode vs y", "y (µm)", f"{arm} diode"),
            )
        v.addWidget(g)
        return w

    def spec_mode(self):
        """"monitors": the spectrometers ARE the pump / probe intensity monitors (p7016);
        "upstream": spectrometers before the SDL, compared with the diamond diodes."""
        return "monitors" if self.cfg.intensity_from == "spectrometers" else "upstream"

    def _build_spec(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        if self.spec_mode() == "monitors":
            txt = ("Pump and probe intensity from the spectrometers: each spectrum is data.adc[frame, b:c] (frame = "
                   "Gotthard frame in the train, b:c = pixel window), edge-baseline subtracted and summed inside the "
                   "shaded band. Drag the band to change what counts as the pump / probe intensity: the whole "
                   "analysis (cleaning, normalisation) follows. The gas monitor (XGM) sees pump + probe together.")
        else:
            txt = ("Spectrometer upstream of the SDL vs the diamond diodes (after the SDL and the monochromators). "
                   "With self-seeded SASE the full integral includes the SASE pedestal, so its correlation with the "
                   "diodes can be close to 0. Drag the shaded band on a spectrum to integrate only part of it "
                   "(e.g. the seed line).")
        v.addWidget(_label(txt, "muted"))
        g = _glw()
        self.spec_plots, self.spec_regions, self.spec_windows = {}, {}, {}
        mon = self.spec_mode() == "monitors"
        # monitors (p7016): one row per spectrometer; upstream (11456): a 2 x 2 block per spectrometer
        cells = ([(0, 0), (0, 1), (1, 0), (1, 1)] if not mon else [(0, c) for c in range(4)])
        for r, (name, sp) in enumerate(self.cfg.spectrometers.items()):
            r0 = r if mon else 2 * r
            unit = "eV" if sp.eV0 is not None and sp.dE is not None else "px"
            where = f"[{sp.frame}, {sp.px[0]}:{sp.px[1]}]" if sp.frame >= 0 else f"[{sp.px[0]}:{sp.px[1]}]"
            ps = theme.style_plot(g.addPlot(row=r0 + cells[0][0], col=cells[0][1]),
                                  f"{sp.label} {where}" if mon else f"{sp.label}: mean spectrum", unit, "counts")
            reg = pg.LinearRegionItem(values=(sp.px[0], sp.px[1] - 1), brush=pg.mkBrush(91, 140, 255, 28))
            reg.setZValue(-5)
            reg.sigRegionChangeFinished.connect(lambda *_, n=name: self._spec_region_changed(n))
            ps.addItem(reg)
            self.spec_regions[name] = reg
            self.spec_windows[name] = None
            if mon:
                cols = ("Distribution", "vs shot number", "vs XGM")
            else:
                cols = ("Integral", "Pump diode", "Probe diode")
            self.spec_plots[name] = (ps,) + tuple(
                theme.style_plot(g.addPlot(row=r0 + cells[c + 1][0], col=cells[c + 1][1]), t, None, None)
                for c, t in enumerate(cols))
        self.xgm_plots = None
        if self.cfg.xgm.source:
            r = len(self.cfg.spectrometers) * (1 if mon else 2)
            names = (("XGM distribution", "XGM vs shot number", "Pump vs probe")
                     if mon else ("XGM distribution", "XGM vs shot number", "Pump diode vs XGM", "Probe diode vs XGM"))
            self.xgm_plots = tuple(theme.style_plot(g.addPlot(row=r if mon else r + c // 2, col=c if mon else c % 2),
                                                    t, None, None) for c, t in enumerate(names))
        if mon:
            for c, f in enumerate((5, 4, 4, 4)):
                g.ci.layout.setColumnStretchFactor(c, f)
        v.addWidget(g)
        return w

    def _spec_region_changed(self, name):
        axis = next((self.scans[s].spec_axis[name] for s in self.sorted_scans()
                     if name in self.scans[s].spec_axis), None)
        if axis is None:
            return
        lo, hi = self.spec_regions[name].getRegion()
        inside = np.flatnonzero((axis >= lo) & (axis <= hi))     # works for a falling eV axis too
        i0, i1 = (int(inside[0]), int(inside[-1]) + 1) if inside.size else (0, 0)
        self.spec_windows[name] = None if (i0 <= 0 and i1 >= len(axis)) else (i0, i1)
        if self.spec_mode() == "monitors" and name in (self.cfg.pump_spec, self.cfg.probe_spec):
            self.recompute()                 # the pump / probe intensity itself changed
        else:
            self._draw_spec()

    def set_spec_visible(self, on):
        on = bool(on)
        self.cfg.spec_enabled = on
        if self.b_spec.isChecked() != on:
            self.b_spec.blockSignals(True)
            self.b_spec.setChecked(on)
            self.b_spec.blockSignals(False)
        self.tabs.setTabVisible(self.spec_tab, on)
        if on:
            self._draw_spec()

    # ============================================================ theme / CRL
    def _theme_text(self):
        dark = theme.mode() == "dark"
        self.b_theme.setText("☀" if dark else "☾")           # icon only: keeps the header narrow
        self.b_theme.setToolTip("Switch to light mode" if dark else "Switch to dark mode")

    def toggle_theme(self):
        old = theme.mode()
        theme.apply(QtWidgets.QApplication.instance(), "light" if old == "dark" else "dark")
        if self.cmb_cmap.currentText() == theme.DEFAULT_CMAP[old]:   # user kept the default: follow theme
            self.set_cmap(theme.DEFAULT_CMAP[theme.mode()])
        self._theme_text()
        self.recompute()

    def set_crl_visible(self, on):
        on = bool(on)
        self.cfg.crl_enabled = on
        if self.b_crl.isChecked() != on:
            self.b_crl.blockSignals(True)
            self.b_crl.setChecked(on)
            self.b_crl.blockSignals(False)
        self.tabs.setTabVisible(self.crl_tab, on)
        if on:
            self._draw_crl()

    # ============================================================ loading
    def log(self, msg):
        self.log_box.appendPlainText(msg)
        self.statusBar().showMessage(msg, 5000)

    def _preset_chosen(self, *_):
        """Switch proposal: 7016 / 11456 load their sources file, custom asks for one.
        The pages differ between proposals, so the window is rebuilt (loaded scans are dropped)."""
        preset = self.cmb_preset.currentData()
        if preset == self.cfg.preset and preset != "custom":
            return
        path = None
        if preset == "custom":
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Sources file for a custom proposal", "",
                                                            "YAML / JSON (*.yaml *.yml *.json);;All files (*)")
        self.switch_preset(preset, path or None)

    def switch_preset(self, preset, custom_path=None):
        try:
            cfg = config_for(preset, custom_path)
        except Exception as e:                       # e.g. a file that is not there: say so, keep the window
            self.log(f"proposal {preset}: {type(e).__name__}: {e}")
            QtWidgets.QMessageBox.warning(self, "Sources file", str(e))
            return None
        if preset == "custom" and not custom_path:
            cfg.proposal = self.sp_prop.value()
        new = XTGMidWindow(cfg, self.backend_kind, self.sim_kw)
        new.resize(self.size())
        new.move(self.pos())
        new.show()
        new.log(f"proposal {cfg.proposal or '(not set)'}: sources from "
                f"{getattr(cfg, '_source_file', '') or custom_path or 'built-in defaults'}")
        if preset in ("7016", "11456") and not getattr(cfg, "_source_file", ""):
            new.log(f"{preset}: Sources_{preset}.yaml not found anywhere (also not in {PRESET_DIR}): "
                    "built-in defaults. Re-unzip the package.")
        elif preset == "11456" and str(PRESET_DIR) in getattr(cfg, "_source_file", str(PRESET_DIR)):
            new.log("11456: no Sources_11456.yaml found in the start folder, using the template (placeholders)")
        XTGMidWindow.last = new
        self.close()
        return new

    def _backend_changed(self, kind):
        self.backend_kind = kind
        self.backend = make_backend(kind, self.cfg, **(self.sim_kw if kind == "simulated" else {}))
        self.pills["backend"].setText(kind)
        sims = getattr(self.backend, "scans", lambda: [])()
        if sims and not self.ed_scans.text():                 # hint only: nothing is loaded until Load
            self.ed_scans.setText(f"{min(sims)}-{max(sims)}")

    def _ensure_rois(self):
        """Place the default ROIs on the reference run's image before the other runs are
        reduced (they are reduced with the ROIs as they are at that moment)."""
        d = self.scans.get(self.ref_scan)
        if d is None:
            return
        for det, v in self.views.items():
            if not v.panel.rois and det in d.images:
                v.panel.shape = d.mean_image(det).shape
                v.panel.defaults()

    def roi_spec(self):
        """{detector: (all ROI geometries, background geometries)} for reducing the runs
        that are not the reference run (switched-off ROIs too, so ticking them is free)."""
        return {det: ([r.geometry() for r in v.panel.rois], v.panel.background_geoms())
                for det, v in self.views.items()}

    def load(self, scans, append=False):
        self.cfg.proposal = self.sp_prop.value()
        if not append:
            self.scans.clear()
            self.use.clear()
            self._cache.clear()
            if getattr(self, "ref_scan", None) not in scans:
                self.ref_scan = scans[0] if scans else None
        elif getattr(self, "ref_scan", None) not in self.scans and scans:
            self.ref_scan = scans[0]
        known = delay_file.load(self.delays_path())
        self._typed_order = list(dict.fromkeys((list(getattr(self, '_typed_order', [])) if append else []) + list(scans)))
        scans = sorted(scans, key=lambda s: s != self.ref_scan)    # reference first: it places the ROIs
        ref = [s for s in scans if s == self.ref_scan]
        for s, d in load_many(self.backend, [(s, None) for s in ref]):
            self._add_loaded(s, d, known)
        self._ensure_rois()
        rest = [s for s in scans if s != self.ref_scan]
        spec = self.roi_spec()
        t0, n_done = time.time(), 0

        def tick():
            self.statusBar().showMessage(f"reading {len(rest)} runs ({n_done} done, {time.time() - t0:.0f} s)")
            QtWidgets.QApplication.processEvents()

        for s, d in load_many(self.backend, [(s, spec) for s in rest], tick=tick):
            n_done += 1
            self._add_loaded(s, d, known)
        self.statusBar().clearMessage()
        self._fill_ref_combo()
        self._fill_bkg_combo()
        self._update_delay_label()
        n_file = sum(self.scans[s].delay_from == "file" for s in scans if s in self.scans)
        if n_file:
            self.log(f"{n_file} delays filled in from {self.delays_path().name}")
        self._fill_scan_table()
        self._fill_sources()
        self.recompute()

    def _add_loaded(self, s, d, known):
        if isinstance(d, Exception):        # keep going, report in the log
            self.log(f"scan {s}: {type(d).__name__}: {d}")
            return
        d.delay_data, d.delay_from = d.delay, "data"
        if s in known:                      # the delays file wins over the data
            d.delay, d.delay_from = known[s], "file"
        self.scans[s] = d
        self.use.setdefault(s, True)
        self._cache = {k: v for k, v in self._cache.items() if k[1] != s}
        self.log(f"scan {s}: {d.n} shots, delay {d.delay:g} fs")
        for name, what in d.masks.items():
            self.log(f"scan {s}: {name} bad-pixel mask {what}")
        for note in d.notes:
            self.log(f"scan {s}: {note}")

    def _fill_scan_table(self):
        self.tbl.blockSignals(True)
        self.tbl.setRowCount(len(self.scans))
        order = self.cmb_order.currentText()
        if order == "run number":
            rows = sorted(self.scans)
        elif order == "as typed":
            rows = [s for s in getattr(self, "_typed_order", []) if s in self.scans]
            rows += [s for s in self.scans if s not in rows]
        else:
            rows = self.sorted_scans(all_=True)
        tips = {"data": "read from the data", "file": f"from {self.delays_path().name}",
                "typed": f"typed here, saved to {self.delays_path().name}"}
        for i, s in enumerate(rows):
            d = self.scans[s]
            on = QtWidgets.QTableWidgetItem()
            on.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            on.setCheckState(QtCore.Qt.Checked if self.use[s] else QtCore.Qt.Unchecked)
            on.setData(QtCore.Qt.UserRole, s)
            self.tbl.setItem(i, 0, on)
            for j, val, edit in ((1, str(s), False), (2, "" if np.isnan(d.delay) else f"{d.delay:g}", True),
                                 (3, str(d.n), False), (4, "", False)):
                it = QtWidgets.QTableWidgetItem(val)
                if not edit:
                    it.setFlags(QtCore.Qt.ItemIsEnabled)
                if j == 2:
                    it.setToolTip(f"delay {tips.get(d.delay_from, '')}; type a value to set it, "
                                  "clear it to go back to the data")
                    if d.delay_from != "data":
                        f = it.font()
                        f.setBold(True)
                        it.setFont(f)
                self.tbl.setItem(i, j, it)
        self.tbl.blockSignals(False)

    def _scan_item_changed(self, item):
        s = self.tbl.item(item.row(), 0).data(QtCore.Qt.UserRole)
        if item.column() == 0:
            self.use[s] = item.checkState() == QtCore.Qt.Checked
        elif item.column() == 2:
            self.set_delay(s, item.text())
            QtCore.QTimer.singleShot(0, self._fill_scan_table)     # re-sort / re-style after the edit
        self._fill_sources()
        self._schedule()

    # ------------------------------------------------------------ run -> delay file
    def delays_path(self):
        return delay_file.path_for(self.cfg.proposal)

    def _update_delay_label(self):
        p = self.delays_path()
        n = len(delay_file.load(p))
        self.lbl_delays.setText(f"Typed delays are saved to {p.name} ({n} run{'s' * (n != 1)}) and filled in "
                                "when a run is loaded again. Bold = from that file.")
        self.lbl_delays.setToolTip(str(p))

    def set_delay(self, s, text):
        """Delay of run s typed in the table: saved to the delays file (empty -> removed,
        back to the delay read from the data)."""
        d = self.scans[s]
        table = delay_file.load(self.delays_path())
        try:
            val = float(str(text).strip())
        except ValueError:
            val = np.nan
        if np.isfinite(val):
            d.delay, d.delay_from = val, "typed"
            table[s] = val
        else:
            d.delay, d.delay_from = d.delay_data, "data"
            table.pop(s, None)
        try:
            delay_file.save(self.delays_path(), table)
        except OSError as e:
            self.log(f"could not save {self.delays_path()}: {e}")
        self._update_delay_label()

    def _fill_sources(self):
        for det, view in self.panes().items():
            cur = view.source.currentText()
            view.source.blockSignals(True)
            view.source.clear()
            view.source.addItem(ALL_SCANS)
            for s in self.sorted_scans():
                view.source.addItem(f"scan {s}  (Δt = {self.scans[s].delay:g} fs)", s)
            i = view.source.findText(cur)
            view.source.setCurrentIndex(max(i, 0))
            view.source.blockSignals(False)
            if view is self.mosaic:
                self._draw_mosaic()
            else:
                self._draw_image(det)

    def _show_sources(self):
        lines = [f"{'name':<14}{'source':<52}key"]
        for n, d in self.cfg.detectors.items():
            lines.append(f"{n:<14}{d.data.source:<52}{d.data.key}")
        for n, sp in self.cfg.spectrometers.items():
            lines.append(f"{n:<14}{sp.data.source:<52}{sp.data.key}")
        if self.cfg.apply_mask:
            lines.append("")
            lines.append("bad-pixel masks (mask > 0 -> NaN):")
            for n, d in list(self.cfg.detectors.items()) + list(self.cfg.spectrometers.items()):
                lines.append(f"{n:<14}{mask_key_for(d.data.key, d.mask_key) or '(none)'}")
        else:
            lines.append("")
            lines.append("bad-pixel masks: off")
        lines.append("")
        if not self.cfg.photonize:
            lines.append("photonize Jungfraus: off")
        elif self.cfg.photon_energy_keV and self.cfg.photon_energy_keV > 0:
            lines.append(f"photonize Jungfraus: {self.cfg.photon_energy_keV:g} keV (fixed)")
        else:
            pe = self.cfg.photon_energy
            lines.append(f"photonize Jungfraus: photon energy from {pe.source} / {pe.key}")
        for k in ("pump", "probe", "delay") + self.cfg.CRL_KEYS:
            s = getattr(self.cfg, k)
            lines.append(f"{k:<14}{s.source:<52}{s.key}")
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Data sources")
        box.setText("Source names (PLACEHOLDER = not known yet). Edit them in a YAML file and start with "
                    "--config; --write-config dumps these defaults.")
        box.setDetailedText("\n".join(lines))
        box.exec_() if hasattr(box, "exec_") else box.exec()

    def _update_placeholder_pill(self):
        ph = self.cfg.placeholders()
        self.pill_ph.setVisible(bool(ph))
        self.pill_ph.setText(f"{len(ph)} placeholder{'s' * (len(ph) != 1)}")
        self.pill_ph.setToolTip("Placeholder source names (edit them in the sources file):\n" + "\n".join(ph))
        self.pills["backend"].setText(self.backend_kind)

    # ============================================================ analysis
    def sorted_scans(self, all_=False):
        ss = [s for s in self.scans if all_ or self.use.get(s)]
        return sorted(ss, key=lambda s: (np.nan_to_num(self.scans[s].delay, nan=np.inf), s))

    def _schedule(self, *_):
        self._timer.start()

    def inten(self, s, arm):
        """Per-shot pump or probe intensity of scan s: the diode, or (p7016) the integral of
        the arm's spectrometer inside the band drawn on the spectrometer page."""
        d = self.scans[s]
        name = self.cfg.pump_spec if arm == "pump" else self.cfg.probe_spec
        if self.cfg.intensity_from != "spectrometers" or name not in d.spectra:
            return d.pump if arm == "pump" else d.probe
        win = self.spec_windows.get(name)
        key = ("inten", s, name, win)
        if key not in self._cache:
            self._cache[key] = an.spectrum_integral(d.spectra[name], self.cfg.spectrometers[name].n_edge, win)[0]
        return self._cache[key]

    def roi_values(self, det, scan, roi, bkg):
        """Per-shot ROI signal of a scan; in 'run' background mode minus the mean of the
        background run (its kept shots)."""
        v = self._roi_values(det, scan, roi, bkg)
        mode, run = self.bkg_mode()
        if mode != "run":
            return v
        b = self._roi_values(det, run, roi, bkg)
        keep = self.keep.get(run)
        b = b[keep] if keep is not None and len(keep) == len(b) else b
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return v - np.nanmean(b)

    def _roi_values(self, det, scan, roi, bkg):
        key = (det, scan, roi.geometry(), tuple(bkg))
        if key not in self._cache:
            if len(self._cache) > 2000:        # old ROI positions
                self._cache.clear()
            d = self.scans[scan]
            if det in d.stats:                 # not the reference run: numbers kept at load time
                v = d.stats[det].signal(roi.geometry(), bkg)
                if v is None:                  # ROI moved since: this run must be read again
                    self._stale.add(scan)
                    return np.full(d.n, np.nan)
                self._cache[key] = v
            else:
                self._cache[key] = an.roi_signal(d.images[det], roi.geometry(), bkg)
        v = self._cache[key]
        if self.per_pixel():
            n = self._roi_npx(det, scan, roi)
            v = v / n if n else np.full(len(v), np.nan)
        return v

    def per_pixel(self):
        return hasattr(self, "cmb_value") and self.cmb_value.currentIndex() == 1

    def _roi_npx(self, det, scan, roi):
        """Pixels of the ROI that are ever finite in this scan (masked / gap pixels do not count)."""
        key = ("npx", det, scan, roi.geometry())
        if key not in self._cache:
            d = self.scans[scan]
            if det in d.stats:
                from .reduce import geom_key
                n = d.stats[det].npx.get(geom_key(roi.geometry()), 0)
            else:
                n = an.roi_stats(d.images[det], roi.geometry())[1]
            self._cache[key] = n
        return self._cache[key]

    # ------------------------------------------------------------ reference run / light runs
    def _fill_ref_combo(self):
        self.cmb_ref.blockSignals(True)
        self.cmb_ref.clear()
        for s in sorted(self.scans):
            self.cmb_ref.addItem(f"{s}  (all images)" if s == self.ref_scan else str(s), s)
        i = self.cmb_ref.findData(self.ref_scan)
        self.cmb_ref.setCurrentIndex(max(i, 0))
        self.cmb_ref.blockSignals(False)

    def set_reference(self, scan):
        """Make `scan` the run whose images are kept; the old one is reduced to numbers."""
        if scan is None or scan == self.ref_scan or scan not in self.scans:
            return
        old, self.ref_scan = self.ref_scan, scan
        self.log(f"reference run: {scan} (all images kept); run {old} keeps only its numbers")
        self._reread([s for s in (old, scan) if s in self.scans])
        self._fill_ref_combo()
        self._fill_bkg_combo()

    def _reread(self, scans, merge=False):
        """Read runs again with the current ROIs / reference run, keeping delay, 'use' and
        (merge=True) the ROI numbers computed before."""
        spec = self.roi_spec()

        def tick():
            self.statusBar().showMessage(f"reading {len(scans)} run(s) again")
            QtWidgets.QApplication.processEvents()

        for s, d in load_many(self.backend, [(s, None if s == self.ref_scan else spec) for s in scans], tick=tick):
            if isinstance(d, Exception):
                self.log(f"scan {s}: {type(d).__name__}: {d}")
                continue
            old = self.scans[s]
            d.delay, d.delay_data, d.delay_from = old.delay, old.delay_data, old.delay_from
            if merge:
                for det, st in d.stats.items():
                    if det in old.stats:
                        old.stats[det].merge(st)
                        d.stats[det] = old.stats[det]
            self.scans[s] = d
            self._cache = {k: v for k, v in self._cache.items() if k[1] != s}
        self.statusBar().clearMessage()
        if getattr(self, "pos_win", None) is not None:
            self.pos_win._cache.clear()
        self.recompute()

    def recalc_stale(self):
        stale = sorted(self._stale & set(self.scans))
        if stale:
            self.log(f"ROIs changed: reading runs {', '.join(map(str, stale))} again")
            self._reread(stale, merge=True)

    def _update_stale(self):
        n = len(self._stale & set(self.scans))
        self.b_recalc.setVisible(n > 0)
        self.b_recalc.setText(f"Recalculate {n} run{'s' * (n != 1)} for the new ROIs")

    def memory_mb(self) -> float:
        tot = 0
        for d in self.scans.values():
            tot += sum(a.nbytes for a in d.images.values() if a is not None)
            tot += sum(st.nbytes for st in d.stats.values())
            tot += sum(a.nbytes for a in d.spectra.values() if a is not None)
        return tot / 1e6

    def signal_keys(self):
        """(label, detector, roi) for every signal ROI.  Label = the ROI name; only when two
        detectors have a ROI of the same name is a short detector tag added (JF1, JF2 …)."""
        rois = [(det, r) for det, view in self.views.items() for r in view.panel.signal_rois()]
        count = {}
        for _, r in rois:
            count[r.name] = count.get(r.name, 0) + 1
        out, seen = [], {}
        for det, r in rois:
            label = r.name
            if count[r.name] > 1:
                tag = self.cfg.detectors[det].label.replace("Jungfrau ", "JF").replace("Jungfrau", "JF")
                label = f"{r.name} [{tag}]"
            seen[label] = seen.get(label, 0) + 1
            if seen[label] > 1:                     # same name twice on one detector: keep both
                label += f" ({seen[label]})"
            out.append((label, det, r))
        return out

    def recompute(self):
        scans = self.sorted_scans()
        self._stale = set()
        cut, skipped = {}, {}
        for s in scans:
            r = {}
            self.keep[s] = an.clean_scan(self.inten(s, 'pump'), self.inten(s, 'probe'), self.sp_mad.value(),
                                         self.sp_min.value(), self.cb_pos.isChecked(), report=r)
            for k, v in r["cut"].items():
                cut[k] = cut.get(k, 0) + v
            for k in r["skipped"]:
                skipped.setdefault(k, []).append(s)
        txt = "cut: " + (", ".join(f"{v} {k}" for k, v in cut.items() if v) or "none")
        if skipped:
            txt += "<br><span style='color:#e0a030'>not applied (would cut all shots): " + "; ".join(
                f"{k} in scan {', '.join(map(str, v))}" for k, v in skipped.items()) + "</span>"
            msg = "; ".join(f"{k}: scans {', '.join(map(str, v))}" for k, v in skipped.items())
            if msg != getattr(self, "_last_skip_msg", None):
                self.log(f"cut skipped, it would remove every shot ({msg})")
                self._last_skip_msg = msg
        self.lbl_cuts.setText(txt)
        for i in range(self.tbl.rowCount()):
            s = self.tbl.item(i, 0).data(QtCore.Qt.UserRole)
            if s in self.keep and self.use.get(s):
                self.tbl.blockSignals(True)
                self.tbl.item(i, 4).setText(str(int(self.keep[s].sum())))
                self.tbl.blockSignals(False)
        self._compute_series(scans)
        self._draw_delay(scans)
        cur = self.cmb_sig.currentText()
        self.cmb_sig.blockSignals(True)
        self.cmb_sig.clear()
        self.cmb_sig.addItems([k[0] for k in self.signal_keys()])
        self.cmb_sig.setCurrentIndex(max(self.cmb_sig.findText(cur), 0))
        self.cmb_sig.blockSignals(False)
        self._draw_corr()
        self._draw_crl()
        self._draw_spec()
        self._draw_param()
        if getattr(self, "pos_win", None) is not None:
            self.pos_win.refresh()
        self.fit_page.refresh()
        n_all = sum(self.scans[s].n for s in scans)
        n_keep = sum(int(self.keep[s].sum()) for s in scans)
        delays = {self.scans[s].delay for s in scans if np.isfinite(self.scans[s].delay)}
        self.pills["scans"].setText(f"{len(scans)} scans")
        self.pills["delays"].setText(f"{len(delays)} delays")
        self.pills["shots"].setText(f"{n_keep}/{n_all} shots kept")
        self.pills["memory"].setText(f"{self.memory_mb():.0f} MB")
        self.pills["memory"].setToolTip("memory used by the loaded runs (reference run: all images;\n"
                                        "other runs: only their numbers)")
        self._update_stale()
        self._update_bkg_flag()

    def _compute_series(self, scans):
        """series[label] = dict(raw=(t, y, e, n), norm=(t, y, e, n), color)."""
        self.series = {}
        norm = self.cmb_norm.currentText()
        delays = [self.scans[s].delay for s in scans]
        for label, det, roi in self.signal_keys():
            bkg = self.bkg_geoms(det)
            sig = [self.roi_values(det, s, roi, bkg)[self.keep[s]] for s in scans]
            nrm = [an.normaliser(norm, self.inten(s, 'pump'), self.inten(s, 'probe'))[self.keep[s]] for s in scans]
            ones = [np.ones_like(x) for x in sig]
            self.series[label] = dict(raw=an.delay_series(delays, sig, ones),
                                      norm=an.delay_series(delays, sig, nrm), color=roi.color,
                                      det=det, dash=det != next(iter(self.views)))

    def _curve(self, p, t, y, e, color, name, dash=False):
        if len(t) == 0:
            return
        style = QtCore.Qt.DashLine if dash else QtCore.Qt.SolidLine
        p.addItem(pg.ErrorBarItem(x=t, y=y, height=2 * np.nan_to_num(e), pen=pg.mkPen(color + (150,), width=1)))
        p.plot(t, y, pen=pg.mkPen(color, width=1, style=style), symbol="o", symbolSize=4,
               symbolBrush=color, symbolPen=None, name=name)

    def _draw_delay(self, scans):
        for p in (self.p_raw, self.p_norm, self.p_diode):
            p.clear()
        for label, s in self.series.items():
            self._curve(self.p_raw, *s["raw"][:3], s["color"], label, s["dash"])
            self._curve(self.p_norm, *s["norm"][:3], s["color"], label, s["dash"])
        delays = [self.scans[s].delay for s in scans]
        for name, attr, color in (("pump", "pump", theme.PUMP), ("probe", "probe", theme.PROBE)):
            vals = [self.inten(s, attr)[self.keep[s]] for s in scans]
            t, y, e, _ = an.delay_series(delays, vals, [np.ones_like(v) for v in vals])
            self._curve(self.p_diode, t, y, e, color, name)
        n_txt = self.cmb_norm.currentText()
        theme.set_title(self.p_norm, f"Normalised by {n_txt}")

    def _scatter(self, p, x, y, colors, title=None, with_r=True, profile=False):
        p.clear()
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum():
            if colors is None:
                brushes = pg.mkBrush(theme.neutral(120))
            else:                       # one QBrush per distinct colour keeps pyqtgraph's symbol cache hot
                u, inv = np.unique(np.asarray(colors)[ok], axis=0, return_inverse=True)
                brushes = np.array([pg.mkBrush(*c) for c in u.tolist()], dtype=object)[inv.ravel()]
            p.addItem(pg.ScatterPlotItem(x[ok], y[ok], size=5, pen=None, brush=brushes))
        if profile and ok.sum() > 20:      # binned mean of y vs x: shows non-linear dependence
            edges = np.unique(np.nanpercentile(x[ok], np.linspace(2, 98, 13)))
            idx = np.digitize(x[ok], edges)
            xm = [np.nanmean(x[ok][idx == i]) for i in range(1, len(edges)) if (idx == i).sum() > 3]
            ym = [np.nanmean(y[ok][idx == i]) for i in range(1, len(edges)) if (idx == i).sum() > 3]
            p.plot(xm, ym, pen=pg.mkPen(theme.pal()["text"], width=1.5), symbol="o", symbolSize=4,
                   symbolBrush=theme.pal()["text"], symbolPen=None)
        if title:
            r = an.pearson(x, y) if with_r else np.nan
            theme.set_title(p, f"{title}  ·  r = {r:.2f}" if np.isfinite(r) else title)

    def _concat(self, fn):
        scans = self.sorted_scans()
        if not scans:
            return np.array([])
        return np.concatenate([np.asarray(fn(s), float)[self.keep[s]] for s in scans])

    def _delay_colors(self):
        scans = self.sorted_scans()
        if not scans:
            return np.zeros((0, 4), np.uint8)
        t = self._concat(lambda s: np.full(self.scans[s].n, self.scans[s].delay))
        fin = np.isfinite(t)
        lo, hi = (np.min(t[fin]), np.max(t[fin])) if fin.any() else (0, 1)
        u = np.where(fin, (t - lo) / max(hi - lo, 1e-9), 0.5)
        cols = pg.colormap.get("viridis").map(u, mode="byte")
        cols[:, 3] = 150
        return cols

    def _draw_corr(self):
        pump = self._concat(lambda s: self.inten(s, 'pump'))
        probe = self._concat(lambda s: self.inten(s, 'probe'))
        cols = self._delay_colors()
        self._scatter(self.c_pp, probe, pump, cols, "Pump vs probe")
        if self.cmb_diode_view.currentText() == "vs shot number":
            self._draw_diode_trace()
        else:
            for p, x, c, name in ((self.c_hpu, pump, theme.PUMP, "pump"), (self.c_hpr, probe, theme.PROBE, "probe")):
                p.clear()
                theme.set_title(p, self.arm_label(name).capitalize())
                p.setLabel("bottom", name)
                p.setLabel("left", "shots")
                x = x[np.isfinite(x)]
                if x.size:
                    h, e = np.histogram(x, bins=50)
                    p.plot(e, h, stepMode="center", fillLevel=0, brush=c + (110,), pen=pg.mkPen(c, width=1))
        keys = {k[0]: k for k in self.signal_keys()}
        sel = keys.get(self.cmb_sig.currentText())
        for p in (self.c_spu, self.c_spr, self.c_sn):
            p.clear()
        if sel is None or not self.sorted_scans():
            return
        _, det, roi = sel
        bkg = self.bkg_geoms(det)
        sig = self._concat(lambda s: self.roi_values(det, s, roi, bkg))
        nrm_name = self.cmb_norm.currentText()
        nrm = self._concat(lambda s: an.normaliser(nrm_name, self.inten(s, 'pump'), self.inten(s, 'probe')))
        self._scatter(self.c_spu, pump, sig, cols, f"{roi.name} vs pump")
        self._scatter(self.c_spr, probe, sig, cols, f"{roi.name} vs probe")
        self._scatter(self.c_sn, nrm, sig, cols, f"{roi.name} vs {nrm_name}")
        self.c_sn.setLabel("bottom", nrm_name)

    def spec_integrals(self, name):
        """(integral, pump, probe, scans) over kept shots of used scans, or None if no data."""
        scans = [s for s in self.sorted_scans() if name in self.scans[s].spectra]
        if not scans:
            return None
        integ = np.concatenate([self.spec_values(s, name)[self.keep[s]] for s in scans])
        pump = np.concatenate([self.inten(s, 'pump')[self.keep[s]] for s in scans])
        probe = np.concatenate([self.inten(s, 'probe')[self.keep[s]] for s in scans])
        return integ, pump, probe, scans

    def spec_values(self, s, name):
        """Per-shot integral of spectrometer `name` in scan s, inside its band."""
        if self.spec_mode() == "monitors" and name in (self.cfg.pump_spec, self.cfg.probe_spec):
            return self.inten(s, "pump" if name == self.cfg.pump_spec else "probe")
        sp, win = self.cfg.spectrometers[name], self.spec_windows.get(name)
        key = ("spec", s, name, win)
        if key not in self._cache:
            self._cache[key] = an.spectrum_integral(self.scans[s].spectra[name], sp.n_edge, win)[0]
        return self._cache[key]

    def _hist(self, p, x, color, title, xlabel):
        p.clear()
        theme.set_title(p, title)
        p.setLabel("bottom", xlabel)
        p.setLabel("left", "shots")
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size:
            h, e = np.histogram(x, bins=50)
            p.plot(e, h, stepMode="center", fillLevel=0, brush=tuple(color[:3]) + (110,),
                   pen=pg.mkPen(tuple(color[:3]), width=1))

    def _trace(self, p, fn, color, title, ylabel):
        """fn(s) -> per-shot values vs shot number, scans in acquisition (scan-number) order.
        Kept shots coloured, cut shots grey ×; dashed lines mark scan boundaries."""
        p.clear()
        theme.set_title(p, title)
        p.setLabel("bottom", "shot number (scan order)")
        p.setLabel("left", ylabel)
        scans = sorted(s for s in self.scans if self.use.get(s))
        vals = [(s, fn(s)) for s in scans]
        vals = [(s, np.asarray(y, float)) for s, y in vals if y is not None]
        if not vals:
            return
        y = np.concatenate([v for _, v in vals])
        keep = np.concatenate([self.keep.get(s, np.ones(self.scans[s].n, bool)) for s, _ in vals])
        x = np.arange(len(y))
        c = tuple(color[:3])
        p.plot(x[~keep], y[~keep], pen=None, symbol="x", symbolSize=6, symbolPen=pg.mkPen(theme.neutral(160)))
        p.plot(x[keep], y[keep], pen=pg.mkPen(c + (90,), width=1), symbol="o", symbolSize=3,
               symbolBrush=c, symbolPen=None)
        start = 0
        for s, v in vals:
            if start:
                p.addItem(pg.InfiniteLine(start, angle=90, pen=pg.mkPen(theme.neutral(90), style=QtCore.Qt.DashLine)))
            start += len(v)

    def xgm_values(self, s):
        x = self.scans[s].xgm
        return None if x is None else np.asarray(x, float)

    def _spec_color(self, name):
        if name == self.cfg.pump_spec:
            return theme.PUMP
        if name == self.cfg.probe_spec:
            return theme.PROBE
        return theme.neutral()

    def _draw_spec(self):
        if not hasattr(self, "spec_plots") or not self.cfg.spec_enabled:
            return
        mon = self.spec_mode() == "monitors"
        xgm_all = self._concat(lambda s: self.xgm_values(s) if self.xgm_values(s) is not None
                               else np.full(self.scans[s].n, np.nan))
        for name, (ps, p1, p2, p3) in self.spec_plots.items():
            for p in (p1, p2, p3):
                p.clear()
            for it in [i for i in ps.listDataItems()]:
                ps.removeItem(it)
            res = self.spec_integrals(name)
            if res is None:
                theme.set_title(p1, "no data from this spectrometer")
                continue
            integ, pump, probe, scans = res
            axis = self.scans[scans[0]].spec_axis[name]
            reg = self.spec_regions[name]
            if getattr(reg, "_axis_len", None) != len(axis) or getattr(reg, "_axis0", None) != axis[0]:
                reg._axis_len, reg._axis0 = len(axis), axis[0]        # band starts over the whole axis (px or eV)
                reg.blockSignals(True)
                reg.setRegion((float(np.min(axis)), float(np.max(axis))))
                reg.blockSignals(False)
                self.spec_windows[name] = None
            mean = np.nanmean(np.concatenate([self.scans[s].spectra[name][self.keep[s]] for s in scans]), axis=0)
            ps.plot(axis, mean, pen=pg.mkPen(theme.pal()["text"], width=1))
            win = self.spec_windows.get(name)
            what = "integral" if win is None else f"integral [{axis[win[0]]:g}, {axis[min(win[1], len(axis)) - 1]:g}]"
            label = self.cfg.spectrometers[name].label
            col = self._spec_color(name)
            if mon:
                short = label.split(" ")[0]
                self._hist(p1, integ, col, f"{short} distribution", what)
                self._trace(p2, lambda s, n=name: self.spec_values(s, n), col, f"{short} vs shot", what)
                if np.isfinite(xgm_all).any():
                    self._scatter(p3, xgm_all, integ, np.tile(tuple(col[:3]) + (130,), (len(integ), 1)),
                                  f"{short} vs XGM", profile=True)
                    p3.setLabel("bottom", "XGM")
                    p3.setLabel("left", what)
                else:
                    theme.set_title(p3, "no XGM data")
            else:
                self._hist(p1, integ, theme.neutral(), "Integral" if win is None else f"Integral {what[9:]}", what)
                for p in (p2, p3):
                    p.setLabel("bottom", what)
                self._scatter(p2, integ, pump, np.tile(theme.PUMP + (130,), (len(integ), 1)), "Pump diode", profile=True)
                self._scatter(p3, integ, probe, np.tile(theme.PROBE + (130,), (len(integ), 1)), "Probe diode", profile=True)
                p2.setLabel("left", "pump diode")
                p3.setLabel("left", "probe diode")
        self._draw_xgm(xgm_all)

    def _draw_xgm(self, xgm_all):
        if not self.xgm_plots:
            return
        for p in self.xgm_plots:
            p.clear()
        if not np.isfinite(xgm_all).any():
            theme.set_title(self.xgm_plots[0], "no XGM data in these scans")
            return
        c = theme.neutral()
        self._hist(self.xgm_plots[0], xgm_all, c, "XGM distribution", "XGM")
        self._trace(self.xgm_plots[1], self.xgm_values, c, "XGM vs shot", "XGM")
        pump = self._concat(lambda s: self.inten(s, "pump"))
        probe = self._concat(lambda s: self.inten(s, "probe"))
        if self.spec_mode() == "monitors":
            pl = self.cfg.spectrometers.get(self.cfg.pump_spec)
            ql = self.cfg.spectrometers.get(self.cfg.probe_spec)
            pn, qn = (pl.label.split(" ")[0] if pl else "pump"), (ql.label.split(" ")[0] if ql else "probe")
            self._scatter(self.xgm_plots[2], probe, pump, self._delay_colors(), f"{pn} vs {qn}")
            self.xgm_plots[2].setLabel("bottom", ql.label if ql else "probe")
            self.xgm_plots[2].setLabel("left", pl.label if pl else "pump")
        else:
            for p, y, col, name in ((self.xgm_plots[2], pump, theme.PUMP, "Pump diode"),
                                    (self.xgm_plots[3], probe, theme.PROBE, "Probe diode")):
                self._scatter(p, xgm_all, y, np.tile(col + (130,), (len(y), 1)), f"{name} vs XGM", profile=True)
                p.setLabel("bottom", "XGM")
                p.setLabel("left", name.lower())

    def _draw_diode_trace(self):
        """Each intensity monitor vs shot number (scan order)."""
        for p, attr, c in ((self.c_hpu, "pump", theme.PUMP), (self.c_hpr, "probe", theme.PROBE)):
            self._trace(p, lambda s, a=attr: self.inten(s, a), c, f"{self.arm_label(attr)} vs shot",
                        self.arm_label(attr))

    def arm_label(self, arm):
        """'pump diode', or the spectrometer that measures the arm (p7016: HIREX / DES)."""
        if self.spec_mode() == "monitors":
            name = self.cfg.pump_spec if arm == "pump" else self.cfg.probe_spec
            sp = self.cfg.spectrometers.get(name)
            if sp is not None:
                return f"{arm} ({sp.label.split(' ')[0]})"
        return f"{arm} diode"

    def _draw_crl(self):
        if not hasattr(self, "crl_plots") or not self.cfg.crl_enabled:
            return
        scans = [s for s in self.sorted_scans() if self.scans[s].crl]
        for arm, (pxy, px, py) in self.crl_plots.items():
            for p in (pxy, px, py):
                p.clear()
            if not scans:
                pxy.addItem(pg.TextItem("no CRL position data in these scans", color=theme.pal()["muted"]))
                continue
            x = self._concat(lambda s: self.scans[s].crl.get(f"crl_{arm}_x", np.full(self.scans[s].n, np.nan)))
            y = self._concat(lambda s: self.scans[s].crl.get(f"crl_{arm}_y", np.full(self.scans[s].n, np.nan)))
            inten = self._concat(lambda s: self.inten(s, arm))
            c = theme.PUMP if arm == "pump" else theme.PROBE
            fin = np.isfinite(inten)
            if fin.any():
                lo, hi = np.nanpercentile(inten, [2, 98])
                u = np.round(np.clip((inten - lo) / max(hi - lo, 1e-9), 0, 1) * 15) / 15
                cm = pg.colormap.get("viridis").map(np.nan_to_num(u), mode="byte")
                cm[:, 3] = 170
                cols = cm
            else:
                cols = None
            self._scatter(pxy, x, y, cols, f"{arm.capitalize()} arm pointing (colour = {arm} diode)",
                          with_r=False)
            self._scatter(px, x, inten, np.tile(c + (130,), (len(x), 1)), f"{arm} diode vs x",
                          profile=True)
            self._scatter(py, y, inten, np.tile(c + (130,), (len(y), 1)), f"{arm} diode vs y",
                          profile=True)

    def _draw_image(self, det):
        view = self.views[det]
        scans = self.sorted_scans()
        if not scans:
            view.show_image(None)
            return
        s = view.source.currentData()
        if s is None or s not in self.scans:
            n = np.array([self.scans[k].n for k in scans], float)
            im = sum(self.scans[k].mean_image(det) * w for k, w in zip(scans, n)) / n.sum()
        else:
            im = self.scans[s].mean_image(det)
        view.show_image(im.astype(np.float32))
        d = self.cfg.detectors[det]
        if d.kind == "agipd_tiles":                  # number the stacked tiles
            view.set_labels([(c0, r0, f"tile {t}") for t, (r0, c0) in agipd.stack_origins(d.tiles).items()])

    def _draw_mosaic(self):
        if self.mosaic is None:
            return
        scans = [s for s in self.sorted_scans() if self.scans[s].overview]
        if not scans:
            self.mosaic.show_image(None)
            return
        s = self.mosaic.source.currentData()
        use = [s] if s in self.scans and self.scans[s].overview else scans
        n = np.array([self.scans[k].n for k in use], float)
        tiles = {}
        for t in range(agipd.N_TILES):
            have = [(self.scans[k].overview[t], w) for k, w in zip(use, n) if t in self.scans[k].overview]
            if have:
                tiles[t] = sum(a * w for a, w in have) / sum(w for _, w in have)
        self.mosaic.show_image(agipd.assemble(tiles))

    # ============================================================ export
    def export_csv(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            f.write(f"# normalisation: {self.cmb_norm.currentText()}; MAD cut {self.sp_mad.value()}; "
                    f"min diode/median {self.sp_min.value()}\n")
            for det, view in self.views.items():
                f.write(f"# {det} ROIs: {view.panel.to_json().replace(chr(10), ' ')}\n")
            w.writerow(["roi", "delay_fs", "signal", "signal_err", "norm", "norm_err", "n_shots"])
            for label, s in self.series.items():
                t, y, e, n = s["raw"]
                _, yn, en, _ = s["norm"]
                for row in zip(t, y, e, yn, en, n):
                    w.writerow([label, *[f"{v:.6g}" for v in row]])

    def _export(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export", "xtg_delay_series.csv", "CSV (*.csv)")
        if path:
            self.export_csv(path)
            self.log(f"wrote {path}")


# ---------------------------------------------------------------- entry point
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="XTG analysis GUI for European XFEL MID")
    ap.add_argument("--config", help="YAML/JSON file with the source names (see --write-config)")
    ap.add_argument("--write-config", metavar="FILE", help="write the default (placeholder) config and exit")
    ap.add_argument("--backend", choices=["simulated", "extra_data"], default="extra_data",
                    help="extra_data = real runs on Maxwell (default); simulated = fake data for testing")
    ap.add_argument("--proposal", type=int, help="proposal number (7016 / 11456 pick their preset)")
    ap.add_argument("--preset", choices=list(PRESETS),
                    help="7016 = previous beamtime (AGIPD, HIREX/DES, XGM), 11456 = this beamtime "
                         "(default), custom = --config file")
    ap.add_argument("--scans", help="scans to load at start, e.g. '20-40' (default: none, press Load)")
    ap.add_argument("--light", action="store_true", help="start in light mode")
    ap.add_argument("--no-crl", action="store_true", help="hide the CRL pointing page")
    ap.add_argument("--no-spec", action="store_true", help="hide the upstream spectrometer page")
    args = ap.parse_args(argv)

    from .config import preset_of_file
    preset = args.preset or ((preset_of_file(args.config) or "custom") if args.config else
                             str(args.proposal) if str(args.proposal) in ("7016", "11456") else
                             "custom" if args.proposal else "11456")
    try:
        cfg = config_for(preset, args.config)
    except (OSError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.write_config:
        cfg.save(args.write_config)
        print(f"wrote {args.write_config}")
        return 0
    if args.proposal is not None:
        cfg.proposal = args.proposal
    if args.no_crl:
        cfg.crl_enabled = False
    if args.no_spec:
        cfg.spec_enabled = False
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    theme.apply(app, "light" if args.light else "dark")
    win = XTGMidWindow(cfg, args.backend)
    if args.scans:
        win.ed_scans.setText(args.scans)
    win.show()
    if args.scans and (args.backend == "simulated" or cfg.proposal):   # nothing is loaded unless asked
        QtCore.QTimer.singleShot(0, lambda: win.load(parse_scans(args.scans)))
    return app.exec_() if hasattr(app, "exec_") else app.exec()
