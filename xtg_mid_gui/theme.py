"""Dark / light themes: palettes, Qt stylesheet and pyqtgraph styling.

``apply(app, "dark"|"light")`` switches the whole application; plots register
with :func:`style_plot` so :func:`restyle` can recolour them in place.
"""
from __future__ import annotations

import pyqtgraph as pg

PALETTES = {
    "dark": dict(bg="#0e1014", panel="#151820", card="#1b1f29", card_hi="#232836", border="#2b3140",
                 text="#e7eaf0", muted="#8d96a8", accent="#5b8cff", accent_hi="#7aa2ff",
                 warn="#f5a524", warn_bg="#3a2a10", plot_bg="#151820", axis="#9aa3b2", neutral=(170, 176, 190)),
    "light": dict(bg="#eef0f4", panel="#ffffff", card="#f7f8fa", card_hi="#e9edf3", border="#d3d8e1",
                  text="#1b1f29", muted="#5d6677", accent="#3a6ff7", accent_hi="#5b8cff",
                  warn="#a35f00", warn_bg="#fff1d6", plot_bg="#ffffff", axis="#4a5262", neutral=(110, 116, 128)),
}

# data colours, readable on both backgrounds
PUMP = (240, 130, 40)
PROBE = (50, 140, 240)
ROI_COLORS = [(230, 73, 128), (18, 170, 130), (132, 94, 247), (225, 160, 0),
              (34, 139, 230), (240, 101, 60), (100, 180, 60), (200, 80, 200)]

_mode = "dark"
_plots: list = []          # registered PlotItems
_widgets: list = []        # registered GraphicsLayoutWidgets


def mode() -> str:
    return _mode


def pal() -> dict:
    return PALETTES[_mode]


def qss(p: dict) -> str:
    return f"""
* {{ font-family: "Inter", "Segoe UI", "Helvetica Neue", "DejaVu Sans", Arial; font-size: 12px; color: {p['text']}; }}
QMainWindow, QWidget#root {{ background: {p['bg']}; }}
QWidget#sidebar, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {p['panel']}; border: none; }}
QFrame#card {{ background: {p['card']}; border: 1px solid {p['border']}; border-radius: 10px; }}
QLabel#cardTitle {{ color: {p['muted']}; font-size: 11px; font-weight: 600; letter-spacing: 1px; }}
QLabel#appTitle {{ font-size: 17px; font-weight: 700; }}
QLabel#muted {{ color: {p['muted']}; }}
QLabel#pill {{ background: {p['card_hi']}; border: 1px solid {p['border']}; border-radius: 11px; padding: 3px 10px; }}
QLabel#tbc {{ background: {p['warn_bg']}; border: 1px solid {p['warn']}; border-radius: 8px;
             padding: 6px 10px; color: {p['warn']}; font-weight: 600; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {p['panel']}; border: 1px solid {p['border']}; border-radius: 6px; padding: 4px 6px;
    selection-background-color: {p['accent']}; selection-color: white; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border: 1px solid {p['accent']}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {p['card']}; color: {p['text']}; border: 1px solid {p['border']};
    selection-background-color: {p['accent']}; selection-color: white; outline: none; }}
QComboBox QAbstractItemView::item {{ color: {p['text']}; min-height: 22px; padding: 0 6px; }}
QComboBox QAbstractItemView::item:selected {{ background: {p['accent']}; color: white; }}
QCheckBox:disabled, QLabel:disabled, QComboBox:disabled {{ color: {p['muted']}; }}
QCheckBox::indicator:disabled {{ background: {p['card_hi']}; border: 1px solid {p['border']}; }}
QPushButton {{ background: {p['card_hi']}; border: 1px solid {p['border']}; border-radius: 7px; padding: 5px 11px; }}
QPushButton:hover {{ border: 1px solid {p['accent']}; }}
QPushButton:checked {{ background: {p['accent']}; color: white; border: 1px solid {p['accent']}; }}
QPushButton#primary {{ background: {p['accent']}; border: 1px solid {p['accent']}; color: white; font-weight: 600; }}
QPushButton#primary:hover {{ background: {p['accent_hi']}; }}
QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border-radius: 4px; border: 1px solid {p['border']}; background: {p['panel']}; }}
QCheckBox::indicator:checked {{ background: {p['accent']}; border: 1px solid {p['accent']}; }}
QTabWidget::pane {{ border: 1px solid {p['border']}; border-radius: 10px; background: {p['panel']}; top: -1px; }}
QTabBar::tab {{ background: transparent; color: {p['muted']}; padding: 7px 14px; border: none; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {p['text']}; border-bottom: 2px solid {p['accent']}; }}
QTableWidget {{ background: {p['panel']}; border: 1px solid {p['border']}; border-radius: 6px; gridline-color: {p['border']}; }}
QHeaderView::section {{ background: {p['card']}; color: {p['muted']}; border: none; border-bottom: 1px solid {p['border']}; padding: 4px; }}
QTableWidget::item:selected {{ background: {p['card_hi']}; color: {p['text']}; }}
QPlainTextEdit {{ background: {p['panel']}; border: 1px solid {p['border']}; border-radius: 6px; color: {p['muted']};
                  font-family: "JetBrains Mono", "Menlo", "DejaVu Sans Mono", monospace; font-size: 11px; }}
QSplitter::handle {{ background: {p['bg']}; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: {p['border']}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QStatusBar {{ background: {p['panel']}; color: {p['muted']}; }}
QToolTip {{ background: {p['card_hi']}; color: {p['text']}; border: 1px solid {p['border']}; }}
"""


def apply(app, new_mode: str | None = None):
    global _mode
    if new_mode:
        _mode = new_mode
    p = pal()
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True,
                        background=p["plot_bg"], foreground=p["axis"])
    app.setStyleSheet(qss(p))
    restyle()


def register_widget(w):
    _widgets.append(w)
    w.setBackground(pal()["plot_bg"])
    return w


def style_plot(p: pg.PlotItem, title=None, x=None, y=None):
    p._xtg = dict(title=title, x=x, y=y)
    if p not in _plots:
        _plots.append(p)
    _style(p)
    return p


def _style(p):
    c = pal()
    meta = p._xtg
    if meta["title"]:
        p.setTitle(f"<span style='color:{c['text']};font-size:10.5pt;font-weight:600'>{meta['title']}</span>")
    for ax, lab in (("bottom", meta["x"]), ("left", meta["y"])):
        a = p.getAxis(ax)
        a.setPen(pg.mkPen(c["border"]))
        a.setTextPen(pg.mkPen(c["axis"]))
        _no_si_prefix(a)
        if lab:
            a.setLabel(lab, color=c["axis"])
    p.showGrid(x=True, y=True, alpha=0.15)
    if p.legend is not None:
        p.legend.setLabelTextColor(c["text"])
        p.legend.setBrush(pg.mkBrush(c["panel"] + "cc"))
        p.legend.setPen(pg.mkPen(c["border"]))


def _no_si_prefix(axis):
    """Plain tick values (no hidden x1000 'm' scaling).  pyqtgraph 0.14's
    updateAutoSIPrefix ignores enableAutoSIPrefix(False), so override it."""
    def update(ax=axis):
        ax.autoSIPrefixScale, ax.labelUnitPrefix = 1.0, ""
        ax._updateLabel()
    axis.autoSIPrefix = False
    axis.updateAutoSIPrefix = update
    update()


def set_title(p, title):
    p._xtg["title"] = title
    _style(p)


def restyle():
    for w in list(_widgets):
        try:
            w.setBackground(pal()["plot_bg"])
        except RuntimeError:          # widget deleted
            _widgets.remove(w)
    for p in list(_plots):
        try:
            _style(p)
        except RuntimeError:
            _plots.remove(p)


def neutral(alpha=255):
    return (*pal()["neutral"], alpha)


# ---------------------------------------------------------------- image colour maps
def _lin(*hexes):
    cols = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in hexes]
    return pg.ColorMap(np_linspace(len(cols)), cols)


def np_linspace(n):
    return [i / (n - 1) for i in range(n)]


# name -> factory.  The "white ..." maps send low counts to white (good on a light page).
COLORMAPS = {
    "viridis": lambda: pg.colormap.get("viridis"),
    "inferno": lambda: pg.colormap.get("inferno"),
    "magma": lambda: pg.colormap.get("magma"),
    "plasma": lambda: pg.colormap.get("plasma"),
    "cividis": lambda: pg.colormap.get("cividis"),
    "grey (black = low)": lambda: _lin("#000000", "#ffffff"),
    "white → blue": lambda: _lin("#ffffff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"),
    "white → viridis": lambda: _lin("#ffffff", "#b5de2b", "#35b779", "#26828e", "#3e4989", "#440154"),
    "white → red": lambda: _lin("#ffffff", "#fee391", "#fe9929", "#cc4c02", "#662506"),
    "grey (white = low)": lambda: _lin("#ffffff", "#000000"),
}
DEFAULT_CMAP = {"dark": "viridis", "light": "white → blue"}


def colormap(name):
    return COLORMAPS.get(name, COLORMAPS["viridis"])()


def error_bars(t, y, e, color, alpha=230, width=1.5):
    """Error bars (±e) with short caps, drawn opaque enough to be seen next to the markers."""
    import numpy as np
    t, y, e = (np.asarray(a, float) for a in (t, y, e))
    ts = np.unique(t[np.isfinite(t)])
    gap = np.min(np.diff(ts)) if len(ts) > 1 else (abs(ts[0]) or 1.0) if len(ts) else 1.0
    span = (ts[-1] - ts[0]) if len(ts) > 1 else gap
    beam = min(0.4 * gap, 0.015 * span) if span > 0 else 0.0
    return pg.ErrorBarItem(x=t, y=y, height=2 * np.nan_to_num(e), beam=beam,
                           pen=pg.mkPen(tuple(color[:3]) + (alpha,), width=width))


def link_error_bars(curve, eb, plot=None, y=None, e=None):
    """Hide / show the error bars with their curve (legend click); with `plot`, the y axis
    is then rescaled to the curves still shown (error bars included)."""
    import numpy as np
    curve._eb = eb
    curve._rescales = plot is not None
    if y is not None:
        y = np.asarray(y, float)
        e = np.zeros_like(y) if e is None else np.nan_to_num(np.asarray(e, float))
        curve._ylims = (y - e, y + e)

    def shown():
        eb.setVisible(curve.isVisible())
        if plot is not None:
            rescale_y(plot)
    curve.visibleChanged.connect(shown)
    return curve


def rescale_y(plot):
    """y range of `plot` to its visible curves (error bars included); x is left alone."""
    import numpy as np
    lo, hi = [], []
    for it in plot.listDataItems():
        if not it.isVisible():
            continue
        lims = getattr(it, "_ylims", None)
        if lims is None:
            yd = it.yData
            if yd is None:
                continue
            lims = (np.asarray(yd, float), np.asarray(yd, float))
        lo.append(np.ravel(lims[0]))
        hi.append(np.ravel(lims[1]))
    if not lo:
        return
    lo, hi = np.concatenate(lo), np.concatenate(hi)
    ok = np.isfinite(lo) & np.isfinite(hi)
    if not ok.any():
        return
    a, b = float(lo[ok].min()), float(hi[ok].max())
    if b <= a:
        a, b = a - 0.5, b + 0.5
    plot.getViewBox().setYRange(a, b, padding=0.05)


def rescale_on_hide(plot):
    """Every curve of `plot` hidden / shown from the legend rescales the y axis."""
    for it in plot.listDataItems():
        if not getattr(it, "_rescales", False):
            it._rescales = True
            if getattr(it, "_eb", None) is None:
                it.visibleChanged.connect(lambda p=plot: rescale_y(p))
