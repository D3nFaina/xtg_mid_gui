"""Small Qt helpers that let the windows shrink to fit a 1280 px wide screen."""
from __future__ import annotations

from pyqtgraph.Qt import QtCore, QtWidgets


class ShrinkCombo(QtWidgets.QComboBox):
    """Combo box that asks for `pref` px when there is room but can be squeezed down
    to `least` px (the current text is then cut short) when the window is narrow."""

    def __init__(self, pref=200, least=110, parent=None):
        super().__init__(parent)
        self._pref, self._least = pref, least

    def sizeHint(self):
        s = super().sizeHint()
        return QtCore.QSize(max(s.width(), self._pref), s.height())

    def minimumSizeHint(self):
        s = super().minimumSizeHint()
        return QtCore.QSize(min(s.width(), self._least), s.height())


def elastic(label: QtWidgets.QLabel, tip=True) -> QtWidgets.QLabel:
    """Let a one-line label be cut short instead of forcing the window wider; the full
    text stays available as its tooltip."""
    label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, label.sizePolicy().verticalPolicy())
    if tip:
        label.setToolTip(label.text().strip())
    return label


def fit_to_screen(win: QtWidgets.QWidget, width: int, height: int):
    """resize() to (width, height), but no larger than the screen's free area
    (minus room for the title bar and frame)."""
    scr = win.screen() if hasattr(win, "screen") else None
    scr = scr or QtWidgets.QApplication.primaryScreen()
    if scr is not None:
        g = scr.availableGeometry()
        width, height = min(width, g.width() - 16), min(height, g.height() - 48)
    win.resize(width, height)
