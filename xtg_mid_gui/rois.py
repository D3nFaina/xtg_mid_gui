"""ROIs (elliptical by default, or rectangular) drawn on a detector image, and the
per-detector table that adds, deletes, renames, reshapes and toggles them."""
from __future__ import annotations

import json
import math

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from . import theme

# Default ROIs as fractions of the image: name -> (row, col, half-height, half-width, background?)
DEFAULT_ROIS = {
    "jf1": [("TG +1", 0.5, 0.25, 0.14, 0.09, False), ("Bragg", 0.5, 0.5, 0.16, 0.10, False),
            ("TG -1", 0.5, 0.75, 0.14, 0.09, False), ("Background", 0.15, 0.5, 0.08, 0.25, True)],
    "jf2": [("TG +2", 0.35, 0.3, 0.14, 0.09, False), ("Bragg 2", 0.55, 0.55, 0.18, 0.12, False),
            ("TG -2", 0.75, 0.8, 0.14, 0.09, False), ("Background", 0.12, 0.75, 0.07, 0.18, True)],
    # p7016 AGIPD stack of tiles 8, 10, 15 (top to bottom)
    "agipd_sub": [("TG +1 (tile 8)", 0.175, 0.586, 0.06, 0.06, False), ("TG -1 (tile 10)", 0.485, 0.39, 0.06, 0.06, False),
                  ("Bragg (tile 15)", 0.84, 0.488, 0.07, 0.07, False), ("Background", 0.84, 0.8, 0.06, 0.06, True)],
}
GENERIC = [(f"ROI {i + 1}", 0.5, (i + 1) / 5, 0.12, 0.07, i == 3) for i in range(4)]


SHAPES = ("ellipse", "rectangle")


class RoiSpec(QtCore.QObject):
    """One ROI, elliptical (default) or rectangular: movable, resizable and
    rotatable with the mouse.  Each ROI has its own shape."""
    changed = QtCore.Signal()

    def __init__(self, name, center, radii, color, angle=0.0, background=False, enabled=True, shape="ellipse"):
        super().__init__()
        self.name, self.color = name, tuple(color)
        self.background, self.enabled = background, enabled
        self.shape = shape if shape in SHAPES else "ellipse"
        self.label = pg.TextItem(name, color=self.color, anchor=(0, 1))
        self._make_item(center, radii, angle)

    def _make_item(self, center, radii, angle):
        (cx, cy), (rx, ry) = center, radii
        # pyqtgraph rotates an ROI about its pos (corner), so place the corner such
        # that the centre stays at (cx, cy) for any angle
        a = math.radians(angle)
        pos = (cx - (rx * math.cos(a) - ry * math.sin(a)), cy - (rx * math.sin(a) + ry * math.cos(a)))
        kw = dict(pen=self._pen(), hoverPen=pg.mkPen((255, 255, 255), width=2), angle=angle)
        if self.shape == "ellipse":
            item = pg.EllipseROI(pos, (2 * rx, 2 * ry), **kw)
        else:
            item = pg.ROI(pos, (2 * rx, 2 * ry), **kw)
            item.addScaleHandle([1, 1], [0, 0])
            item.addScaleHandle([0, 0], [1, 1])
            item.addScaleHandle([1, 0.5], [0, 0.5])
            item.addScaleHandle([0.5, 1], [0.5, 0])
            item.addRotateHandle([1, 0], [0.5, 0.5])
        item.setZValue(10)
        self.label.setParentItem(item)
        item.sigRegionChangeFinished.connect(lambda *_: self.changed.emit())
        self.item = item

    def set_shape(self, shape):
        """Swap ellipse <-> rect keeping centre, size, angle.  Returns the old item
        (the caller removes it from the view and adds the new one)."""
        if shape == self.shape or shape not in SHAPES:
            return None
        cx, cy, rx, ry, ang, _ = self.geometry()
        old = self.item
        self.label.setParentItem(None)
        self.shape = shape
        self._make_item((cx, cy), (rx, ry), ang)
        self.refresh()
        return old

    def _pen(self):
        style = QtCore.Qt.DashLine if self.background else QtCore.Qt.SolidLine
        return pg.mkPen(self.color, width=2, style=style)

    def refresh(self):
        self.item.setPen(self._pen())
        self.label.setText(self.name + ("  (bkg)" if self.background else ""))
        self.label.setColor(self.color)
        self.item.setVisible(self.enabled)

    def geometry(self):
        """(cx, cy, rx, ry, angle, shape) in image pixels (x = column, y = row);
        rx, ry are semi-axes (ellipse) or half-widths (rect)."""
        w, h = self.item.size()
        c = self.item.mapToParent(QtCore.QPointF(w / 2, h / 2))
        return (round(c.x(), 2), round(c.y(), 2), round(abs(w) / 2, 2), round(abs(h) / 2, 2),
                round(self.item.angle(), 2), self.shape)

    def to_dict(self):
        cx, cy, rx, ry, a, shape = self.geometry()
        return dict(name=self.name, shape=shape, center=[cx, cy], radii=[rx, ry], angle=a,
                    background=self.background, enabled=self.enabled, color=list(self.color))

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["center"], d["radii"], d.get("color", theme.ROI_COLORS[0]),
                   d.get("angle", 0.0), d.get("background", False), d.get("enabled", True),
                   d.get("shape", "ellipse"))


class RoiPanel(QtWidgets.QWidget):
    """Table of the ROIs of ONE detector: on/off, editable name, background flag.
    Shape column: ellipse (default) or rect, per ROI.
    Buttons: + ROI, Delete, Save, Load.  Emits changed() after any edit."""
    changed = QtCore.Signal()

    def __init__(self, det: str, label: str, view: pg.ViewBox, parent=None):
        super().__init__(parent)
        self.det, self.view = det, view
        self.rois: list[RoiSpec] = []
        self.shape = (100, 160)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["on", "name (double-click to edit)", "shape", "bkg"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(140)
        self.table.itemChanged.connect(self._item_changed)
        self.table.cellClicked.connect(self._cell_clicked)
        lay.addWidget(self.table)
        row = QtWidgets.QHBoxLayout()
        self.b_add, self.b_del = QtWidgets.QPushButton("+ ROI"), QtWidgets.QPushButton("Delete")
        self.b_save, self.b_load = QtWidgets.QPushButton("Save"), QtWidgets.QPushButton("Load")
        for b in (self.b_add, self.b_del, self.b_save, self.b_load):
            row.addWidget(b)
        lay.addLayout(row)
        self.b_add.clicked.connect(lambda: self.spawn())
        self.b_del.clicked.connect(self.delete_selected)
        self.b_save.clicked.connect(self._save)
        self.b_load.clicked.connect(self._load)
        self._updating = False

    # ------------------------------------------------------------ management
    def next_color(self):
        used = {s.color for s in self.rois}
        for c in theme.ROI_COLORS:
            if c not in used:
                return c
        return theme.ROI_COLORS[len(self.rois) % len(theme.ROI_COLORS)]

    def add(self, spec: RoiSpec):
        self.rois.append(spec)
        self.view.addItem(spec.item)
        spec.refresh()
        spec.changed.connect(self.changed)
        self._rebuild()
        self.changed.emit()
        return spec

    def spawn(self, name=None, frac=None, background=False):
        """New ellipse; default: centre of the image, named 'ROI n'."""
        ny, nx = self.shape
        if frac is None:
            frac = (0.5, 0.5, 0.12, 0.08)
        fr, fc, hr, hc = frac
        if name is None:
            names = {s.name for s in self.rois}
            i = len(self.rois) + 1
            while f"ROI {i}" in names:
                i += 1
            name = f"ROI {i}"
        return self.add(RoiSpec(name, (fc * nx, fr * ny), (hc * nx, hr * ny), self.next_color(),
                                background=background))

    def defaults(self):
        self.clear()
        for name, fr, fc, hr, hc, bkg in DEFAULT_ROIS.get(self.det, GENERIC):
            self.spawn(name, (fr, fc, hr, hc), bkg)

    def remove(self, spec):
        self.rois.remove(spec)
        self.view.removeItem(spec.item)
        self._rebuild()
        self.changed.emit()

    def delete_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows and self.rois:
            rows = [len(self.rois) - 1]
        for r in rows:
            self.remove(self.rois[r])

    def clear(self):
        for s in list(self.rois):
            self.view.removeItem(s.item)
        self.rois.clear()
        self._rebuild()

    def set_shape(self, spec, shape):
        old = spec.set_shape(shape)
        if old is None:
            return
        self.view.removeItem(old)
        self.view.addItem(spec.item)
        self._rebuild()
        self.changed.emit()

    def rename(self, spec, name):
        spec.name = name
        spec.refresh()
        self._rebuild()
        self.changed.emit()

    # ------------------------------------------------------------ table
    def _rebuild(self):
        self._updating = True
        self.table.setRowCount(len(self.rois))
        for i, s in enumerate(self.rois):
            on = QtWidgets.QTableWidgetItem()
            on.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            on.setCheckState(QtCore.Qt.Checked if s.enabled else QtCore.Qt.Unchecked)
            pm = QtGui.QPixmap(10, 10)
            pm.fill(QtGui.QColor(*s.color))
            on.setIcon(QtGui.QIcon(pm))
            self.table.setItem(i, 0, on)
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(s.name))
            bk = QtWidgets.QTableWidgetItem()
            bk.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            bk.setCheckState(QtCore.Qt.Checked if s.background else QtCore.Qt.Unchecked)
            bk.setToolTip("background ROI: its mean level per pixel is subtracted from the other ROIs")
            sh = QtWidgets.QTableWidgetItem(("◯ " if s.shape == "ellipse" else "▭ ") + s.shape)
            sh.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            sh.setToolTip("click to switch this ROI between ellipse and rectangle")
            self.table.setItem(i, 2, sh)
            self.table.setItem(i, 3, bk)
        self._updating = False

    def _cell_clicked(self, row, col):
        if col == 2 and row < len(self.rois):
            s = self.rois[row]
            self.set_shape(s, "rectangle" if s.shape == "ellipse" else "ellipse")

    def _item_changed(self, item):
        if self._updating:
            return
        s = self.rois[item.row()]
        if item.column() == 0:
            s.enabled = item.checkState() == QtCore.Qt.Checked
        elif item.column() == 1:
            s.name = item.text().strip() or s.name
            if item.text() != s.name:
                QtCore.QTimer.singleShot(0, self._rebuild)
        elif item.column() == 3:
            s.background = item.checkState() == QtCore.Qt.Checked
        s.refresh()
        self.changed.emit()

    # ------------------------------------------------------------ queries
    def signal_rois(self):
        return [s for s in self.rois if s.enabled and not s.background]

    def background_geoms(self):
        return [s.geometry() for s in self.rois if s.enabled and s.background]

    # ------------------------------------------------------------ io
    def to_json(self):
        return json.dumps({"detector": self.det, "rois": [s.to_dict() for s in self.rois]}, indent=2)

    def to_dicts(self):
        return [s.to_dict() for s in self.rois]

    def set_dicts(self, dicts):
        """Replace the ROIs by these (RoiSpec.to_dict dicts); one changed() at the end."""
        self.blockSignals(True)
        try:
            self.clear()
            for r in dicts:
                self.add(RoiSpec.from_dict(r))
        finally:
            self.blockSignals(False)
        self.changed.emit()

    def from_json(self, txt):
        d = json.loads(txt)
        self.clear()
        for r in d["rois"]:
            self.add(RoiSpec.from_dict(r))

    def _save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save ROIs", f"rois_{self.det}.json", "JSON (*.json)")
        if path:
            with open(path, "w") as f:
                f.write(self.to_json())

    def _load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load ROIs", "", "JSON (*.json)")
        if path:
            with open(path) as f:
                self.from_json(f.read())
