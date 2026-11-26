"""Reduce the per-train images of a run to the few numbers the GUI needs, chunk by
chunk, so a run never has to sit in memory as a full image stack.

Only the *reference run* keeps its images (to place the ROIs).  For every other run
the backend feeds its images, a chunk of trains at a time, to an :class:`ImageReducer`
and keeps the resulting :class:`ImageStats`:

* the run's mean image (all trains),
* per train and per ROI: the sum of the ROI pixels (NaN pixels count 0) and the
  number of pixels that are ever finite -> the same background-subtracted signal as
  :func:`analysis.roi_signal` on the full images,
* per train and per signal ROI: centre of mass and max position, with and without
  the background level (for the pointing checks).

A few kB per train instead of MB.  When a ROI moves, the run is reduced again with
the new geometry (see the GUI); geometries reduced before are kept, so moving a ROI
back costs nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis import roi_crop, roi_position

MODES = ("com", "max")


def geom_key(g):
    """ROI geometry as a hashable key, rounded so float noise does not matter."""
    return tuple(round(float(v), 3) if isinstance(v, (int, float, np.floating)) else v for v in g)


@dataclass
class ImageStats:
    """What is kept of one detector of a run that is not the reference run."""
    mean: np.ndarray                                  # (ny, nx) mean image over the run's trains
    sums: dict = field(default_factory=dict)          # geom key -> (n,) per-train ROI sum
    npx: dict = field(default_factory=dict)           # geom key -> pixels ever finite in the ROI
    pos: dict = field(default_factory=dict)           # (geom key, mode, bkg key) -> (x, y) per train

    def has(self, geom, bkg_geoms=()) -> bool:
        return geom_key(geom) in self.sums and all(geom_key(b) in self.sums for b in bkg_geoms)

    def signal(self, geom, bkg_geoms=()):
        """Background-subtracted per-train ROI signal (as analysis.roi_signal), or None
        if this geometry has not been reduced yet."""
        if not self.has(geom, bkg_geoms):
            return None
        k = geom_key(geom)
        s, npx = self.sums[k], self.npx[k]
        if npx == 0:
            return np.full(len(s), np.nan)
        tot, n = np.zeros(len(s)), 0
        for b in bkg_geoms:
            kb = geom_key(b)
            if self.npx[kb]:
                tot = tot + self.sums[kb]
                n += self.npx[kb]
        return s - npx * tot / n if n else s.copy()

    def position(self, geom, mode, bkg_geoms=()):
        """Per-train (x, y) of the centre of mass / max inside the ROI, or None."""
        return self.pos.get((geom_key(geom), mode, tuple(geom_key(b) for b in bkg_geoms)))

    def merge(self, other: "ImageStats"):
        """Add the geometries of a later reduction of the same run (same trains)."""
        self.sums.update(other.sums)
        self.npx.update(other.npx)
        self.pos.update(other.pos)

    @property
    def nbytes(self) -> int:
        return int(self.mean.nbytes + sum(a.nbytes for a in self.sums.values())
                   + sum(x.nbytes + y.nbytes for x, y in self.pos.values()))


class ImageReducer:
    """Feed per-train images chunk by chunk with add(); finish(trains) returns the
    ImageStats on those trains (NaN for a train that never came)."""

    def __init__(self, signal_geoms=(), bkg_geoms=(), keep_images=False):
        self.sig = list(dict.fromkeys(geom_key(g) for g in signal_geoms))
        self.bkg = list(dict.fromkeys(geom_key(g) for g in bkg_geoms))
        self.bkey = tuple(self.bkg)
        self.keep_images = keep_images
        self._t, self._imgs = [], []
        self._sums = {g: [] for g in self.sig + self.bkg}
        self._good = {g: None for g in self.sig + self.bkg}
        self._pos = {}
        self._msum = self._mcnt = None

    def add(self, tids, imgs):
        imgs = np.asarray(imgs, np.float32)
        tids = np.asarray(tids)
        if len(tids) == 0:
            return
        self._t.append(tids)
        fin = np.isfinite(imgs)
        s = np.where(fin, imgs, 0).sum(axis=0, dtype=np.float64)
        c = fin.sum(axis=0)
        self._msum = s if self._msum is None else self._msum + s
        self._mcnt = c if self._mcnt is None else self._mcnt + c
        lvl_tot, lvl_n = np.zeros(len(imgs)), 0
        for g in self.sig + self.bkg:
            ys, xs, m = roi_crop(imgs.shape[1:], g)
            crop = imgs[:, ys, xs][:, m].astype(np.float64)
            self._sums[g].append(np.nansum(crop, axis=1) if crop.size else np.full(len(imgs), np.nan))
            good = np.isfinite(crop).any(axis=0) if crop.size else np.zeros(0, bool)
            self._good[g] = good if self._good[g] is None else (self._good[g] | good)
            if g in self.bkg and good.any():
                lvl_tot += np.nansum(crop[:, good], axis=1)
                lvl_n += int(good.sum())
        lvl = lvl_tot / lvl_n if lvl_n else None
        for g in self.sig:
            for mode in MODES:
                for bk, level in (((), None), (self.bkey, lvl)):
                    if bk and level is None:
                        continue
                    x, y = roi_position(imgs, g, mode, level)
                    self._pos.setdefault((g, mode, bk), ([], []))
                    self._pos[(g, mode, bk)][0].append(x)
                    self._pos[(g, mode, bk)][1].append(y)
        if self.keep_images:
            self._imgs.append(imgs)

    def finish(self, trains=None):
        """(ImageStats, images or None) on `trains` (default: all trains seen, sorted)."""
        t = np.concatenate(self._t) if self._t else np.array([], int)
        trains = np.unique(t) if trains is None else np.asarray(trains)
        idx = _index_of(t, trains)
        hit = idx >= 0

        def align(parts):
            a = np.concatenate(parts) if parts else np.array([])
            out = np.full(len(trains), np.nan)
            out[hit] = a[idx[hit]]
            return out

        with np.errstate(invalid="ignore", divide="ignore"):
            mean = (np.where(self._mcnt > 0, self._msum / np.maximum(self._mcnt, 1), np.nan).astype(np.float32)
                    if self._msum is not None else np.zeros((1, 1), np.float32))
        st = ImageStats(mean)
        for g, parts in self._sums.items():
            st.sums[g] = align(parts)
            st.npx[g] = int(self._good[g].sum()) if self._good[g] is not None else 0
        for k, (xs, ys) in self._pos.items():
            st.pos[k] = (align(xs), align(ys))
        images = None
        if self.keep_images and self._imgs:
            a = np.concatenate(self._imgs)
            images = np.full((len(trains),) + a.shape[1:], np.nan, np.float32)
            images[hit] = a[idx[hit]]
        return st, images


def _index_of(t, trains):
    """Position in t of each train in trains (-1 if absent); t need not be sorted."""
    t = np.asarray(t)
    if len(t) == 0:
        return np.full(len(trains), -1)
    order = np.argsort(t, kind="stable")
    ts = t[order]
    i = np.clip(np.searchsorted(ts, trains), 0, len(ts) - 1)
    return np.where(ts[i] == trains, order[i], -1)


def reduce_images(images, trains, signal_geoms, bkg_geoms, chunk=32):
    """Full (n, ny, nx) stack -> ImageStats, in chunks (used by the simulation and tests)."""
    r = ImageReducer(signal_geoms, bkg_geoms)
    for i in range(0, len(images), chunk):
        r.add(trains[i:i + chunk], images[i:i + chunk])
    return r.finish(trains)[0]
