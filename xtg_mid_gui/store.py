"""Per-run cache of reduced data, one HDF5 file per run.

Everything the GUI needs from a run, except the per-train images, is written once to
``reduced_<proposal>/run_<NNNN>.h5`` in the start folder (next to delays_<proposal>.csv):

* per train: trainIds, pump, probe, XGM, CRL pointing, scanned parameter,
  upstream-spectrometer spectra (compressed),
* per detector: the run's mean image and, for every ROI geometry ever reduced,
  the per-train ROI sum, its pixel count and the peak positions.

A run is read from the raw/proc files only when

* it has no cache file yet, or
* the reading settings changed (sources, masks, photonizing, AGIPD tiles ...):
  the ``fingerprint`` stored in the file no longer matches, or
* a ROI geometry is asked for that the file does not have yet (then only the new
  geometries are added; geometries reduced before stay, so moving a ROI back is free).

Loading a cached run keeps only the per-train numbers in memory; mean images and
spectra stay on disk and are read when a page needs them.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .delays import start_dir

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")      # GPFS / NFS on Maxwell: no flock
from .reduce import MODES, ImageStats, geom_key

VERSION = 1


# ---------------------------------------------------------------- where / which settings
def cache_dir(cfg, backend_name="extra_data") -> Path:
    tag = "sim" if backend_name == "simulated" else str(cfg.proposal or 0)
    return start_dir() / f"reduced_{tag}"


def run_path(cfg, backend_name, scan) -> Path:
    return cache_dir(cfg, backend_name) / f"run_{int(scan):04d}.h5"


def _src(s):
    return [s.source, s.key, s.pulses, s.reduce, s.scale, s.offset]


def fingerprint(cfg, backend_name="extra_data", extra=None) -> str:
    """Hash of every setting that changes what is read from a run (not the ROIs)."""
    d = dict(v=VERSION, backend=backend_name, extra=extra or {}, proposal=cfg.proposal, data=cfg.data,
             max_trains=cfg.max_trains, apply_mask=cfg.apply_mask, mask_per_train=cfg.mask_per_train,
             photonize=cfg.photonize, intensity_from=cfg.intensity_from, delay=_src(cfg.delay),
             xgm=_src(cfg.xgm), crl=[_src(getattr(cfg, k)) for k in cfg.CRL_KEYS] if cfg.crl_enabled else None)
    if cfg.photonize:
        d["photon"] = [cfg.photon_energy_keV, _src(cfg.photon_energy)]
    if cfg.intensity_from == "diodes":
        d["diodes"] = [_src(cfg.pump), _src(cfg.probe)]
    else:
        d["arm_specs"] = [cfg.pump_spec, cfg.probe_spec]
    d["detectors"] = {n: [_src(det.data), det.kind, list(det.tiles), det.mask_key]
                      for n, det in sorted(cfg.detectors.items()) if det.kind != "agipd_overview"}
    if cfg.spec_enabled or cfg.intensity_from == "spectrometers":
        d["spectrometers"] = {n: [_src(sp.data), sp.frame, list(sp.px), sp.mask_key]
                              for n, sp in sorted(cfg.spectrometers.items())}
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def param_tag(cfg) -> str:
    return f"{cfg.param.source}|{cfg.param.key}" if cfg.param_enabled and cfg.param.source else ""


def _gid(key) -> str:
    return hashlib.sha1(repr(key).encode()).hexdigest()[:12]


def _bid(bkey) -> str:
    return "nobkg" if not bkey else "b" + _gid(tuple(bkey))


# ---------------------------------------------------------------- status
def status(path, fp, rois=None) -> str:
    """'ok' (everything there), 'rois' (settings match, some ROI geometries missing) or
    'none' (no usable file: read the run)."""
    path = Path(path)
    if not path.is_file():
        return "none"
    import h5py
    try:
        with h5py.File(path, "r") as f:
            if f.attrs.get("fingerprint") != fp or not f.attrs.get("complete", False):
                return "none"
            for det, (sig, bkg) in (rois or {}).items():
                if f"det/{det}" not in f:
                    return "none"
                if missing_geoms(f, det, sig, bkg)[0]:
                    return "rois"
    except OSError:
        return "none"
    return "ok"


def missing_geoms(f, det, sig, bkg):
    """(signal geometries, background geometries) of a detector not in the open file f
    (a signal geometry also counts as missing if its peak positions for this set of
    background ROIs are not there)."""
    roi = f.get(f"det/{det}/roi")
    have = set(roi.keys()) if roi is not None else set()
    bkey = tuple(geom_key(b) for b in bkg)
    bid = _bid(bkey)
    miss_b = [b for b in bkg if _gid(geom_key(b)) not in have]
    miss_s = []
    for g in sig:
        gid = _gid(geom_key(g))
        if gid not in have:
            miss_s.append(g)
        elif bkg and f"{gid}/pos/{bid}" not in roi:
            miss_s.append(g)
    if miss_b:
        miss_s = list(sig)           # new background: positions with background need every signal ROI
    return miss_s, miss_b


# ---------------------------------------------------------------- write
def _atomic(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".h5.tmp", dir=path.parent)
    os.close(fd)
    return tmp


def _put(g, name, a, compress=False):
    if name in g:
        del g[name]
    a = np.asarray(a)
    kw = dict(compression="gzip", compression_opts=1, shuffle=True) if compress and a.size > 1000 else {}
    g.create_dataset(name, data=a, **kw)


def _write_stats(f, det, st: ImageStats):
    g = f.require_group(f"det/{det}")
    if st.mean is not None and "mean" not in g:
        _put(g, "mean", np.asarray(st.mean, np.float32), compress=True)
    roi = g.require_group("roi")
    for k, s in st.sums.items():
        r = roi.require_group(_gid(k))
        r.attrs["geom"] = json.dumps(list(k))
        r.attrs["npx"] = int(st.npx.get(k, 0))
        _put(r, "sum", np.asarray(s, np.float64))
    for (k, mode, bkey), (x, y) in st.pos.items():
        r = roi.require_group(_gid(k))
        if "geom" not in r.attrs:
            r.attrs["geom"] = json.dumps(list(k))
            r.attrs["npx"] = int(st.npx.get(k, 0))
        p = r.require_group(f"pos/{_bid(bkey)}")
        p.attrs["bkg"] = json.dumps([list(b) for b in bkey])
        _put(p, f"{mode}_x", np.asarray(x, np.float32))
        _put(p, f"{mode}_y", np.asarray(y, np.float32))


def write_run(path, d, fp, stats=None, ptag=""):
    """Write a new cache file for ScanData d (stats: {detector: ImageStats}, default
    d.stats).  Written to a temporary file, then renamed: a half-written file is never seen."""
    import h5py
    stats = d.stats if stats is None else stats
    tmp = _atomic(path)
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs.update(version=VERSION, fingerprint=fp, scan=int(d.scan), complete=False,
                           delay_data=float(d.delay_data if np.isfinite(d.delay_data) else d.delay),
                           photon_energy=float(d.photon_energy), notes=json.dumps(list(d.notes)),
                           masks=json.dumps(d.masks), param_tag=ptag)
            _put(f, "trains", np.asarray(d.train))
            _put(f, "pump", np.asarray(d.pump, np.float64))
            _put(f, "probe", np.asarray(d.probe, np.float64))
            if d.xgm is not None:
                _put(f, "xgm", np.asarray(d.xgm, np.float64))
            if d.param is not None:
                _put(f, "param", np.asarray(d.param, np.float64))
            for k, v in d.crl.items():
                _put(f, f"crl/{k}", np.asarray(v, np.float64))
            for k, v in d.spectra.items():
                _put(f, f"spectra/{k}", np.asarray(v, np.float32), compress=True)
                _put(f, f"spec_axis/{k}", np.asarray(d.spec_axis[k], np.float64))
            for det, st in stats.items():
                _write_stats(f, det, st)
            f.attrs["complete"] = True
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    _MEANS.drop(path)


def add_stats(path, stats):
    """Add ROI geometries to an existing cache file."""
    import h5py
    with h5py.File(path, "a") as f:
        for det, st in stats.items():
            _write_stats(f, det, st)


def set_param(path, param, ptag):
    import h5py
    with h5py.File(path, "a") as f:
        if param is None:
            if "param" in f:
                del f["param"]
        else:
            _put(f, "param", np.asarray(param, np.float64))
        f.attrs["param_tag"] = ptag


def get_param_tag(path) -> str:
    import h5py
    with h5py.File(path, "r") as f:
        return str(f.attrs.get("param_tag", ""))


# ---------------------------------------------------------------- read (lazy)
class _MeanCache:
    """Small LRU of mean images read from cache files (the image pane asks for them often)."""

    def __init__(self, n=16):
        self.n, self.d = n, OrderedDict()

    def get(self, path, det):
        st = os.stat(path)                   # a file rewritten by a worker process is read again
        key = (str(path), det, st.st_mtime_ns, st.st_size, st.st_ino)
        if key in self.d:
            self.d.move_to_end(key)
            return self.d[key]
        import h5py
        with h5py.File(path, "r") as f:
            a = f[f"det/{det}/mean"][()]
        self.d[key] = a
        while len(self.d) > self.n:
            self.d.popitem(last=False)
        return a

    def drop(self, path):
        for k in [k for k in self.d if k[0] == str(path)]:
            del self.d[k]


_MEANS = _MeanCache()


class MeanLoader:
    def __init__(self, path, det):
        self.path, self.det = str(path), det

    def __call__(self):
        return _MEANS.get(self.path, self.det)


class LazySpectra(Mapping):
    """{spectrometer: (n, npx) spectra} read from the cache file on access (not kept)."""
    lazy = True

    def __init__(self, path, names):
        self.path, self.names = str(path), list(names)

    def __getitem__(self, k):
        if k not in self.names:
            raise KeyError(k)
        import h5py
        with h5py.File(self.path, "r") as f:
            return f[f"spectra/{k}"][()]

    def __iter__(self):
        return iter(self.names)

    def __len__(self):
        return len(self.names)


def load_light(path, detectors=None):
    """ScanData of a cached run: per-train numbers in memory, mean images and spectra on disk."""
    import h5py
    from .backends import ScanData
    with h5py.File(path, "r") as f:
        a = f.attrs

        def opt(name):
            return f[name][()] if name in f else None
        stats = {}
        for det in f.get("det", {}):
            if detectors is not None and det not in detectors:
                continue
            g = f[f"det/{det}"]
            st = ImageStats(None, loader=MeanLoader(path, det) if "mean" in g else None)
            for r in g.get("roi", {}).values():
                k = geom_key(tuple(json.loads(r.attrs["geom"])))
                if "sum" in r:
                    st.sums[k] = r["sum"][()]
                st.npx[k] = int(r.attrs.get("npx", 0))
                for p in r.get("pos", {}).values():
                    bkey = tuple(geom_key(tuple(b)) for b in json.loads(p.attrs["bkg"]))
                    for mode in MODES:
                        if f"{mode}_x" in p:
                            st.pos[(k, mode, bkey)] = (p[f"{mode}_x"][()].astype(float),
                                                       p[f"{mode}_y"][()].astype(float))
            stats[det] = st
        spec_names = list(f["spectra"]) if "spectra" in f else []
        d = ScanData(int(a["scan"]), float(a["delay_data"]), {}, f["pump"][()], f["probe"][()],
                     {k: v[()] for k, v in f["crl"].items()} if "crl" in f else {},
                     f["trains"][()], spectra=LazySpectra(path, spec_names),
                     spec_axis={k: f[f"spec_axis/{k}"][()] for k in spec_names},
                     xgm=opt("xgm"), param=opt("param"), delay_data=float(a["delay_data"]),
                     masks=json.loads(a.get("masks", "{}")), photon_energy=float(a.get("photon_energy", np.nan)),
                     notes=json.loads(a.get("notes", "[]")), stats=stats)
    d.cache_path = str(path)
    return d
