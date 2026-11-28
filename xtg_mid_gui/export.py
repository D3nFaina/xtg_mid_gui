"""Reduced-data file (HDF5): what the GUI exports and the Fitting page reads back.

Layout (``h5ls -r file.h5``)::

    /                              attrs: format="xtg_mid_reduced", version, proposal, created,
                                          normalisation, MAD cut, background, ROI value, runs
    /runs/run_0290                 attrs: delay_fs, n_shots, n_kept
        pump, probe                [mean, error, std]  diamond diodes (or arm spectrometers), kept shots
        spec/<name>                [mean, error, std]  upstream spectrometer integral (e.g. hirex), one per beam
        xgm                        [mean, error, std]  if recorded
        <detector>/<roi>           [delay, intensity, error]   normalised
        <detector>/<roi>_raw       [delay, intensity, error]   not normalised
    /<detector>                    attrs: key (jf1 ...), rois (JSON, as the ROI database),
                                          tiles (AGIPD tiles stacked top to bottom, p7016)
        <roi>/trace                (n, 4) delay_fs, intensity, error, n_shots  normalised, runs with
                                   the same delay merged
        <roi>/trace_raw            same, not normalised
        <roi>/per_run              (n_runs, 4) run, delay_fs, intensity, error  normalised
        <roi>/per_run_raw          same, not normalised

<detector> is the detector's label (Jung_diffraction, Jung_transmission, ...).  error is the
standard error of the mean (ratio of means for normalised values).
"""
from __future__ import annotations

import json
import os
import tempfile
import time

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

FORMAT, VERSION = "xtg_mid_reduced", 1


def safe(name: str) -> str:
    """A ROI / detector name usable as an HDF5 group name."""
    return str(name).replace("/", "_").strip() or "_"


def mean_err(x):
    """[mean, standard error, std] of the finite values."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([np.nan, np.nan, np.nan])
    sd = float(np.std(x, ddof=1)) if x.size > 1 else np.nan
    return np.array([float(np.mean(x)), sd / np.sqrt(x.size) if x.size > 1 else np.nan, sd])


def write(path, meta, runs, dets):
    """meta: dict of attributes.  runs: {run: dict(delay, n_shots, n_kept, pump, probe,
    spec={name: arr}, xgm=arr or None)}.  dets: {label: dict(key, rois, traces={roi:
    dict(raw=(t, y, e, n), norm=(t, y, e, n))}, per_run={roi: {run: dict(raw=(d, y, e),
    norm=(d, y, e))}})}.  Written to a temporary file, then renamed."""
    import h5py
    path = str(path)
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".h5.tmp", dir=folder)
    os.close(fd)
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs.update(format=FORMAT, version=VERSION, created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                           runs=np.array(sorted(runs), int), detectors=json.dumps(list(dets)))
            for k, v in meta.items():
                f.attrs[k] = v if isinstance(v, (int, float, str)) else json.dumps(v)
            for r, info in sorted(runs.items()):
                g = f.create_group(f"runs/run_{int(r):04d}")
                g.attrs.update(run=int(r), delay_fs=float(info["delay"]), n_shots=int(info["n_shots"]),
                               n_kept=int(info["n_kept"]))
                for k in ("pump", "probe", "xgm"):
                    if info.get(k) is not None:
                        g[k] = np.asarray(info[k], float)
                for name, a in (info.get("spec") or {}).items():
                    g[f"spec/{safe(name)}"] = np.asarray(a, float)
            for label, d in dets.items():
                g = f.create_group(safe(label))
                g.attrs.update(key=d.get("key", ""), label=label, rois=json.dumps(d.get("rois", [])))
                if d.get("tiles"):
                    g.attrs["tiles"] = np.array(d["tiles"], int)
                for roi, tr in d["traces"].items():
                    rg = g.create_group(safe(roi))
                    rg.attrs["name"] = roi
                    for kind, suffix in (("norm", ""), ("raw", "_raw")):
                        t, y, e, n = (np.asarray(a, float) for a in tr[kind])
                        ds = rg.create_dataset("trace" + suffix, data=np.column_stack([t, y, e, n]) if len(t)
                                               else np.zeros((0, 4)))
                        ds.attrs["columns"] = "delay_fs, intensity, error, n_shots"
                        rows = [(r, *pr[kind]) for r, pr in sorted(d["per_run"].get(roi, {}).items())]
                        ds = rg.create_dataset("per_run" + suffix, data=np.array(rows, float).reshape(-1, 4))
                        ds.attrs["columns"] = "run, delay_fs, intensity, error"
                    for r, pr in d["per_run"].get(roi, {}).items():
                        f[f"runs/run_{int(r):04d}/{safe(label)}/{safe(roi)}"] = np.asarray(pr["norm"], float)
                        f[f"runs/run_{int(r):04d}/{safe(label)}/{safe(roi)}_raw"] = np.asarray(pr["raw"], float)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ---------------------------------------------------------------- reading (Fitting page)
def is_reduced(path) -> bool:
    import h5py
    try:
        with h5py.File(path, "r") as f:
            return f.attrs.get("format") == FORMAT
    except OSError:
        return False


def detectors(path) -> list[str]:
    import h5py
    with h5py.File(path, "r") as f:
        return json.loads(f.attrs["detectors"])


def traces(path, detector, normalised=True) -> dict:
    """{roi name: (delay_fs, intensity, error, n_shots)} of one detector of a reduced file."""
    import h5py
    out = {}
    with h5py.File(path, "r") as f:
        g = f[safe(detector)]
        for rg in g.values():
            a = rg["trace" if normalised else "trace_raw"][()]
            out[rg.attrs.get("name", rg.name.rsplit("/", 1)[-1])] = tuple(a[:, i] for i in range(4))
    return out


def rois(path) -> dict:
    """{'detectors': {detector key: [ROI dicts]}, 'tiles': {detector key: tiles}} of a reduced file."""
    import h5py
    out = {"detectors": {}, "tiles": {}}
    with h5py.File(path, "r") as f:
        for label in json.loads(f.attrs["detectors"]):
            g = f[safe(label)]
            key = str(g.attrs.get("key", "") or label)
            out["detectors"][key] = json.loads(g.attrs.get("rois", "[]"))
            if "tiles" in g.attrs:
                out["tiles"][key] = [int(t) for t in g.attrs["tiles"]]
    return out


def info(path) -> dict:
    """Header of a reduced file (attributes), for the file list."""
    import h5py
    with h5py.File(path, "r") as f:
        return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in f.attrs.items()}
