"""The reduction engine: runs + ROIs in, reduced-data cache files out (no Qt, no GUI).

The GUI only tells this module which runs and ROIs to reduce, then reads the cache
files (store.py).  The same engine runs from a terminal, e.g. on a Maxwell node with
many cores before or while the GUI is open::

    python -m xtg_mid_gui.compute --preset 7016 --runs "286 288-296"
    python -m xtg_mid_gui.compute --preset 11456 --runs 20-60 --rois rois.json --workers 16

A run that is already reduced with the same settings and ROIs is skipped; a run whose
cache lacks some ROIs only gets those added.  The GUI then loads all of them in seconds.
"""
from __future__ import annotations

import json
import sys
import time

from . import store


def full_rois(cfg, rois=None, detectors=None):
    """{detector: (signal geometries, background geometries)} for every image detector
    (detectors: only these), empty lists where no ROI is given (the mean image is still made)."""
    rois = rois or {}
    dets = [d for d in cfg.active_image_detectors() if detectors is None or d in detectors]
    return {d: (list(rois.get(d, ([], []))[0]), list(rois.get(d, ([], []))[1])) for d in dets}


def fp_of(backend):
    return store.fingerprint(backend.cfg, backend.name, backend.fingerprint_extra())


def ensure_run(backend, scan, rois=None, force=False, detectors=None):
    """Make sure the cache file of `scan` holds these ROIs with the current settings.
    Returns {'scan', 'path', 'did'}: did = 'cached' (nothing read), 'rois' (only the new
    ROIs read) or 'read' (whole run read)."""
    cfg = backend.cfg
    rois = full_rois(cfg, rois, detectors)
    fp, ptag = fp_of(backend), store.param_tag(cfg)
    path = store.run_path(cfg, backend.name, scan)
    st = "none" if force else store.status(path, fp, rois)
    did = "cached"
    if st == "none":
        d = backend.load_scan(scan, rois=rois)
        store.write_run(path, d, fp, ptag=ptag)
        did = "read"
    elif st == "rois":
        import h5py
        with h5py.File(path, "r") as f:
            need = {}
            for det, (sig, bkg) in rois.items():
                ms, mb = store.missing_geoms(f, det, sig, bkg)
                if ms or mb:
                    need[det] = (ms, bkg)              # the whole background set: same positions key
        d = backend.load_scan(scan, rois=need)
        store.add_stats(path, d.stats)
        did = "rois"
    if did != "read" and store.get_param_tag(path) != ptag:       # scanned parameter changed
        x = None
        if ptag:
            import numpy as np
            trains = store.load_light(path, detectors=()).train
            x = backend.load_param(scan, trains)
            if x is None:                                        # backend cannot read it alone
                x = backend.load_scan(scan, rois={}).param
            x = None if x is None else np.asarray(x, float)
        store.set_param(path, x, ptag)
    return dict(scan=scan, path=str(path), did=did)


def _ensure_one(cls, cfg, scan, rois, force, detectors):
    """Worker process: reduce one run into its cache file."""
    return ensure_run(cls(cfg), scan, rois, force, detectors)


def ensure_many(backend, scans, rois=None, force=False, detectors=None, workers=None, tick=None):
    """ensure_run for several runs; yields (scan, info dict or the exception) as each is done.
    Real data go to separate processes (`workers` at a time)."""
    from .backends import run_jobs
    jobs = [(s, (rois, force, detectors)) for s in scans]

    def here(s, args):
        return ensure_run(backend, s, *args)
    yield from run_jobs(backend, jobs, here, _ensure_one, workers=workers, tick=tick)


# ---------------------------------------------------------------- command line
def main(argv=None):
    import argparse
    from .backends import make_backend, parse_scans
    from .config import PRESETS, config_for, preset_of_file
    from . import roidb
    ap = argparse.ArgumentParser(description="Reduce MID XTG runs into the cache the GUI reads")
    ap.add_argument("--runs", required=True, help="run numbers, e.g. '286 288-296'")
    ap.add_argument("--preset", choices=list(PRESETS))
    ap.add_argument("--config", help="sources YAML/JSON file")
    ap.add_argument("--proposal", type=int)
    ap.add_argument("--backend", choices=["extra_data", "simulated"], default="extra_data")
    ap.add_argument("--rois", help="ROI file: GUI ROI database (rois_<proposal>.json) or "
                                   "{detector: [ROI dicts]}; default: the ROI database in the start folder")
    ap.add_argument("--roi-run", type=int, help="take the ROIs saved for this run in the ROI database")
    ap.add_argument("--detectors", help="only these detectors, e.g. 'jf1' (default: all)")
    ap.add_argument("--workers", type=int, help="parallel readers (default: half the cores, max 8)")
    ap.add_argument("--force", action="store_true", help="read the runs again even if cached")
    args = ap.parse_args(argv)
    preset = args.preset or ((preset_of_file(args.config) or "custom") if args.config else
                             str(args.proposal) if str(args.proposal) in ("7016", "11456") else "11456")
    cfg = config_for(preset, args.config)
    if args.proposal is not None:
        cfg.proposal = args.proposal
    be = make_backend(args.backend, cfg)
    dets = args.detectors.replace(",", " ").split() if args.detectors else None
    rois = roidb.spec_for_cli(cfg, args.rois, args.roi_run)
    runs = parse_scans(args.runs)
    t0 = time.time()
    n_bad = 0
    for s, info in ensure_many(be, runs, rois, args.force, dets, workers=args.workers):
        if isinstance(info, Exception):
            n_bad += 1
            print(f"run {s}: {type(info).__name__}: {info}", flush=True)
        else:
            print(f"run {s}: {info['did']:<6} {info['path']}  ({time.time() - t0:.0f} s)", flush=True)
    print(json.dumps(dict(runs=len(runs), failed=n_bad, seconds=round(time.time() - t0, 1))))
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
