"""Time / memory check of one run, read in full and reduced (light) form.

    python -m xtg_mid_gui.bench 7                 # proposal 11456
    python -m xtg_mid_gui.bench 7 --proposal 7016
"""
from __future__ import annotations

import argparse
import resource
import sys
import time

import numpy as np

from . import analysis as an
from .backends import default_workers, load_many, make_backend, parse_scans
from .config import config_for


def _rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 / (1024 if sys.platform == "darwin" else 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=int)
    ap.add_argument("--proposal", type=int, default=11456)
    ap.add_argument("--many", help="also time reading these runs light, in parallel, e.g. '7-14'")
    a = ap.parse_args(argv)
    preset = str(a.proposal) if str(a.proposal) in ("7016", "11456") else "custom"
    cfg = config_for(preset, None)
    cfg.proposal = a.proposal
    be = make_backend("extra_data", cfg)

    t0 = time.time()
    full = be.load_scan(a.run)
    t_full = time.time() - t0
    mb_full = sum(v.nbytes for v in full.images.values()) / 1e6
    print(f"full read : {t_full:6.1f} s, {full.n} trains, images {mb_full:8.0f} MB, peak memory {_rss_mb():6.0f} MB")

    rois = {}
    for det, imgs in full.images.items():                 # a ROI on the brightest spot + one background
        m = np.nan_to_num(full.mean_image(det), nan=-np.inf)
        y, x = np.unravel_index(np.argmax(m), m.shape)
        rois[det] = ([(x + 0.5, y + 0.5, 6.0, 6.0, 0.0, "ellipse")], [(m.shape[1] * 0.1, m.shape[0] * 0.1, 6.0, 6.0, 0.0, "ellipse")])
    sig_full = {det: an.roi_signal(full.images[det], g[0][0], g[1]) for det, g in rois.items()}
    del full

    t0 = time.time()
    light = be.load_scan(a.run, rois=rois)
    t_light = time.time() - t0
    mb_light = sum(s.nbytes for s in light.stats.values()) / 1e6
    print(f"light read: {t_light:6.1f} s, {light.n} trains, kept   {mb_light:8.1f} MB")
    for det, (sig, bkg) in rois.items():
        s = light.stats[det].signal(sig[0], bkg)
        d = np.nanmax(np.abs(s - sig_full[det])) / (np.nanmax(np.abs(sig_full[det])) or 1)
        print(f"  {det}: same ROI signal as full read: {'yes' if d < 1e-5 else 'NO (rel. diff %.2g)' % d}")
    if a.many:
        runs = parse_scans(a.many)
        t0 = time.time()
        res = dict(load_many(be, [(r, rois) for r in runs]))
        bad = {r: v for r, v in res.items() if isinstance(v, Exception)}
        print(f"{len(runs)} runs light, {default_workers()} at a time: {time.time() - t0:6.1f} s "
              f"({(time.time() - t0) / len(runs):.1f} s per run)")
        for r, e in bad.items():
            print(f"  run {r}: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
