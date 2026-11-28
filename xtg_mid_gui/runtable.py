"""Record of analysed runs: ``runs_<proposal>.csv`` in the start folder, written automatically.

One row per run, updated each time the run is analysed (rows of other runs are kept)::

    run,delay_fs,delay_from,n_trains,rois,tiles,last_analysed
    290,1500,typed,144,"jf1: TG+1 ellipse (512.0,300.0) r(20.0,15.0); ...",,2026-11-28T17:10:00

delay_from: "data" (read from the run), "file" (delays_<proposal>.csv) or "typed".
rois: detector: name shape (centre x,y) r(radius x,y) [bkg] [off], separated by "; ".
"""
from __future__ import annotations

import csv
import io
import os
import tempfile
import time
from pathlib import Path

from .delays import start_dir

COLUMNS = ["run", "delay_fs", "delay_from", "n_trains", "rois", "tiles", "last_analysed"]


def path_for(proposal) -> Path:
    return start_dir() / (f"runs_{int(proposal)}.csv" if proposal else "runs_custom.csv")


def load(proposal) -> dict:
    """{run: row dict}.  Missing or unreadable file -> {}."""
    try:
        with open(path_for(proposal), newline="") as f:
            return {int(r["run"]): r for r in csv.DictReader(f) if (r.get("run") or "").strip().isdigit()}
    except OSError:
        return {}


def roi_text(det_rois: dict) -> str:
    parts = []
    for det, rois in det_rois.items():
        for r in rois:
            cx, cy = r.get("center", (0, 0))
            rx, ry = r.get("radii", (0, 0))
            flags = (" bkg" if r.get("background") else "") + ("" if r.get("enabled", True) else " off")
            parts.append(f"{det}: {r.get('name')} {r.get('shape', 'ellipse')} ({cx:.1f},{cy:.1f}) "
                         f"r({rx:.1f},{ry:.1f}){flags}")
    return "; ".join(parts)


def update(proposal, rows: dict) -> bool:
    """rows: {run: {"delay_fs", "delay_from", "n_trains", "det_rois", "tiles"}}.  Merged into the file."""
    table = load(proposal)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for run, r in rows.items():
        d = r.get("delay_fs")
        tiles = r.get("tiles") or {}
        table[int(run)] = dict(
            run=int(run),
            delay_fs="" if d is None or d != d else f"{float(d):g}",
            delay_from=r.get("delay_from", ""),
            n_trains=r.get("n_trains", ""),
            rois=roi_text(r.get("det_rois") or {}),
            tiles="; ".join(f"{k}: {' '.join(map(str, v))}" for k, v in tiles.items()),
            last_analysed=now)
    buf = io.StringIO()
    w = csv.DictWriter(buf, COLUMNS, extrasaction="ignore")
    w.writeheader()
    for run in sorted(table):
        w.writerow(table[run])
    p = path_for(proposal)
    try:
        fd, tmp = tempfile.mkstemp(suffix=".csv", dir=p.parent)
        with os.fdopen(fd, "w", newline="") as f:
            f.write(buf.getvalue())
        os.replace(tmp, p)
        return True
    except OSError:
        return False
