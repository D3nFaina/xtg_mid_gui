"""ROI database: which ROIs were used, per run, in ``rois_<proposal>.json`` (start folder).

* ``last``: the ROIs on screen when the GUI was last used; they come back at start.
* ``runs``: for each run, the ROIs it was last analysed with ("ROIs for run 290").

Format (plain JSON, one ROI = the dict of rois.RoiSpec.to_dict)::

    {"version": 1,
     "last": {"saved": "2026-11-28T07:12:00", "detectors": {"jf1": [{...}, ...], "jf2": [...]},
              "tiles": {"agipd_sub": [9, 8, 15]}},          # AGIPD tiles the ROIs sit on (p7016)
     "runs": {"290": {"saved": "...", "detectors": {...}}, ...}}
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from .delays import start_dir


def path_for(proposal) -> Path:
    return start_dir() / (f"rois_{int(proposal)}.json" if proposal else "rois_custom.json")


def load_db(proposal) -> dict:
    try:
        d = json.loads(path_for(proposal).read_text())
    except (OSError, ValueError):
        d = {}
    d.setdefault("version", 1)
    d.setdefault("runs", {})
    return d


def _write(proposal, db):
    p = path_for(proposal)
    try:
        fd, tmp = tempfile.mkstemp(suffix=".json", dir=p.parent)
        with os.fdopen(fd, "w") as f:
            json.dump(db, f, indent=1)
        os.replace(tmp, p)
        return True
    except OSError:
        return False


def _entry(det_rois, tiles=None):
    e = {"saved": time.strftime("%Y-%m-%dT%H:%M:%S"), "detectors": det_rois}
    if tiles:
        e["tiles"] = {k: [int(t) for t in v] for k, v in tiles.items()}
    return e


def save(proposal, det_rois, runs=(), tiles=None):
    """Store det_rois ({detector: [ROI dicts]}) as the last ROIs and as the ROIs of `runs`;
    tiles: {AGIPD detector: tile list} the stacked-tile ROIs belong to."""
    db = load_db(proposal)
    db["last"] = _entry(det_rois, tiles)
    for r in runs:
        db["runs"][str(int(r))] = _entry(det_rois, tiles)
    return _write(proposal, db)


def entry(proposal, run=None) -> dict:
    """The whole saved entry ({'detectors': ..., 'tiles': ...}) of a run, or the last one; {} if none."""
    db = load_db(proposal)
    return (db.get("last") if run is None else db["runs"].get(str(int(run)))) or {}


def last(proposal):
    return (load_db(proposal).get("last") or {}).get("detectors")


def for_run(proposal, run):
    """{detector: [ROI dicts]} saved for this run, or None."""
    return (load_db(proposal)["runs"].get(str(int(run))) or {}).get("detectors")


def runs_with_rois(proposal) -> list[int]:
    return sorted(int(r) for r in load_db(proposal)["runs"])


# ---------------------------------------------------------------- ROI dicts -> geometries
def geometry(d) -> tuple:
    """ROI dict -> (cx, cy, rx, ry, angle, shape), as RoiSpec.geometry()."""
    (cx, cy), (rx, ry) = d["center"], d["radii"]
    return (round(float(cx), 2), round(float(cy), 2), round(abs(float(rx)), 2), round(abs(float(ry)), 2),
            round(float(d.get("angle", 0.0)), 2), d.get("shape", "ellipse"))


def spec_from_dicts(det_rois) -> dict:
    """{detector: (geometries of all ROIs, geometries of the enabled background ROIs)},
    as the GUI's roi_spec (switched-off ROIs are reduced too, so ticking them is free)."""
    return {det: ([geometry(r) for r in rois],
                  [geometry(r) for r in rois if r.get("background") and r.get("enabled", True)])
            for det, rois in (det_rois or {}).items()}


def read_roi_file(path) -> dict:
    """{detector: [ROI dicts]} from a ROI database, a {detector: [dicts]} file or one
    detector's 'Save ROIs' file ({"detector": ..., "rois": [...]})."""
    d = json.loads(Path(path).read_text())
    if "last" in d or "runs" in d:
        return (d.get("last") or {}).get("detectors") or {}
    if "detector" in d and "rois" in d:
        return {d["detector"]: d["rois"]}
    return d


def spec_for_cli(cfg, path=None, run=None) -> dict:
    """ROIs for the command-line reducer: run's ROIs from the database, a file, or the last ROIs."""
    if run is not None:
        det_rois = for_run(cfg.proposal, run)
        if det_rois is None:
            raise SystemExit(f"no ROIs saved for run {run} in {path_for(cfg.proposal)}")
    elif path:
        det_rois = read_roi_file(path)
    else:
        det_rois = last(cfg.proposal)
        if not det_rois:
            print(f"no ROIs in {path_for(cfg.proposal)}: only the per-run numbers and mean images are made")
    return spec_from_dicts(det_rois)
