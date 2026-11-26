"""Run -> delay book-keeping: one small CSV file per proposal.

When no delay motor is recorded (p7016), or it has to be corrected, the delay of
each run is typed once in the scan table.  It is saved straight away to
``delays_<proposal>.csv`` in the start folder (the proposal usr/ folder on
Maxwell) and filled in automatically the next time that run is loaded.

A delay in this file wins over the delay read from the data; clearing it in the
table removes it from the file and goes back to the data value.

File format (plain text, easy to edit by hand)::

    # run, delay_fs
    92, -10
    94, 10
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def start_dir() -> Path:
    """Folder the GUI was started from (set by run_xtg_mid_gui.sh), else the current one."""
    return Path(os.environ.get("XTG_MID_START_DIR") or Path.cwd())


def path_for(proposal) -> Path:
    name = f"delays_{int(proposal)}.csv" if proposal else "delays_custom.csv"
    return start_dir() / name


def load(path) -> dict:
    """{run: delay_fs}.  Missing file -> {}.  Bad lines are skipped."""
    path = Path(path)
    out = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p for p in line.replace(";", ",").replace("\t", ",").replace(" ", ",").split(",") if p]
        if len(parts) < 2:
            continue
        try:
            run, delay = int(float(parts[0])), float(parts[1])
        except ValueError:
            continue                      # e.g. a header line
        if np.isfinite(delay):
            out[run] = delay
    return out


def save(path, table: dict):
    path = Path(path)
    lines = ["# run, delay_fs   (written by the XTG MID GUI; edit freely)"]
    lines += [f"{r}, {table[r]:g}" for r in sorted(table)]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, path)
