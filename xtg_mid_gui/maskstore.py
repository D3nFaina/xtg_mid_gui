"""Bad-pixel masks read once for the whole experiment and kept in masks_<proposal>.npz
in the start folder (next to delays_<proposal>.csv).  A pixel is bad if its mask is
> 0 in more than half of the frames of the first trains read.  Delete the file (or
press "read the masks again" in the GUI) to read them from the data again."""
from __future__ import annotations

import os
import re
import tempfile

import numpy as np

from .delays import start_dir


def path_for(proposal):
    return start_dir() / f"masks_{proposal}.npz"


def _name(key) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", "|".join(map(str, key))).strip("_")


def load(proposal, key):
    p = path_for(proposal)
    try:
        with np.load(p) as z:
            n = _name(key)
            return z[n].astype(bool) if n in z.files else None
    except (OSError, ValueError, KeyError):
        return None


def save(proposal, key, mask):
    """Add one mask to the file (written to a temporary file, then renamed)."""
    p = path_for(proposal)
    try:
        with np.load(p) as z:
            all_ = {k: z[k] for k in z.files}
    except (OSError, ValueError):
        all_ = {}
    all_[_name(key)] = np.asarray(mask, bool)
    try:
        fd, tmp = tempfile.mkstemp(suffix=".npz", dir=p.parent)
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **all_)
        os.replace(tmp, p)
    except OSError:
        pass                            # read-only folder: the mask is just read again next time


def clear(proposal):
    try:
        path_for(proposal).unlink()
    except OSError:
        pass
