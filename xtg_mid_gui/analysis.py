"""Pure analysis functions (no Qt), shared by the GUI, the tests and scripts."""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- ROIs
def ellipse_mask(shape, cx, cy, rx, ry, angle_deg=0.0) -> np.ndarray:
    """Boolean mask of an ellipse centred at (cx, cy) in pixel coords
    (x = column, y = row), semi-axes rx, ry, rotated by angle_deg."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    a = np.deg2rad(angle_deg)
    dx, dy = xx + 0.5 - cx, yy + 0.5 - cy
    u = dx * np.cos(a) + dy * np.sin(a)
    v = -dx * np.sin(a) + dy * np.cos(a)
    return (u / max(rx, 1e-9)) ** 2 + (v / max(ry, 1e-9)) ** 2 <= 1.0


def rect_mask(shape, cx, cy, rx, ry, angle_deg=0.0) -> np.ndarray:
    """Boolean mask of a (rotated) rectangle with half-widths rx, ry, same convention as ellipse_mask."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    a = np.deg2rad(angle_deg)
    dx, dy = xx + 0.5 - cx, yy + 0.5 - cy
    u = dx * np.cos(a) + dy * np.sin(a)
    v = -dx * np.sin(a) + dy * np.cos(a)
    return (np.abs(u) <= rx) & (np.abs(v) <= ry)


def roi_crop(shape, geom):
    """(row slice, col slice, mask in the crop) for geom = (cx, cy, rx, ry, angle[, kind]),
    kind = "ellipse" (default) or "rectangle"."""
    ny, nx = shape
    cx, cy, rx, ry, ang = geom[:5]
    kind = geom[5] if len(geom) > 5 else "ellipse"
    r = (np.hypot(rx, ry) if str(kind).startswith("rect") else max(rx, ry)) + 1
    y0, y1 = int(max(0, np.floor(cy - r))), int(min(ny, np.ceil(cy + r)))
    x0, x1 = int(max(0, np.floor(cx - r))), int(min(nx, np.ceil(cx + r)))
    if y1 <= y0 or x1 <= x0:
        return slice(0, 0), slice(0, 0), np.zeros((0, 0), bool)
    f = rect_mask if str(kind).startswith("rect") else ellipse_mask
    m = f((y1 - y0, x1 - x0), cx - x0, cy - y0, rx, ry, ang)
    return slice(y0, y1), slice(x0, x1), m


ellipse_crop = roi_crop          # backwards-compatible name


def roi_stats(images: np.ndarray, geom) -> tuple[np.ndarray, int]:
    """Per-shot SUM over the ROI (ellipse or rect) and its pixel count (NaN pixels ignored)."""
    ys, xs, m = roi_crop(images.shape[1:], geom)
    if m.sum() == 0:
        return np.full(len(images), np.nan), 0
    crop = images[:, ys, xs][:, m].astype(np.float64)
    good = np.isfinite(crop).any(axis=0)          # pixels in a gap (e.g. between AGIPD tiles) do not count
    npx = int(good.sum())
    if npx == 0:
        return np.full(len(images), np.nan), 0
    return np.nansum(crop[:, good], axis=1), npx


def roi_signal(images, geom, bkg_geoms=()) -> np.ndarray:
    """Integrated ROI counts minus npx * (mean per-pixel level of the background ROIs)."""
    s, npx = roi_stats(images, geom)
    if not bkg_geoms or npx == 0:
        return s
    tot, n = np.zeros(len(images)), 0
    for g in bkg_geoms:
        b, nb = roi_stats(images, g)
        if nb:
            tot += b
            n += nb
    return s - npx * tot / n if n else s


# ---------------------------------------------------------------- spectrometers
def spectrum_integral(spec, n_edge=20, window=None):
    """Per-shot integral and centroid of (n, npx) spectra.

    Each shot gets an edge-baseline subtraction (mean of the first/last n_edge px).
    window = (i0, i1) pixel indices into the spectrum restricts the integral and
    centroid, e.g. to the seed line.  Returns (integral, centroid_px_index)."""
    spec = np.asarray(spec, np.float64)
    if n_edge > 0 and spec.shape[1] > 2 * n_edge:
        base = 0.5 * (spec[:, :n_edge].mean(axis=1) + spec[:, -n_edge:].mean(axis=1))
        spec = spec - base[:, None]
    i0, i1 = (0, spec.shape[1]) if window is None else (max(0, int(window[0])), min(spec.shape[1], int(window[1])))
    sub = spec[:, i0:i1]
    integ = sub.sum(axis=1)
    w = np.clip(sub, 0, None)
    ws = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cen = (w * np.arange(i0, i1)[None]).sum(axis=1) / ws
    cen[ws <= 0] = np.nan
    return integ, cen


# ---------------------------------------------------------------- shot cleaning
def robust_mask(x, nmad=5.0) -> np.ndarray:
    """True = keep. Median/MAD z-score cut (as in the p7016 notebook)."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return ok
    med = np.median(x[ok])
    mad = np.median(np.abs(x[ok] - med)) * 1.4826
    if mad == 0:
        return ok
    return ok & (np.abs(x - med) <= nmad * mad)


def clean_scan(pump, probe, nmad=5.0, min_frac=0.0, positive=True, report=None) -> np.ndarray:
    """Keep shots with finite diodes, (optionally) positive diodes, inside the MAD cut,
    and above min_frac x median (drops X-ray-off shots).

    A cut that would remove EVERY shot of the scan is not applied (e.g. diodes that are
    negative-going or have an offset, or a monitor that is all NaN); its name goes into
    report["skipped"].  report["cut"] counts the shots each applied cut removed."""
    pump, probe = np.asarray(pump, float), np.asarray(probe, float)
    rep = {"cut": {}, "skipped": []} if report is None else report
    rep.setdefault("cut", {})
    rep.setdefault("skipped", [])
    keep = np.isfinite(pump) & np.isfinite(probe)
    if not keep.any():
        rep["skipped"].append("no finite diode values")
        return np.ones(len(pump), bool)

    def apply(name, m):
        nonlocal keep
        new = keep & m
        if new.any():
            rep["cut"][name] = int(keep.sum() - new.sum())
            keep = new
        else:
            rep["skipped"].append(name)

    with np.errstate(invalid="ignore"):
        if positive:
            apply("diode <= 0", (pump > 0) & (probe > 0))
        if nmad > 0:
            apply("MAD", robust_mask(pump, nmad) & robust_mask(probe, nmad))
        if min_frac > 0:
            mp, mq = np.nanmedian(pump[keep]), np.nanmedian(probe[keep])
            if mp > 0 and mq > 0:
                apply("beam off", (pump > min_frac * mp) & (probe > min_frac * mq))
            else:
                rep["skipped"].append("beam off (median <= 0)")
    return keep


# ---------------------------------------------------------------- normalisation
NORMALISERS = {
    "none": lambda pu, pr: np.ones_like(pu),
    "probe": lambda pu, pr: pr,
    "pump": lambda pu, pr: pu,
    "pump x probe": lambda pu, pr: pu * pr,
    "pump^2 x probe": lambda pu, pr: pu ** 2 * pr,
}


def normaliser(name, pump, probe) -> np.ndarray:
    v = np.asarray(NORMALISERS[name](np.asarray(pump, float), np.asarray(probe, float)), float).copy()
    v[~np.isfinite(v) | (v == 0)] = np.nan
    return v


def ratio_of_means(num, den):
    """r = <num>/<den> and its standard error (ratio-estimator residuals),
    which handles correlated num/den correctly.  Returns (r, se, n)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    ok = np.isfinite(num) & np.isfinite(den)
    num, den = num[ok], den[ok]
    n = len(num)
    if n == 0:
        return np.nan, np.nan, 0
    mn, md = num.mean(), den.mean()
    if md == 0:
        return np.nan, np.nan, n
    r = mn / md
    se = np.sqrt(np.sum((num - r * den) ** 2) / max(n - 1, 1) / n) / abs(md)
    return r, se, n


def delay_series(delays, per_scan_num, per_scan_den):
    """Combine scans into a delay series.

    delays: (n_scans,) one delay per scan.  per_scan_num / den: lists of per-shot
    arrays (signal, normaliser) for each scan.  Scans with the same delay are
    merged (all their shots pooled).  Returns sorted (delay, value, error, n_shots).
    """
    delays = np.asarray(delays, float)
    out = []
    for d in np.unique(delays[np.isfinite(delays)]):
        idx = np.flatnonzero(delays == d)
        num = np.concatenate([per_scan_num[i] for i in idx])
        den = np.concatenate([per_scan_den[i] for i in idx])
        out.append((d, *ratio_of_means(num, den)))
    if not out:
        z = np.array([])
        return z, z, z, z
    a = np.array(out, float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def param_steps(x, max_steps=60):
    """Motor positions of a stepped scan: groups values closer than 0.5 % of the range.
    Returns the step centres, or None if the values are not clearly stepped or there
    are more than max_steps positions (a continuous scan, e.g. a drifting temperature)."""
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    if x.size == 0:
        return np.array([])
    span = x[-1] - x[0]
    if span <= 0:
        return np.array([x[0]])
    cuts = np.flatnonzero(np.diff(x) > 0.005 * span)
    if cuts.size + 1 > max_steps or cuts.size == 0:
        return None
    groups = np.split(x, cuts + 1)
    centres = np.array([g.mean() for g in groups])
    width = max(g[-1] - g[0] for g in groups)
    if width > 0.25 * np.diff(centres).min():          # values spread out: not a stepped motor
        return None
    return centres


def param_series(x, num, den=None, mode="auto", nbins=30, max_steps=60, min_shots=3):
    """Signal vs a parameter scanned inside the run(s).

    x, num, den: per-shot arrays (parameter, signal, normaliser; den=None -> 1).
    mode "auto": one point per motor step if the parameter is stepped (<= max_steps
    distinct positions), else nbins equal-width bins; "bins": always nbins bins.
    Each point = ratio of means of the shots in it (as for the delay series).
    Returns (centre, value, error, n_shots); points with < min_shots are dropped."""
    x, num = np.asarray(x, float), np.asarray(num, float)
    den = np.ones_like(num) if den is None else np.asarray(den, float)
    ok = np.isfinite(x) & np.isfinite(num) & np.isfinite(den)
    x, num, den = x[ok], num[ok], den[ok]
    z = np.array([])
    if x.size == 0:
        return z, z, z, z
    steps = param_steps(x, max_steps) if mode == "auto" else None
    if steps is not None and steps.size:
        mid = (steps[1:] + steps[:-1]) / 2
        idx = np.searchsorted(mid, x)                 # nearest step
        groups = range(steps.size)
    else:
        lo, hi = x.min(), x.max()
        edges = np.linspace(lo, hi if hi > lo else lo + 1, int(max(nbins, 1)) + 1)
        idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
        groups = range(len(edges) - 1)
    out = []
    for g in groups:
        m = idx == g
        if m.sum() >= min_shots:
            out.append((x[m].mean(), *ratio_of_means(num[m], den[m])))
    if not out:
        return z, z, z, z
    a = np.array(out, float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


# ---------------------------------------------------------------- ROI position check
def roi_bkg_level(images, bkg_geoms=()) -> np.ndarray:
    """Per-frame mean per-pixel level of the background ROIs (0 if none)."""
    tot, n = np.zeros(len(images)), 0
    for g in bkg_geoms:
        b, nb = roi_stats(images, g)
        if nb:
            tot += b
            n += nb
    return tot / n if n else tot


def roi_position(images, geom, mode="com", bkg_level=None):
    """Per-frame position (x, y) in pixel coords (x = column, y = row, pixel centre at +0.5,
    as the ROIs) of the intensity inside the ROI.

    mode "com": centre of mass of (I - background), negative values set to 0;
    mode "max": the brightest pixel.  NaN where the ROI is empty."""
    images = np.asarray(images)
    n = len(images)
    ys, xs, m = roi_crop(images.shape[1:], geom)
    nanxy = np.full(n, np.nan), np.full(n, np.nan)
    if m.sum() == 0:
        return nanxy
    crop = images[:, ys, xs].astype(np.float64)
    if bkg_level is not None:
        crop = crop - np.asarray(bkg_level, float).reshape(-1, 1, 1)
    yy, xx = np.mgrid[ys, xs]
    xx, yy = xx[m] + 0.5, yy[m] + 0.5
    v = crop[:, m]                                          # (n, npx)
    if mode == "max":
        ok = np.isfinite(v).any(axis=1)
        k = np.argmax(np.where(np.isfinite(v), v, -np.inf), axis=1)
        x, y = xx[k].astype(float), yy[k].astype(float)
        x[~ok], y[~ok] = np.nan, np.nan
        return x, y
    w = np.clip(np.nan_to_num(v, nan=0.0), 0, None)
    tot = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(tot > 0, (w * xx).sum(axis=1) / tot, np.nan)
        y = np.where(tot > 0, (w * yy).sum(axis=1) / tot, np.nan)
    return x, y
