"""XTG delay-trace fitting, ported from Den's GaO_Analysis_mac.ipynb (freq2_dump*,
freq3_dump*, freq3_dump2_2nd_harm, fit_exp, do_fft_*).  No Qt.

Model (heterodyne-free TG: the measured signal is the modulus squared of the
grating amplitude), time t in ps, frequencies in THz (1/ps):

    S(t) = | H(t) * ( a_exp exp(-(t-t0)/tau)
                      - sum_i a_i sin(2 pi nu_i (t-t0) + phi_i) [exp(-(t-t0)/tau_i)]
                      [- a_h sin(2 pi 2 nu_1 (t-t0) + phi_1) exp(-(t-t0)/(tau/2))]
                      + y0 ) |^2
    H(t) = 0.5 (1 + erf((t - t0)/sigma))

* n_osc = 1, 2 or 3 oscillations; each one damped (own tau_i) or not.
  freq2_dump1 = 2 osc, damped (False, True); freq2_dump2 = (True, True);
  freq3_dump1 = (False, True, False); freq3_dump2 = (False, True, True);
  freq3_dump3 = (True, True, True).
* harmonic=True adds the 2nd harmonic of nu_1 (freq3_dump2_2nd_harm).
* thermal part = the same expression with all oscillations removed (fit_exp).

The GUI shows nu in GHz (x1000), as the notebook prints it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.special import erf
except ImportError:                    # pragma: no cover - scipy ships with exfel-python
    from math import erf as _erf
    erf = np.vectorize(_erf)


@dataclass
class Param:
    name: str
    value: float
    lo: float
    hi: float
    fixed: bool = False
    unit: str = ""
    display: float = 1.0     # GUI shows value * display (nu: THz -> GHz)


@dataclass
class ModelSpec:
    n_osc: int = 2
    damped: list = field(default_factory=lambda: [False, True, True])
    harmonic: bool = False

    def names(self) -> list[str]:
        out = ["t0", "sigma", "a_exp", "tau"]
        for i in range(1, self.n_osc + 1):
            out += [f"a{i}", f"nu{i}", f"phi{i}"]
            if self.damped[i - 1]:
                out.append(f"tau{i}")
        if self.harmonic:
            out.append("a_h")
        return out + ["y0"]

    def label(self) -> str:
        d = ", ".join("damped" if self.damped[i] else "undamped" for i in range(self.n_osc))
        return f"{self.n_osc} oscillation{'s' if self.n_osc > 1 else ''} ({d})" + (" + 2nd harmonic" if self.harmonic else "")


def default_params(spec: ModelSpec, t=None, nus_thz=(2.4, 4.0, 6.0)) -> dict[str, Param]:
    """Starting values / bounds in the spirit of the notebook's p0 / cons (t in ps)."""
    span = 2.0 if t is None or len(t) < 2 else float(np.nanmax(t) - np.nanmin(t))
    p = {
        "t0": Param("t0", 0.0, -0.5, 0.5, unit="ps"),
        "sigma": Param("sigma", 0.05, 0.001, 0.5, unit="ps"),
        "a_exp": Param("a_exp", 0.9, 0.1, 1.5),
        "tau": Param("tau", max(span / 2, 0.05), 0.01, 1000.0, unit="ps"),
        "a_h": Param("a_h", 0.05, 0.0, 1.0),
        "y0": Param("y0", 0.3, -1.0, 1.0),
    }
    for i in range(1, 4):
        nu = nus_thz[i - 1]
        p[f"a{i}"] = Param(f"a{i}", 0.1, 0.0, 1.0)
        p[f"nu{i}"] = Param(f"nu{i}", nu, 0.8 * nu, 1.2 * nu, unit="GHz", display=1000.0)
        p[f"phi{i}"] = Param(f"phi{i}", 0.1, -3.14, 3.14, unit="rad")
        p[f"tau{i}"] = Param(f"tau{i}", max(span / 2, 0.05), 0.01, 1000.0, unit="ps")
    return {n: p[n] for n in spec.names()}


# ---------------------------------------------------------------- model
def _amplitude(t, v: dict, spec: ModelSpec, oscillations=True):
    t = np.asarray(t, float)
    dt = t - v["t0"]
    amp = v["a_exp"] * np.exp(-dt / v["tau"]) + v["y0"]
    if oscillations:
        for i in range(1, spec.n_osc + 1):
            osc = v[f"a{i}"] * np.sin(2 * np.pi * v[f"nu{i}"] * dt + v[f"phi{i}"])
            if spec.damped[i - 1]:
                osc = osc * np.exp(-dt / v[f"tau{i}"])
            amp = amp - osc
        if spec.harmonic:
            amp = amp - v["a_h"] * np.sin(2 * np.pi * 2 * v["nu1"] * dt + v["phi1"]) * np.exp(-dt / (v["tau"] / 2))
    with np.errstate(over="ignore", invalid="ignore"):
        return np.abs(0.5 * (1 + erf(dt / v["sigma"])) * amp) ** 2


def model(t, v: dict, spec: ModelSpec):
    """Full fit function."""
    return _amplitude(t, v, spec, True)


def thermal(t, v: dict, spec: ModelSpec | None = None):
    """Thermal part alone: step x (exponential decay + offset), squared."""
    return _amplitude(t, v, spec or ModelSpec(n_osc=0), False)


@dataclass
class FitResult:
    values: dict
    errors: dict
    spec: ModelSpec
    least_squares: float
    success: bool = True
    message: str = ""
    fitted: tuple = ()          # parameters this fit actually used (free or fixed)


def fit(t, y, spec: ModelSpec, params: dict[str, Param], thermal_only=False, maxfev=20000) -> FitResult:
    """Bounded least squares (scipy curve_fit, as in the notebook).  Fixed
    parameters are held at their value.  thermal_only fits fit_exp's model."""
    from scipy.optimize import curve_fit

    t, y = np.asarray(t, float), np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    names = ["t0", "sigma", "a_exp", "tau", "y0"] if thermal_only else spec.names()
    free = [n for n in names if not params[n].fixed]
    fixed = {n: params[n].value for n in names if params[n].fixed}
    func = thermal if thermal_only else model

    def f(tt, *a):
        return func(tt, {**fixed, **dict(zip(free, a))}, spec)

    p0 = [min(max(params[n].value, params[n].lo), params[n].hi) for n in free]
    lo = [params[n].lo for n in free]
    hi = [params[n].hi for n in free]
    hi = [h if h > l else l + 1e-9 for l, h in zip(lo, hi)]
    try:
        popt, pcov = curve_fit(f, t, y, p0=p0, bounds=(lo, hi), maxfev=maxfev)
        err = np.sqrt(np.clip(np.diag(pcov), 0, None))
        success, msg = True, ""
    except Exception as e:                 # RuntimeError: no convergence, ValueError: bad input
        popt, err, success, msg = np.array(p0), np.full(len(p0), np.nan), False, str(e)
    values = {**fixed, **dict(zip(free, popt))}
    errors = {**{n: 0.0 for n in fixed}, **dict(zip(free, err))}
    for n in params:                      # keep params not used by this fit (thermal-only)
        values.setdefault(n, params[n].value)
        errors.setdefault(n, np.nan)
    res = y - func(t, values, spec)
    return FitResult(values, errors, spec, float(np.sum(res ** 2)), success, msg, tuple(names))


# ---------------------------------------------------------------- FFT
def uniform(t, y, step=None):
    """Resample onto a uniform grid (the delay points need not be evenly spaced).
    step defaults to the median spacing."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    o = np.argsort(t)
    t, y = t[o], y[o]
    if len(t) < 4:
        return t, y
    step = step or float(np.median(np.diff(t)))
    if not step > 0:                       # repeated delays: fall back to the mean spacing
        step = (t[-1] - t[0]) / (len(t) - 1)
    if not step > 0:
        return t, y
    tu = np.arange(t[0], t[-1] + 1e-9 * step, step)   # no grid point beyond the last delay
    return tu, np.interp(tu, t, y)


def spacing_report(t, rel_tol=0.01):
    """(is_even, min_step, max_step, median_step) of the sorted delay points."""
    t = np.sort(np.asarray(t, float)[np.isfinite(t)])
    if len(t) < 3:
        return True, np.nan, np.nan, np.nan
    d = np.diff(t)
    med = float(np.median(d))
    return bool(d.max() - d.min() <= rel_tol * med), float(d.min()), float(d.max()), med


def fft_with_grid(t, y, t_min=None, step=None):
    """Single-sided amplitude spectrum 2/N |FFT| (as in do_fft_lorentz) of y(t)
    for t >= t_min, computed on an EVEN grid: the points are first linearly
    interpolated onto a grid of `step` (default median spacing).
    Returns (t_grid, y_grid, freq [THz], amplitude)."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    if t_min is not None:
        keep = t >= t_min
        t, y = t[keep], y[keep]
    tu, yu = uniform(t, y, step)
    n = len(yu)
    if n < 4:
        z = np.array([])
        return tu, yu, z, z
    f = np.fft.rfftfreq(n, d=tu[1] - tu[0])
    a = 2.0 / n * np.abs(np.fft.rfft(yu - yu.mean()))
    return tu, yu, f[: n // 2], a[: n // 2]


def fft_amplitude(t, y, t_min=None, step=None):
    """(freq [THz], amplitude) - see fft_with_grid."""
    return fft_with_grid(t, y, t_min, step)[2:]


def guess_frequencies(freq, amp, n, fmin=None):
    """The n strongest local maxima of an amplitude spectrum (THz), strongest first."""
    freq, amp = np.asarray(freq), np.asarray(amp)
    if len(amp) < 3:
        return []
    fmin = freq[1] if fmin is None else fmin
    pk = [i for i in range(1, len(amp) - 1) if amp[i] >= amp[i - 1] and amp[i] >= amp[i + 1] and freq[i] >= fmin]
    pk.sort(key=lambda i: -amp[i])
    return sorted(float(freq[i]) for i in pk[:n])
