"""Data backends.  A backend turns a scan number into a :class:`ScanData`.

* :class:`SimulatedBackend` - fake XTG delay scan, works anywhere.
* :class:`ExtraDataBackend` - processed run files on Maxwell via ``extra_data``,
  using the source names from :class:`~xtg_mid_gui.config.SourceConfig`.
  Untested on real data (names unknown); written so that only the config changes.

One scan = one run = one pump-probe delay.  The GUI loads several scans and
combines them into a delay series.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np

from . import agipd
from .analysis import photonize, spectrum_integral
from .config import SourceConfig, Source, mask_key_for
from . import maskstore
from .reduce import ImageReducer, reduce_images


@dataclass
class ScanData:
    scan: int
    delay: float                       # fs, NaN if unknown (type it in the scan table)
    images: dict                       # detector name -> (n, ny, nx) float32
    pump: np.ndarray                   # (n,) pump diode
    probe: np.ndarray                  # (n,) probe diode
    crl: dict = field(default_factory=dict)   # "crl_pump_x" ... -> (n,), may be empty
    train: np.ndarray | None = None
    spectra: dict = field(default_factory=dict)    # upstream spectrometer name -> (n, npx) raw spectra
    spec_axis: dict = field(default_factory=dict)  # spectrometer name -> (npx,) pixel or eV
    overview: dict = field(default_factory=dict)   # AGIPD tile -> (512, 128) mean image (all-tiles view)
    xgm: np.ndarray | None = None                  # (n,) gas monitor, None if not recorded
    param: np.ndarray | None = None                # (n,) parameter scanned inside the run, None if not read
    delay_data: float = np.nan                     # delay as read from the data (before the delays file)
    delay_from: str = "data"                       # "data", "file" (delays_<proposal>.csv) or "typed"
    masks: dict = field(default_factory=dict)      # name -> what the bad-pixel mask did (for the log)
    photon_energy: float = np.nan                  # keV used to photonize the Jungfraus, NaN = not photonized
    notes: list = field(default_factory=list)      # other messages for the log
    stats: dict = field(default_factory=dict)      # detector -> ImageStats when its images were not kept
    cache_path: str = ""                           # reduced-data cache file (store.py) it was loaded from
    _mean: dict = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return len(self.pump)

    @property
    def reduced(self) -> bool:
        """True for a run whose images were reduced to numbers (not the reference run)."""
        return bool(self.stats)

    def mean_image(self, det: str) -> np.ndarray:
        if det in self.stats:
            return self.stats[det].get_mean()
        if det not in self._mean:
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)          # NaN gaps between AGIPD tiles
                self._mean[det] = np.nanmean(self.images[det], axis=0, dtype=np.float64).astype(np.float32)
        return self._mean[det]


class Backend:
    name = "base"

    def __init__(self, cfg: SourceConfig):
        self.cfg = cfg

    def load_scan(self, scan: int, rois=None) -> ScanData:     # pragma: no cover - interface
        """rois=None: keep every per-train image (reference run).  rois = {detector:
        (signal geometries, background geometries)}: keep only ImageStats for those
        detectors (all other runs); detectors not in the dict are not read at all."""
        raise NotImplementedError

    def load_param(self, scan: int, trains):
        """Only the scanned parameter of a loaded scan, matched to its trains.  None if the
        backend cannot read it alone (then the GUI reloads the whole scan)."""
        return None

    def fingerprint_extra(self) -> dict:
        """Backend settings that change the data (part of the cache fingerprint, store.py)."""
        return {}


# ---------------------------------------------------------------- simulated
# deliberately NOT evenly spaced (coarse before t0, fine around the oscillations, coarse late)
SIM_DELAYS = (list(range(-300, 0, 60)) + list(range(0, 1240, 40)) + list(range(1300, 2600, 100)))


def xtg_response(t):
    """Toy XTG response: step at t0 (50 fs rise), fast decay + two damped
    oscillations (2.38 and 4.0 THz) on a long-lived plateau.  Arbitrary units."""
    from math import erf
    t = np.asarray(t, float)
    step = 0.5 * (1 + np.vectorize(erf)(t / 50.0))
    tp = np.clip(t, 0, None)
    return step * (0.55 * np.exp(-tp / 250) + 0.45) * (1 + 0.25 * np.cos(2 * np.pi * tp / 420) * np.exp(-tp / 900)
                                                     + 0.12 * np.cos(2 * np.pi * tp / 250) * np.exp(-tp / 600))


# feature positions as fractions of the image (row, col, sigma_rows, sigma_cols)
SIM_FEATURES = {
    "jf1": {"TG +1": (0.5, 0.25, 0.06, 0.05), "Bragg": (0.5, 0.5, 0.07, 0.06),
            "TG -1": (0.5, 0.75, 0.06, 0.05)},
    "jf2": {"TG +2": (0.35, 0.3, 0.06, 0.05), "Bragg 2": (0.55, 0.55, 0.08, 0.07),
            "TG -2": (0.75, 0.8, 0.06, 0.05)},
}


class SimulatedBackend(Backend):
    """Scans 101.. with one delay each, taken in a shuffled delay order so the
    GUI has to sort / combine them.  Physics is only roughly XTG-like:

    * pump, probe: SASE-like gamma fluctuations, partly correlated (same source)
    * self-seeded SASE: broad spiky SASE pedestal + narrow seed line on the upstream
      spectrometers; the diodes (after SDL + monochromators) see mostly the seed line,
      so diode vs full spectrometer integral is only weakly correlated
    * CRL pointing: small jitter around 0 (feedback loop); the diode drops when off-centre
    * TG orders ~ pump^2 * probe * response(delay); Bragg ~ probe
    """
    name = "simulated"

    def __init__(self, cfg: SourceConfig | None = None, n_shots=40, delays=None, seed=7, first_scan=101):
        super().__init__(cfg or SourceConfig())
        self.n_shots = n_shots
        rng = np.random.default_rng(seed)
        delays = list(SIM_DELAYS if delays is None else delays)
        order = rng.permutation(len(delays))
        self.delays = {first_scan + i: float(delays[j]) for i, j in enumerate(order)}
        self.seed = seed

    def scans(self) -> list[int]:
        return sorted(self.delays)

    def fingerprint_extra(self) -> dict:
        return dict(n_shots=self.n_shots, seed=self.seed, delays=sorted(self.delays.items()))

    def load_scan(self, scan: int, rois=None) -> ScanData:
        if scan not in self.delays:
            raise KeyError(f"simulated scans are {min(self.delays)}-{max(self.delays)}")
        rng = np.random.default_rng(self.seed * 1000 + scan)
        n, delay = self.n_shots, self.delays[scan]
        sase = rng.gamma(9.0, 1 / 9.0, n)                       # total SASE pulse energy
        seed = rng.gamma(3.0, 1 / 3.0, n) * (0.75 + 0.25 * sase)  # seeded line, weakly tied to it
        pump_true = seed * rng.gamma(30.0, 1 / 30.0, n)
        probe_true = seed * rng.gamma(30.0, 1 / 30.0, n) * 0.6
        crl = {k: rng.normal(0, 2.5, n) for k in SourceConfig.CRL_KEYS}
        clip_p = np.exp(-(crl["crl_pump_x"] ** 2 + crl["crl_pump_y"] ** 2) / (2 * 9.0 ** 2))
        clip_q = np.exp(-(crl["crl_probe_x"] ** 2 + crl["crl_probe_y"] ** 2) / (2 * 9.0 ** 2))
        pump = pump_true * clip_p
        probe = probe_true * clip_q
        pump_d = pump * (1 + rng.normal(0, 0.03, n))          # diode noise
        probe_d = probe * (1 + rng.normal(0, 0.03, n))
        resp = np.full(n, float(xtg_response(delay)))        # per shot, so a scanned motor can modulate it
        param = None
        if self.cfg.param_enabled and self.cfg.param.source:
            # motor stepped inside the run (e.g. sample rotation): 9 steps, rocking curve on the signal
            param = np.repeat(np.linspace(-1.0, 1.0, 9), int(np.ceil(n / 9)))[:n] + rng.normal(0, 0.003, n)
            rock = np.exp(-0.5 * ((param - 0.25) / 0.35) ** 2)
            resp = resp * (0.2 + 0.8 * rock)                   # TG efficiency; beam intensities unchanged
        images, overview = {}, {}
        for det, d in self.cfg.detectors.items():
            if d.kind == "agipd_overview":
                if rois is None:
                    overview = self._agipd_overview(rng, pump, probe, resp)
                continue
            if rois is not None and det not in rois:          # detector not asked for
                continue
            if d.kind == "agipd_tiles":
                per_tile = {t: self._agipd_tile(rng, t, pump, probe, resp) for t in d.tiles}
                images[det] = agipd.stack(per_tile, d.tiles)
                continue
            ny, nx = d.shape
            yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
            img = np.empty((n, ny, nx), np.float32)
            base = 2.0 + 0.3 * xx / nx                         # sloped background
            feats = SIM_FEATURES.get(det, SIM_FEATURES["jf1"])
            prof = {}
            for name, (fr, fc, sr, sc) in feats.items():
                prof[name] = np.exp(-((yy - fr * ny) ** 2 / (2 * (sr * ny) ** 2)
                                      + (xx - fc * nx) ** 2 / (2 * (sc * nx) ** 2)))
            gain = 1.0 if det == "jf1" else 0.4
            for i in range(n):
                frame = base.copy()
                for name, p in prof.items():
                    if name.startswith("Bragg"):
                        amp = 60 * gain * probe[i]
                    else:
                        amp = 25 * gain * pump[i] ** 2 * probe[i] * resp[i] + 0.4
                    frame += amp * p
                img[i] = frame + rng.normal(0, 1.0, (ny, nx)).astype(np.float32)
            images[det] = img
        spectra, axes = {}, {}
        own = {self.cfg.pump_spec: pump, self.cfg.probe_spec: probe}
        for i, (name, sp) in enumerate(self.cfg.spectrometers.items()):
            npx = sp.px[1] - sp.px[0]
            x = np.arange(npx, dtype=float)
            ped = np.exp(-0.5 * ((x - npx / 2) / (npx / 6)) ** 2)
            line = np.exp(-0.5 * ((x - npx / 2 - 3 * i) / max(npx / 160, 2.5)) ** 2)
            spikes = rng.gamma(1.5, 1 / 1.5, (n, npx))
            k = np.ones(7) / 7
            spikes = np.apply_along_axis(lambda r: np.convolve(r, k, "same"), 1, spikes)
            if self.cfg.intensity_from == "spectrometers" and name in own:
                # this spectrometer IS the arm's intensity monitor (p7016: HIREX = pump, DES = probe)
                spec = (6 * own[name][:, None] * ped * spikes + 60 * own[name][:, None] * line
                        + rng.normal(0, 1.0, (n, npx)))
            else:
                spec = (40 * sase[:, None] * ped * spikes + 120 * seed[:, None] * line
                        + rng.normal(0, 1.0, (n, npx)))
            spectra[name], axes[name] = spec.astype(np.float32), _spec_axis(sp)
        if self.cfg.intensity_from == "spectrometers":
            for arm, name in (("pump", self.cfg.pump_spec), ("probe", self.cfg.probe_spec)):
                if name in spectra:
                    val = spectrum_integral(spectra[name], self.cfg.spectrometers[name].n_edge)[0]
                    pump_d, probe_d = (val, probe_d) if arm == "pump" else (pump_d, val)
        xgm = None
        if self.cfg.xgm.source:
            xgm = (0.6 * sase + 0.4 * (pump_true + probe_true)) * 900 * (1 + rng.normal(0, 0.04, n))
        trains = np.arange(n) + scan * 10000
        stats = {}
        for det, (sig, bkg) in (rois or {}).items():     # not the reference run: keep only numbers
            if det in images:
                stats[det] = reduce_images(images.pop(det), trains, sig, bkg)
        return ScanData(scan, delay, images, pump_d, probe_d, crl if self.cfg.crl_enabled else {},
                        trains, spectra=spectra, spec_axis=axes,
                        overview=overview, xgm=xgm, param=param, stats=stats)

    # ------------------------------------------------------------ simulated AGIPD
    # tile -> features (row, col, sigma_row, sigma_col) in the stored 512 x 128 tile, kind
    SIM_TILES = {15: [(250, 64, 22, 9, "Bragg")], 8: [(300, 70, 18, 8, "TG")], 10: [(200, 58, 18, 8, "TG")]}

    def _tile_profile(self, t):
        yy, xx = np.mgrid[0:agipd.TILE_SHAPE[0], 0:agipd.TILE_SHAPE[1]].astype(np.float32)
        base = 1.5 + 0.2 * yy / agipd.TILE_SHAPE[0] + 0.05 * t
        feats = [(np.exp(-((yy - r) ** 2 / (2 * sr ** 2) + (xx - c) ** 2 / (2 * sc ** 2))), kind)
                 for r, c, sr, sc, kind in self.SIM_TILES.get(t, [])]
        return base, feats

    @staticmethod
    def _amp(kind, pump, probe, resp):
        return 60 * probe if kind == "Bragg" else 25 * pump ** 2 * probe * resp + 0.4

    def _agipd_tile(self, rng, t, pump, probe, resp):
        base, feats = self._tile_profile(t)
        out = np.empty((len(pump),) + agipd.TILE_SHAPE, np.float32)
        for i in range(len(pump)):
            frame = base.copy()
            for prof, kind in feats:
                frame += self._amp(kind, pump[i], probe[i], resp[i]) * prof
            out[i] = frame + rng.normal(0, 1.0, agipd.TILE_SHAPE).astype(np.float32)
        return out

    def _agipd_overview(self, rng, pump, probe, resp):
        out = {}
        for t in range(agipd.N_TILES):
            base, feats = self._tile_profile(t)
            m = base.copy()
            for prof, kind in feats:
                m += float(np.mean(self._amp(kind, pump, probe, resp))) * prof
            out[t] = (m + rng.normal(0, 0.15, agipd.TILE_SHAPE)).astype(np.float32)
        return out


def _spec_axis(sp):
    pix = np.arange(sp.px[0], sp.px[1], dtype=float)
    return pix if sp.eV0 is None or sp.dE is None else sp.eV0 + sp.dE * pix


# ---------------------------------------------------------------- extra_data
def reduce_pulses(a: np.ndarray, src: Source, image_ndim: int = 0) -> np.ndarray:
    """Collapse the pulse/cell axis (axis 1) if there is one.

    ``image_ndim`` = trailing axes that belong to the value itself (2 for images).
    """
    a = np.asarray(a)
    extra = a.ndim - 1 - image_ndim
    if extra <= 0:
        return a.astype(np.float32 if image_ndim else np.float64) * src.scale + src.offset
    if extra > 1:
        a = a.reshape(a.shape[0], -1, *a.shape[a.ndim - image_ndim:])
    if src.pulses != "all":
        lo, _, hi = src.pulses.partition(":")
        a = a[:, slice(int(lo) if lo else None, int(hi) if hi else None)]
    if a.dtype.kind == "f":               # nan-sum / nan-mean without the copies np.nanmean makes
        fin = ~np.isnan(a)
        out = np.sum(a, axis=1, where=fin, dtype=np.float64 if not image_ndim else a.dtype)
        if src.reduce != "sum":
            with np.errstate(invalid="ignore", divide="ignore"):
                out = out / fin.sum(axis=1)
    else:
        out = a.sum(axis=1) if src.reduce == "sum" else a.mean(axis=1)
    return out.astype(np.float32 if image_ndim else np.float64) * src.scale + src.offset


CHUNK = 32                     # trains read at a time from the run files


def per_train_mean(tids, arr):
    """Average all entries that share a trainId (e.g. several AGIPD frames per train).
    Returns (sorted unique trainIds, (n_trains, ...) means)."""
    tids = np.asarray(tids)
    u, inv = np.unique(tids, return_inverse=True)
    if len(u) == len(tids):
        o = np.argsort(tids)
        return tids[o], np.asarray(arr)[o]
    arr = np.asarray(arr)
    out = np.zeros((len(u),) + arr.shape[1:], np.float64)
    cnt = np.zeros((len(u),) + arr.shape[1:], np.int64)
    fin = np.isfinite(arr)
    np.add.at(out, inv, np.where(fin, arr, 0))        # nan-mean: NaN (masked) entries do not count
    np.add.at(cnt, inv, fin)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, out / cnt, np.nan)
    return u, mean.astype(arr.dtype if arr.dtype.kind == "f" else np.float64)


class ExtraDataBackend(Backend):
    """Processed run files on Maxwell.  Scan number = run number."""
    name = "extra_data"

    def load_param(self, scan, trains):
        """Read only the scanned parameter (fast: no detector data)."""
        import extra_data as ex
        cfg = self.cfg
        if trains is None or not cfg.param.source:
            return None
        run = ex.open_run(cfg.proposal, scan, data=cfg.data)
        kd = run[cfg.param.source, cfg.param.key]
        t, a = per_train_mean(kd.train_id_coordinates(), reduce_pulses(kd.ndarray(), cfg.param))
        return match_trains(t, a, trains)

    def load_scan(self, scan: int, rois=None) -> ScanData:
        import extra_data as ex
        cfg = self.cfg
        if not cfg.proposal:
            raise ValueError("set the proposal number first")
        run = ex.open_run(cfg.proposal, scan, data=cfg.data)
        if cfg.max_trains:
            run = run.select_trains(ex.by_index[: cfg.max_trains])

        masks = {}                     # name -> [mask key, masked values, all values] or a message
        fixed_masks = {}               # (source, mask key, roi) -> bad pixels read once (mask_per_train off)

        def masked(name, kd, a, source, data_key, mask_key, roi=None):
            """a (float, read from kd = run[source, data_key]) with NaN where the bad-pixel
            mask next to it (mask_key, "auto" = <group>.mask) is > 0."""
            mk = mask_key_for(data_key, mask_key) if cfg.apply_mask else ""
            if not mk:
                return a
            fixed = not getattr(cfg, "mask_per_train", False)
            fkey = (source, mk, None if roi is None else repr(roi))
            if fixed and fkey not in fixed_masks:
                stored = maskstore.load(cfg.proposal, fkey)           # read once for the experiment
                if stored is not None and stored.shape == a.shape[1:]:
                    fixed_masks[fkey] = stored
                    masks[name] = [f"{mk} (from {maskstore.path_for(cfg.proposal).name})", 0, 0]
            if fixed and fkey in fixed_masks:
                m0 = fixed_masks[fkey]
                if m0 is None or m0.shape != a.shape[1:]:
                    return a
                a = np.asarray(a, np.float32 if a.dtype != np.float64 else np.float64)
                a[:, m0] = np.nan
                info = masks.get(name)
                if isinstance(info, list):
                    info[1] += int(m0.sum()) * len(a)
                    info[2] += a.size
                return a
            try:
                mkd = run[source, mk]
            except Exception:
                masks.setdefault(name, f"no {mk} in the run: not masked")
                return a
            tids, mtids = np.asarray(kd.train_id_coordinates()), np.asarray(mkd.train_id_coordinates())
            if mtids.shape != tids.shape or np.any(mtids != tids):      # kd is a chunk / selection
                mkd = mkd.select_trains(ex.by_id[np.unique(tids).tolist()])
            m = (mkd.ndarray() if roi is None else mkd.ndarray(roi=roi)) > 0
            if m.shape != a.shape:
                masks[name] = f"{mk} has shape {m.shape}, data {a.shape}: not masked"
                if fixed:
                    fixed_masks[fkey] = None
                return a
            if fixed:                                                 # bad in most frames = bad pixel
                m0 = m.mean(axis=0) > 0.5
                fixed_masks[fkey] = m0
                maskstore.save(cfg.proposal, fkey, m0)
                m = np.broadcast_to(m0, m.shape)
            info = masks.get(name)
            info = info if isinstance(info, list) else [mk, 0, 0]
            info[1] += int(m.sum())
            info[2] += m.size
            masks[name] = info
            a = np.asarray(a, np.float32 if a.dtype != np.float64 else np.float64)
            a[m] = np.nan
            return a

        def read(src: Source, image_ndim=0, optional=False):
            if not src.source:
                return None
            try:
                kd = run[src.source, src.key]
            except Exception:
                if optional:
                    return None
                raise
            a = reduce_pulses(kd.ndarray(), src, image_ndim)
            return per_train_mean(kd.train_id_coordinates(), a)      # one value per train

        def read_tile(src: Source, tile: int, trains=None, name="", mask_key=""):
            """(trainIds, (n, 512, 128)) frames of one AGIPD tile averaged per train."""
            source = src.source.format(tile=tile)
            kd = run[source, src.key]
            if trains:
                kd = kd.select_trains(ex.by_index[:trains])
            parts_t, parts_a = [], []
            chunks = kd.split_trains(trains_per_part=32) if hasattr(kd, "split_trains") else [kd]
            for ch in chunks:                                     # chunks keep the memory low
                a = np.asarray(ch.ndarray(), np.float32)
                a = masked(f"{name} tile {tile}", ch, a, source, src.key, mask_key)
                a = a.reshape(-1, *a.shape[-2:])
                t, m = per_train_mean(ch.train_id_coordinates(), a)
                parts_t.append(t)
                parts_a.append(m.astype(np.float32))
            return np.concatenate(parts_t), np.concatenate(parts_a)

        notes = []

        def photon_energy():
            """keV for photonizing: the fixed value, else the median of the undulator setting."""
            if cfg.photon_energy_keV and cfg.photon_energy_keV > 0:
                return float(cfg.photon_energy_keV), "fixed in the config"
            src = cfg.photon_energy
            kd = run[src.source, src.key]
            v = np.asarray(kd.ndarray(), float)
            e = float(np.nanmedian(v)) * src.scale + src.offset
            if np.nanmax(v) - np.nanmin(v) > 1e-3 * abs(e):
                notes.append(f"photon energy changed during the run ({np.nanmin(v):g} … {np.nanmax(v):g}): "
                             f"median {e:g} keV used")
            if not np.isfinite(e) or e <= 0:
                raise ValueError(f"photon energy {e} from {src.source} / {src.key}")
            return e, f"{src.source} / {src.key}"

        E_ph = np.nan
        if cfg.photonize and any(d.kind == "image" for d in cfg.detectors.values()):
            try:
                E_ph, where = photon_energy()
                notes.append(f"Jungfraus photonized with {E_ph:g} keV ({where})")
            except Exception as e:
                notes.append(f"photon energy not read ({type(e).__name__}: {e}): Jungfraus NOT photonized")

        def read_image(name, det, reducer=None):
            """(trainIds, (n, ny, nx)) per-train images of a camera-like detector (Jungfrau), read
            in chunks of trains: bad-pixel mask, then photonizing, per frame, then the
            memory-cell / pulse average and the train average.  With a reducer the chunks
            go into it and only the trainIds come back: (trainIds, None)."""
            src = det.data
            kd = run[src.source, src.key]
            parts_t, parts_a = [], []
            chunks = kd.split_trains(trains_per_part=CHUNK) if hasattr(kd, "split_trains") else [kd]
            for ch in chunks:                                     # chunks keep the memory low
                a = np.asarray(ch.ndarray(), np.float32)
                a = masked(name, ch, a, src.source, src.key, det.mask_key)
                if np.isfinite(E_ph):
                    a = photonize(a, E_ph)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN pixels
                    a = reduce_pulses(a, src, image_ndim=2)
                t, m = per_train_mean(ch.train_id_coordinates(), a)
                parts_t.append(t)
                if reducer is None:
                    parts_a.append(m.astype(np.float32))
                else:
                    reducer.add(t, m)
            t = np.concatenate(parts_t) if parts_t else np.array([], np.uint64)
            return (t, np.concatenate(parts_a)) if reducer is None else (np.unique(t), None)

        def read_stack_reduced(name, det, reducer):
            """AGIPD tiles of a non-reference run: read all tiles one chunk of trains at a
            time, stack them as in the analysis view and reduce; returns the trainIds."""
            kds = {t: run[det.data.source.format(tile=t), det.data.key] for t in det.tiles}
            tids = None
            for kd in kds.values():
                u = np.unique(np.asarray(kd.train_id_coordinates()))
                tids = u if tids is None else np.intersect1d(tids, u)
            for i in range(0, len(tids), CHUNK):
                chunk = tids[i:i + CHUNK]
                per = {}
                for t, kd in kds.items():
                    ch = kd.select_trains(ex.by_id[chunk.tolist()])
                    a = np.asarray(ch.ndarray(), np.float32)
                    a = masked(f"{name} tile {t}", ch, a, det.data.source.format(tile=t), det.data.key,
                               det.mask_key)
                    a = a.reshape(-1, *a.shape[-2:])
                    tt, m = per_train_mean(ch.train_id_coordinates(), a)
                    full = np.full((len(chunk),) + m.shape[1:], np.nan, np.float32)
                    j = np.searchsorted(chunk, tt)
                    full[j] = m
                    per[t] = full
                reducer.add(chunk, agipd.stack(per, det.tiles))
            return tids, None

        parts, overview = {}, {}
        reducers = {name: ImageReducer(*rois[name]) for name in (rois or {}) if name in cfg.detectors}
        if cfg.intensity_from == "diodes":
            parts["pump"], parts["probe"] = read(cfg.pump), read(cfg.probe)
        tile_parts = {}
        for name, det in cfg.detectors.items():
            if det.kind == "agipd_overview":
                if rois is not None:                              # light run: the all-tiles view uses the reference run
                    continue
                for t in det.tiles:
                    try:
                        _, a = read_tile(det.data, t, cfg.overview_trains or None, name, det.mask_key)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)   # always-masked pixels
                            overview[t] = np.nanmean(a, axis=0).astype(np.float32)
                    except Exception:
                        continue                                  # tile not recorded
            elif rois is not None and name not in reducers:       # detector not asked for
                continue
            elif det.kind == "agipd_tiles" and name in reducers:
                parts[name] = read_stack_reduced(name, det, reducers[name])
            elif det.kind == "agipd_tiles":
                tile_parts[name] = {t: read_tile(det.data, t, name=name, mask_key=det.mask_key) for t in det.tiles}
                tids = None
                for t, (tt, _) in tile_parts[name].items():
                    tids = tt if tids is None else np.intersect1d(tids, tt)
                parts[name] = (tids, None)                        # stacked after train matching
            else:
                parts[name] = read_image(name, det, reducers.get(name))
        if cfg.crl_enabled:
            for k in cfg.CRL_KEYS:
                parts[k] = read(getattr(cfg, k), optional=True)
        parts["xgm"] = read(cfg.xgm, optional=True)
        param_tv = None                                    # not used to choose the trains: NaN where missing
        if cfg.param_enabled and cfg.param.source:
            param_tv = read(cfg.param, optional=True)       # wrong name -> scan loads, page says "no parameter"
        spec_names = []
        if cfg.spec_enabled or cfg.intensity_from == "spectrometers":
            for name, sp in cfg.spectrometers.items():
                try:
                    kd = run[sp.data.source, sp.data.key]
                except Exception:
                    if name in (cfg.pump_spec, cfg.probe_spec) and cfg.intensity_from == "spectrometers":
                        raise
                    continue
                # data.adc[frame, px0:px1] = the notebook's np.s_[a, b:c] (a = frame in the train)
                roi = np.s_[sp.px[0]:sp.px[1]] if sp.frame < 0 else np.s_[sp.frame, sp.px[0]:sp.px[1]]
                a = kd.ndarray(roi=roi).astype(np.float32)   # ndarray(roi): avoids the VDS/units crash
                a = masked(name, kd, a, sp.data.source, sp.data.key, sp.mask_key, roi=roi)      # HIREX: data.mask > 0 -> NaN
                parts[name] = per_train_mean(kd.train_id_coordinates(), a.reshape(len(a), -1))
                spec_names.append(name)
        common = None
        for v in parts.values():
            if v is not None:
                common = v[0] if common is None else np.intersect1d(common, v[0])
        if common is None or len(common) == 0:
            raise ValueError(f"run {scan}: no trains common to all sources")

        def pick(v):
            return None if v is None or v[1] is None else v[1][np.searchsorted(v[0], common)]

        images, stats = {}, {}
        for name, det in cfg.detectors.items():
            if name in reducers:
                stats[name] = reducers[name].finish(common)[0]
            elif det.kind == "agipd_tiles" and name in tile_parts:
                per = {t: tt_a[1][np.searchsorted(tt_a[0], common)] for t, tt_a in tile_parts[name].items()}
                images[name] = agipd.stack(per, det.tiles)
            elif det.kind == "image" and name in parts:
                images[name] = pick(parts[name])
        spectra = {n: pick(parts[n]) for n in spec_names}
        if cfg.intensity_from == "spectrometers":
            pump = spectrum_integral(spectra[cfg.pump_spec], cfg.spectrometers[cfg.pump_spec].n_edge)[0]
            probe = spectrum_integral(spectra[cfg.probe_spec], cfg.spectrometers[cfg.probe_spec].n_edge)[0]
        else:
            pump, probe = pick(parts["pump"]), pick(parts["probe"])
        delay = np.nan
        if cfg.delay.source and not cfg.delay.placeholder:
            try:
                kd = run[cfg.delay.source, cfg.delay.key]
                delay = float(np.nanmedian(kd.ndarray())) * cfg.delay.scale + cfg.delay.offset
            except Exception:
                pass
        crl = {k: pick(parts[k]) for k in cfg.CRL_KEYS if parts.get(k) is not None}
        return ScanData(scan, delay, images, pump, probe, crl, common, spectra=spectra,
                        spec_axis={n: _spec_axis(cfg.spectrometers[n]) for n in spec_names},
                        overview=overview, xgm=pick(parts.get("xgm")),
                        param=None if param_tv is None else match_trains(*param_tv, common),
                        masks={n: (f"{v[0]}: {v[1]} of {v[2]} values masked" if isinstance(v, list) else v)
                               for n, v in masks.items()},
                        photon_energy=E_ph, notes=notes, stats=stats)


def match_trains(t, a, trains):
    """Per-train values a (at sorted trainIds t) on the trains of a scan; NaN where missing."""
    trains = np.asarray(trains)
    out = np.full(len(trains), np.nan)
    t = np.asarray(t)
    if len(t) == 0:
        return out
    a = np.asarray(a, float).reshape(len(a), -1)[:, 0]
    i = np.clip(np.searchsorted(t, trains), 0, len(t) - 1)
    hit = t[i] == trains
    out[hit] = a[i[hit]]
    return out


def make_backend(kind: str, cfg: SourceConfig, **kw) -> Backend:
    if kind == "simulated":
        return SimulatedBackend(cfg, **kw)
    if kind == "extra_data":
        return ExtraDataBackend(cfg)
    raise ValueError(kind)


def parse_scans(text: str) -> list[int]:
    """'101-105 110 112' -> [101, 102, 103, 104, 105, 110, 112]."""
    out = []
    for tok in text.replace(",", " ").split():
        if "-" in tok[1:]:
            a, b = tok.split("-", 1)
            a, b = int(a), int(b)
            out.extend(range(min(a, b), max(a, b) + 1))
        else:
            out.append(int(tok))
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------- several runs at once
def _load_one(cls, cfg, scan, rois):
    """Worker process: read one run (real data) and send back its ScanData."""
    return cls(cfg).load_scan(scan, rois=rois)


def default_workers():
    """Parallel readers: XTG_MID_WORKERS if set, else half the cores (max 8)."""
    try:
        return max(1, int(os.environ["XTG_MID_WORKERS"]))
    except (KeyError, ValueError):
        n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 2)
        return max(1, min(8, n // 2))


def run_jobs(backend, jobs, here, remote, workers=None, tick=None):
    """Run one job per run; yields (scan, result or the exception) as each one is done.

    jobs: [(scan, arg)].  Real data with more than one run go to separate processes:
    remote(type(backend), backend.cfg, scan, *arg) (or (..., scan, arg) if arg is not a
    tuple), `workers` at a time; otherwise here(scan, arg) runs in this process, one run
    after the other.  tick() is called about 10 times a second while waiting (GUI)."""
    jobs = list(jobs)
    workers = default_workers() if workers is None else workers
    if not isinstance(backend, ExtraDataBackend) or workers <= 1 or len(jobs) <= 1:
        for s, arg in jobs:
            if tick:
                tick()
            try:
                yield s, here(s, arg)
            except Exception as e:      # noqa: BLE001 - reported by the caller
                yield s, e
        return
    import concurrent.futures as cf
    import multiprocessing as mp
    with cf.ProcessPoolExecutor(min(workers, len(jobs)), mp_context=mp.get_context("spawn")) as pool:
        futs = {pool.submit(remote, type(backend), backend.cfg, s, *(arg if isinstance(arg, tuple) else (arg,))): s
                for s, arg in jobs}
        pending = set(futs)
        while pending:
            done, pending = cf.wait(pending, timeout=0.1, return_when=cf.FIRST_COMPLETED)
            for f in done:
                try:
                    yield futs[f], f.result()
                except Exception as e:  # noqa: BLE001
                    yield futs[f], e
            if tick:
                tick()


def load_many(backend, jobs, workers=None, tick=None):
    """Read several runs into memory; yields (scan, ScanData or the exception).
    jobs: [(scan, rois)] (see Backend.load_scan)."""
    yield from run_jobs(backend, [(s, (r,)) for s, r in jobs], lambda s, a: backend.load_scan(s, rois=a[0]),
                        _load_one, workers=workers, tick=tick)
