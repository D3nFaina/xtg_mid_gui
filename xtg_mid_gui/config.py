"""Source names and settings: the ONLY place that knows what the data is called.

The real MID source / key names for the next beamtime are not known yet, so
every name below is a placeholder.  When the names are known, either edit the
defaults here or (better) write a YAML file and start the GUI with
``--config sources.yaml``.  ``python -m xtg_mid_gui --write-config sources.yaml``
dumps the current defaults as a starting point.

Everything downstream (backends, analysis, GUI) refers to the logical names
``jf1``, ``jf2``, ``pump``, ``probe``, ``spec1``, ``crl_pump_x`` ... and never to a
Karabo device name.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

try:
    import yaml
except ImportError:          # pragma: no cover - yaml ships with exfel-python
    yaml = None
import json


@dataclass
class Source:
    """One Karabo source/key pair plus how to reduce it to one value per train."""
    source: str
    key: str
    # For per-pulse / per-cell data: which slice along the first non-train axis
    # to keep ("all" = average everything, or "a:b" python slice).
    pulses: str = "all"
    reduce: str = "mean"          # "mean" or "sum" over the kept pulses / cells
    scale: float = 1.0            # value -> physical units (diode V -> uJ, mm -> um ...)
    offset: float = 0.0

    @property
    def placeholder(self) -> bool:
        return self.source.startswith("PLACEHOLDER")


@dataclass
class Detector:
    label: str                     # what the GUI shows, e.g. "Jungfrau 1"
    data: Source
    shape: tuple = (100, 160)      # simulated image shape (rows, cols); real data sets its own
    # "image"         : one camera image per train (Jungfrau)
    # "agipd_overview": all AGIPD tiles, mean image per scan, drawn as on the real detector
    #                   (no ROIs; click a tile to add / remove it from the analysis subset)
    # "agipd_tiles"   : the user-chosen AGIPD tiles, one image per train, stacked; ROIs live here
    # For the AGIPD kinds, data.source contains "{tile}", e.g. MID_DET_AGIPD1M-1/CORR/{tile}CH0:output
    kind: str = "image"
    tiles: list = field(default_factory=list)
    # Bad-pixel mask next to the data (pixels with mask > 0 become NaN when apply_mask is on).
    # "auto" = same group as data.key: image.data -> image.mask, data.adc -> data.mask;
    # "" = no mask.  A mask key that is not in the run is ignored.
    mask_key: str = "auto"


@dataclass
class Spectrometer:
    """Upstream (before the SDL) single-shot spectrometer."""
    label: str
    data: Source
    frame: int = -1                # row of a 2D readout to use (-1 = data are already 1D)
    px: tuple = (0, 400)           # pixel window [lo, hi)
    n_edge: int = 20               # edge pixels per side for the baseline subtraction
    eV0: float | None = None       # optional calibration E = eV0 + dE * px
    dE: float | None = None
    mask_key: str = "auto"         # bad-pixel mask (HIREX: data.mask), as Detector.mask_key


def mask_key_for(data_key: str, mask_key: str) -> str:
    """The mask key to read for a data key ("" = none): "auto" -> <group>.mask."""
    if mask_key != "auto":
        return mask_key or ""
    group, dot, _ = data_key.rpartition(".")
    return f"{group}.mask" if dot else ""


def _spec(n):
    return Spectrometer(f"Upstream spectrometer {n}",
                        Source(f"PLACEHOLDER_SA2_XTD_SPEC{n}/CORR/RECEIVER:daqOutput", "data.adc"))


def _jf(n):
    return Detector(f"Jungfrau {n}", Source(f"PLACEHOLDER_MID_EXP_JF{n}/CORR/JNGFR0{n}:daqOutput",
                                            "data.adc", pulses="all", reduce="mean"))


@dataclass
class SourceConfig:
    # --- where the processed data are ---------------------------------
    proposal: int = 0                         # 0 = not set yet
    data: str = "proc"                        # extra_data: "proc", "raw" or "all"
    # --- detectors: logical name -> detector ---------------------------
    detectors: dict = field(default_factory=lambda: {"jf1": _jf(1), "jf2": _jf(2)})
    # --- diamond diodes: pump and probe arm intensity -----------------
    pump: Source = field(default_factory=lambda: Source(
        "PLACEHOLDER_MID_EXP_SDL/DIODE/PUMP:output", "data.intensity"))
    probe: Source = field(default_factory=lambda: Source(
        "PLACEHOLDER_MID_EXP_SDL/DIODE/PROBE:output", "data.intensity"))
    # --- one delay per scan (fs).  Empty source -> typed in the scan table
    delay: Source = field(default_factory=lambda: Source(
        "PLACEHOLDER_MID_EXP_SDL/MDL/DELAY", "actualPosition.value", scale=1.0))
    # --- upstream spectrometers (before the SDL): integral vs the diodes
    spec_enabled: bool = True
    spectrometers: dict = field(default_factory=lambda: {"spec1": _spec(1)})   # one, before the SDL
    # --- CRL beam-position monitors -----------------------------------
    # Pointing of each arm before the SDL, kept centred by a feedback loop.
    # Empty source -> page shows "no data".
    crl_enabled: bool = True
    crl_pump_x: Source = field(default_factory=lambda: Source(
        "PLACEHOLDER_MID_OPT_CRL/BPM/PUMP:output", "data.x"))
    crl_pump_y: Source = field(default_factory=lambda: Source(
        "PLACEHOLDER_MID_OPT_CRL/BPM/PUMP:output", "data.y"))
    crl_probe_x: Source = field(default_factory=lambda: Source(
        "PLACEHOLDER_MID_OPT_CRL/BPM/PROBE:output", "data.x"))
    crl_probe_y: Source = field(default_factory=lambda: Source(
        "PLACEHOLDER_MID_OPT_CRL/BPM/PROBE:output", "data.y"))
    max_trains: int = 0                       # 0 = all trains of a scan
    # --- bad-pixel masks of the detectors and spectrometers (see Detector.mask_key):
    #     masked pixels (mask > 0) are set to NaN and left out of every sum / mean
    apply_mask: bool = True
    # --- photonize the Jungfraus (detectors of kind "image"; corrected data in keV):
    #     per frame, before the memory-cell / train average: values < 0 -> NaN,
    #     others -> round(value / photon energy), i.e. photon counts
    photonize: bool = False
    # photon energy in keV: the undulator setting (median over the run) ...
    photon_energy: Source = field(default_factory=lambda: Source(
        "MID_XTD1_UND/DOOCS/ENERGY", "actualPosition.value"))
    photon_energy_keV: float = 0.0            # ... or this fixed value if > 0
    # --- which proposal preset this config belongs to ("7016", "11456" or "custom")
    preset: str = "custom"
    # --- where the pump / probe intensities come from:
    #     "diodes"        -> the pump / probe sources above (diamond diodes, 11456)
    #     "spectrometers" -> integral of the spectrometers named in pump_spec / probe_spec
    #                        (HIREX = pump, DES = probe, as in p7016)
    intensity_from: str = "diodes"
    pump_spec: str = ""
    probe_spec: str = ""
    # --- gas monitor (XGM): sees pump + probe.  Empty source -> not shown.
    xgm: Source = field(default_factory=lambda: Source("", "data.intensityTD", pulses="0:1"))
    overview_trains: int = 200                # trains averaged for the all-tiles AGIPD view (0 = all)
    # --- parameter scanned INSIDE one run (sample rotation, grating-sample distance,
    #     temperature ...), read per train.  Empty source -> page off.
    param_enabled: bool = False
    param_label: str = "parameter"
    param_unit: str = ""
    param: Source = field(default_factory=lambda: Source("", ""))

    CRL_KEYS = ("crl_pump_x", "crl_pump_y", "crl_probe_x", "crl_probe_y")

    # ------------------------------------------------------------ io
    def to_dict(self) -> dict:
        d = asdict(self)
        for det in d["detectors"].values():
            det["shape"] = list(det["shape"])
        for sp in d["spectrometers"].values():
            sp["px"] = list(sp["px"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SourceConfig":
        cfg = cls()
        names = {f.name for f in fields(cls)}
        for k, v in d.items():
            if k not in names:
                raise KeyError(f"unknown config key {k!r}")
            if k == "detectors":
                cfg.detectors = {
                    name: Detector(det.get("label", name), Source(**det["data"]),
                                   tuple(det.get("shape", (100, 160))), det.get("kind", "image"),
                                   [int(t) for t in det.get("tiles", [])],
                                   det.get("mask_key", "auto") or "")
                    for name, det in v.items()}
            elif k == "spectrometers":
                cfg.spectrometers = {
                    name: Spectrometer(sp.get("label", name), Source(**sp["data"]),
                                       **{a: (tuple(b) if a == "px" else b) for a, b in sp.items()
                                          if a not in ("label", "data")})
                    for name, sp in v.items()}
            elif isinstance(getattr(cfg, k), Source):
                setattr(cfg, k, Source(**v))
            else:
                setattr(cfg, k, v)
        return cfg

    def save(self, path):
        path = Path(path)
        txt = (yaml.safe_dump(self.to_dict(), sort_keys=False) if yaml and path.suffix in (".yaml", ".yml")
               else json.dumps(self.to_dict(), indent=2))
        path.write_text(txt)

    @classmethod
    def load(cls, path) -> "SourceConfig":
        path = Path(path)
        txt = path.read_text()
        if path.suffix in (".yaml", ".yml") and yaml is None:
            raise ImportError(f"reading {path.name} needs PyYAML: pip install pyyaml")
        d = yaml.safe_load(txt) if path.suffix in (".yaml", ".yml") else json.loads(txt)
        return cls.from_dict(d or {})

    def placeholders(self) -> list[str]:
        """Logical names whose source is still a placeholder."""
        out = [f"{n} ({d.label})" for n, d in self.detectors.items() if d.data.placeholder]
        out += [f"{n} ({d.label})" for n, d in self.spectrometers.items() if d.data.placeholder]
        keys = ("delay",) + (("pump", "probe") if self.intensity_from == "diodes" else ())
        keys += self.CRL_KEYS if self.crl_enabled else ()
        for k in keys + ("xgm",):
            if getattr(self, k).placeholder:
                out.append(k)
        return out

    # ------------------------------------------------------------ helpers
    def image_detectors(self) -> dict:
        """Detectors that carry per-shot images and ROIs (everything but the AGIPD overview)."""
        return {n: d for n, d in self.detectors.items() if d.kind != "agipd_overview"}

    def overview_detector(self):
        return next(((n, d) for n, d in self.detectors.items() if d.kind == "agipd_overview"), (None, None))


# ---------------------------------------------------------------- in-run parameter presets
# From the p7016 notebooks (rocking_MID.ipynb): label -> (source, key, unit)
PARAM_PRESETS = {
    "phi, sample rotation (motor T3)": ("MID_EXP_UPP/MDL/DATA_SELECTOR_1", "midExpUppMotorT3.actualPosition", "deg"),
    "motor T2 (theta?)": ("MID_EXP_UPP/MDL/DATA_SELECTOR_1", "midExpUppMotorT2.actualPosition", ""),
    "sample temperature (Cobra gas)": ("MID_EXP_UPP/MDL/DATA_SELECTOR_2",
                                       "midSaeCryoCryoCobra800.statusNode_gasTemp.value", "K"),
}


# ---------------------------------------------------------------- proposal presets
PRESETS = {"7016": "7016 (previous beamtime)", "11456": "11456 (this beamtime)", "custom": "custom"}
PRESET_DIR = Path(__file__).with_name("presets")


def _search_dirs():
    """Where a sources file is looked for: the folder run_xtg_mid_gui.sh was started from
    (the proposal usr/ folder on Maxwell), the current folder, the folder that holds the
    code, then the presets shipped with it."""
    import os
    start = os.environ.get("XTG_MID_START_DIR")           # set by run_xtg_mid_gui.sh
    try:
        cwd = [Path.cwd()]
    except OSError:                                       # current folder was deleted
        cwd = []
    dirs = ([Path(start)] if start else []) + cwd + [Path(__file__).resolve().parents[1], PRESET_DIR]
    return list(dict.fromkeys(dirs))


def _find_in(d: Path, name: str):
    """d/name, also with other letter case (Maxwell is case-sensitive: sources_7016.yaml)."""
    if (d / name).is_file():
        return d / name
    try:
        return next((f for f in d.iterdir() if f.name.lower() == name.lower() and f.is_file()), None)
    except OSError:
        return None


def find_preset_file(preset: str):
    names = {"7016": ["Sources_7016.yaml"], "11456": ["Sources_11456.yaml"]}.get(preset, [])
    for d in _search_dirs():
        for n in names:
            f = _find_in(d, n)
            if f is not None:
                return f
    return None


def find_sources_file(path):
    """A sources file given by the user (--config / file dialog): as typed, else the same
    name in the start folder, the current folder, the code folder or the shipped presets."""
    p = Path(path).expanduser()
    if p.is_file():
        return p
    for d in _search_dirs():
        f = _find_in(d, p.name)
        if f is not None:
            return f
    raise FileNotFoundError(f"cannot find {path}: looked as typed and in "
                            + ", ".join(str(d) for d in _search_dirs()))


def preset_of_file(path):
    """'Sources_7016.yaml' -> '7016' (any folder, any letter case), else None."""
    n = Path(path).name.lower()
    return next((k for k in ("7016", "11456") if n == f"sources_{k}.yaml"), None)


def config_for(preset: str, custom_path=None) -> "SourceConfig":
    """Config of a proposal preset.

    * 7016  : Sources_7016.yaml (AGIPD tiles, HIREX = pump, DES = probe, XGM)
    * 11456 : Sources_11456.yaml; only the first upstream spectrometer is kept
              (one spectrometer before the SDL)
    * custom: custom_path if given, else the built-in defaults
    """
    if preset == "custom" and custom_path and preset_of_file(custom_path):
        preset = preset_of_file(custom_path)          # --config Sources_7016.yaml = proposal 7016
        if not Path(custom_path).expanduser().is_file():
            custom_path = None                        # found by name below
    if preset != "custom" and custom_path:            # an explicit file for a proposal preset
        path = find_sources_file(custom_path)
        cfg = SourceConfig.load(path)
        cfg.preset, cfg.proposal, cfg._source_file = preset, int(preset), str(path)
        if preset == "11456" and len(cfg.spectrometers) > 1:
            first = next(iter(cfg.spectrometers))
            cfg.spectrometers = {first: cfg.spectrometers[first]}
        return cfg
    if preset == "custom":
        cfg = SourceConfig.load(find_sources_file(custom_path)) if custom_path else SourceConfig()
        cfg.preset = "custom"
        return cfg
    path = find_preset_file(preset)
    cfg = SourceConfig.load(path) if path else SourceConfig()
    cfg.preset = preset
    cfg.proposal = int(preset)
    cfg._source_file = str(path) if path else ""
    if preset == "11456" and len(cfg.spectrometers) > 1:
        first = next(iter(cfg.spectrometers))
        cfg.spectrometers = {first: cfg.spectrometers[first]}
    return cfg
