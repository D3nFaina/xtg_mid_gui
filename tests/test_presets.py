"""Proposal presets (7016 / 11456 / custom), AGIPD tile layout, and the extra_data
reader for p7016 (AGIPD tiles, HIREX / DES data.adc[frame, b:c], XGM) on fake runs."""
import sys
import types

import numpy as np
import pytest

from xtg_mid_gui import agipd
from xtg_mid_gui import analysis as an
from xtg_mid_gui.backends import ExtraDataBackend, SimulatedBackend, per_train_mean
from xtg_mid_gui.config import SourceConfig, config_for


def test_preset_7016_matches_the_p7016_notebook():
    cfg = config_for("7016")
    assert cfg.proposal == 7016 and cfg.preset == "7016" and cfg.intensity_from == "spectrometers"
    kinds = {n: d.kind for n, d in cfg.detectors.items()}
    assert kinds == {"agipd": "agipd_overview", "agipd_sub": "agipd_tiles"}
    assert cfg.detectors["agipd"].tiles == list(range(16))
    assert cfg.detectors["agipd"].data.source == "MID_DET_AGIPD1M-1/CORR/{tile}CH0:output"
    h, d = cfg.spectrometers[cfg.pump_spec], cfg.spectrometers[cfg.probe_spec]
    assert h.data.source.startswith("SA2_XTD6_HIREX") and (h.frame, h.px, h.n_edge) == (20, (1750, 2250), 50)
    assert d.data.source.startswith("MID_EXP_DES") and (d.frame, d.px) == (1, (1100, 1250))
    assert cfg.xgm.source == "SA2_XTD1_XGM/XGM/DOOCS:output" and cfg.xgm.pulses == "0:1"
    assert not cfg.crl_enabled and cfg.placeholders() == []


def test_preset_11456_uses_Sources_11456_yaml_with_one_spectrometer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = config_for("11456")                       # no Sources_11456.yaml here: template
    assert cfg.proposal == 11456 and len(cfg.spectrometers) == 1
    two = SourceConfig()
    two.spectrometers["spec2"] = two.spectrometers["spec1"]
    two.pump.source = "MID_REAL/DIODE/PUMP:output"
    two.save(tmp_path / "Sources_11456.yaml")
    cfg = config_for("11456")
    assert cfg._source_file == str(tmp_path / "Sources_11456.yaml")
    assert cfg.pump.source == "MID_REAL/DIODE/PUMP:output"
    assert list(cfg.spectrometers) == ["spec1"]     # only the spectrometer before the SDL


def test_agipd_layout_and_stack():
    o = agipd.module_origins()
    assert sorted(o) == list(range(16))
    canvas = np.zeros(agipd.canvas_shape(), int)
    mh, mw = agipd.module_size()
    assert (mh, mw) == (128, 8 * 64 + 7 * 2)        # 8 ASICs with 2 px gaps
    for t, (r0, c0) in o.items():                    # tiles do not overlap
        canvas[r0:r0 + mh, c0:c0 + mw] += 1
        assert agipd.module_at(r0 + 5, c0 + 5) == t
    assert canvas.max() == 1
    tiles = {t: np.full((512, 128), t, np.float32) for t in range(16)}
    im = agipd.assemble(tiles)
    r0, c0 = o[15]
    assert im[r0 + 1, c0 + 1] == 15 and np.isnan(im[r0, c0 + 64:c0 + 66]).all()   # ASIC gap
    assert o[15][0] < o[12][0] and o[15][1] > o[3][1] and o[15][0] > o[11][0]     # top tile, bottom right
    one = np.zeros((512, 128), np.float32)
    one[158, 20] = 1                                 # Bragg pixel 158 of tile 15: 3rd ASIC from the left
    r, c = np.argwhere(agipd.assemble({15: one}) == 1)[0]
    assert (r - r0, c - c0) == (20, 158 + 2 * 2)
    per = {t: np.arange(3 * 512 * 128, dtype=np.float32).reshape(3, 512, 128) + t for t in (15, 8)}
    st = agipd.stack(per, [15, 8])
    assert st.shape == (3,) + agipd.stack_shape([8, 15]) == (3, 2 * 128 + agipd.STACK_GAP, 512)
    assert st[1, 0, 7] == per[15][1, 7, 0]          # tile drawn transposed, in the order given (15 on top)
    assert np.isnan(st[0, 130]).all()
    assert agipd.parse_tiles("15, 8 3-4 99 8") == [15, 8, 3, 4]      # order kept: top to bottom


def test_roi_on_tile_gap_counts_only_real_pixels():
    img = np.ones((2, 20, 20), np.float32)
    img[:, 10:] = np.nan
    s, npx = an.roi_stats(img, (10, 10, 30, 30, 0, "rectangle"))
    assert npx == 200 and np.allclose(s, 200)


def test_simulated_7016():
    b = SimulatedBackend(config_for("7016"), n_shots=12, delays=[0, 200])
    d = b.load_scan(101)
    assert set(d.overview) == set(range(16)) and d.overview[15].shape == (512, 128)
    assert d.images["agipd_sub"].shape == (12,) + agipd.stack_shape([8, 10, 15])
    assert set(d.spectra) == {"hirex", "des"} and d.xgm.shape == (12,)
    assert np.allclose(d.pump, an.spectrum_integral(d.spectra["hirex"], 50)[0])


# ---------------------------------------------------------------- fake extra_data
class _KD:
    def __init__(self, tids, data):
        self.tids, self.data = np.asarray(tids), np.asarray(data)

    def train_id_coordinates(self):
        return self.tids

    def ndarray(self, roi=None):
        return self.data if roi is None else np.stack([e[roi] for e in self.data])

    def select_trains(self, sel):
        keep = np.asarray(sel.ids) if isinstance(sel, _ById) else np.unique(self.tids)[sel]
        m = np.isin(self.tids, keep)
        return _KD(self.tids[m], self.data[m])

    def split_trains(self, trains_per_part=32):
        u = np.unique(self.tids)
        for i in range(0, len(u), trains_per_part):
            m = np.isin(self.tids, u[i:i + trains_per_part])
            yield _KD(self.tids[m], self.data[m])


class _Run:
    def __init__(self, keys):
        self.keys = keys

    def __getitem__(self, sk):
        if sk not in self.keys:
            raise KeyError(sk)
        return self.keys[sk]

    def select_trains(self, sel):
        return self


class _ByIndex:
    def __getitem__(self, s):
        return s


class _ById:
    def __init__(self, ids=None):
        self.ids = ids

    def __getitem__(self, ids):
        return _ById(ids)


def _fake_run(n=6):
    rng = np.random.default_rng(0)
    tids = np.arange(1000, 1000 + n)
    keys = {}
    for t in range(16):                              # 3 AGIPD frames per train
        frames = np.repeat(tids, 3)
        data = np.zeros((3 * n, 512, 128), np.float32)
        data[:] = t + np.tile([0.0, 1.0, 2.0], n)[:, None, None]   # mean over frames = t + 1
        keys[(f"MID_DET_AGIPD1M-1/CORR/{t}CH0:output", "image.data")] = _KD(frames, data)
    hirex = rng.normal(0, 0.01, (n, 30, 2400)).astype(np.float32)
    hirex[:, 20, 2000] = 100.0                       # line only in frame 20
    hirex[:, 5, 2000] = 1e6                          # another frame: must be ignored
    des = rng.normal(0, 0.01, (n, 4, 1280)).astype(np.float32)
    des[:, 1, 1180] = 40.0
    keys[("SA2_XTD6_HIREX/CORR/GOTTHARD_RECEIVER:daqOutput", "data.adc")] = _KD(tids, hirex)
    keys[("MID_EXP_DES/DET/GOTTHARD_RECEIVER:daqOutput", "data.adc")] = _KD(tids, des)
    xgm = np.zeros((n, 1000))
    xgm[:, 0] = np.arange(n) + 500
    keys[("SA2_XTD1_XGM/XGM/DOOCS:output", "data.intensityTD")] = _KD(tids, xgm)
    return _Run(keys)


def test_extra_data_reader_7016(monkeypatch):
    fake = types.SimpleNamespace(open_run=lambda prop, run, data="proc": _fake_run(), by_index=_ByIndex(), by_id=_ById())
    monkeypatch.setitem(sys.modules, "extra_data", fake)
    cfg = config_for("7016")
    d = ExtraDataBackend(cfg).load_scan(42)
    assert d.n == 6 and np.isnan(d.delay)
    assert np.allclose(d.overview[3], 4.0)            # 3 frames per train averaged
    st = d.images["agipd_sub"]
    assert st.shape == (6,) + agipd.stack_shape([8, 10, 15]) and np.allclose(st[:, 0, 0], 9.0)
    assert d.spectra["hirex"].shape == (6, 500) and d.spectra["des"].shape == (6, 150)
    assert np.allclose(d.pump, 100, atol=1) and np.allclose(d.probe, 40, atol=1)   # frame 20 / frame 1
    assert np.allclose(d.xgm, np.arange(6) + 500)


def test_per_train_mean():
    t, m = per_train_mean([5, 5, 6, 7, 7, 7], np.array([1.0, 3, 10, 1, 2, 3]))
    assert list(t) == [5, 6, 7] and list(m) == [2, 10, 2]


def test_preset_7016_prefers_Sources_7016_in_start_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = config_for("7016")
    assert cfg._source_file.endswith("presets/Sources_7016.yaml")
    cfg.detectors["agipd_sub"].tiles = [3]
    cfg.save(tmp_path / "Sources_7016.yaml")
    monkeypatch.setenv("XTG_MID_START_DIR", str(tmp_path))
    cfg = config_for("7016")
    assert cfg._source_file == str(tmp_path / "Sources_7016.yaml") and cfg.detectors["agipd_sub"].tiles == [3]


def test_delays_file_roundtrip(tmp_path):
    from xtg_mid_gui import delays
    p = tmp_path / "delays_7016.csv"
    assert delays.load(p) == {}
    delays.save(p, {96: 0.0, 92: -10.0, 94: 10.5})
    p.write_text(p.read_text() + "95 ; 20\nrun, delay\n# comment\n")
    assert delays.load(p) == {92: -10.0, 94: 10.5, 95: 20.0, 96: 0.0}


def test_parse_free_run_lists():
    from xtg_mid_gui.backends import parse_scans
    assert parse_scans("92, 96, 94, 95") == [92, 96, 94, 95]
    assert parse_scans("95 96 98-99, 96") == [95, 96, 98, 99]


def test_sources_file_found_from_any_folder(tmp_path, monkeypatch):
    """Den, 2026-11-26: 'Cannot find Sources_7016.yaml'.  The name alone must work from any
    folder (shipped preset), in any letter case, and a missing file must say where it looked."""
    import pytest
    from xtg_mid_gui.config import PRESET_DIR, config_for, preset_of_file
    from xtg_mid_gui.gui import main
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XTG_MID_START_DIR", str(tmp_path))
    cfg = config_for("custom", "Sources_7016.yaml")           # --config Sources_7016.yaml
    assert cfg.preset == "7016" and cfg._source_file == str(PRESET_DIR / "Sources_7016.yaml")
    assert "agipd" in cfg.detectors
    assert config_for("7016")._source_file == str(PRESET_DIR / "Sources_7016.yaml")
    assert config_for("11456")._source_file == str(PRESET_DIR / "Sources_11456.yaml")
    (tmp_path / "sources_7016.yaml").write_text((PRESET_DIR / "Sources_7016.yaml").read_text())
    assert config_for("7016")._source_file.endswith("sources_7016.yaml")   # start folder first, any case
    assert preset_of_file("/x/SOURCES_11456.yaml") == "11456" and preset_of_file("my.yaml") is None
    with pytest.raises(FileNotFoundError, match="cannot find nothere.yaml"):
        config_for("custom", "nothere.yaml")
    assert main(["--config", "nothere.yaml"]) == 2              # clear error, no traceback


def _masked_fake_run(n=6):
    """_fake_run plus a HIREX data.mask flagging the line pixel, and NaN dead pixels
    (run 286 of p7016: NaN pixels in the HIREX window made every pump integral NaN)."""
    run = _fake_run(n)
    hk = ("SA2_XTD6_HIREX/CORR/GOTTHARD_RECEIVER:daqOutput", "data.adc")
    hirex = run.keys[hk].data
    hirex[:, 20, 1795] = np.nan                      # in the edge-baseline region
    hirex[:, 20, 1839] = np.nan                      # inside the window
    mask = np.zeros(hirex.shape, np.uint32)
    mask[:, 20, 2000] = 4                            # the line pixel is "bad"
    run.keys[(hk[0], "data.mask")] = _KD(run.keys[hk].tids, mask)
    ak = ("MID_DET_AGIPD1M-1/CORR/8CH0:output", "image.data")
    am = np.zeros(run.keys[ak].data.shape, np.uint32)
    am[:, 0, 0] = 1
    run.keys[(ak[0], "image.mask")] = _KD(run.keys[ak].tids, am)
    return run


@pytest.mark.parametrize("apply", [True, False])
def test_extra_data_bad_pixel_masks(monkeypatch, apply):
    fake = types.SimpleNamespace(open_run=lambda prop, run, data="proc": _masked_fake_run(), by_index=_ByIndex(), by_id=_ById())
    monkeypatch.setitem(sys.modules, "extra_data", fake)
    cfg = config_for("7016")
    assert cfg.apply_mask and cfg.spectrometers["hirex"].mask_key == "data.mask"
    cfg.apply_mask = apply
    d = ExtraDataBackend(cfg).load_scan(42)
    assert np.isfinite(d.pump).all()                 # NaN dead pixels no longer poison the integral
    st = d.images["agipd_sub"]
    if apply:
        assert np.allclose(d.pump, 0, atol=1)         # masked line pixel left out
        assert np.isnan(st[:, 0, 0]).all() and "hirex" in d.masks
    else:
        assert np.allclose(d.pump, 100, atol=1)
        assert np.allclose(st[:, 0, 0], 9.0) and not d.masks


def test_mask_key_for():
    from xtg_mid_gui.config import mask_key_for
    assert mask_key_for("image.data", "auto") == "image.mask"
    assert mask_key_for("data.adc", "auto") == "data.mask"
    assert mask_key_for("data.adc", "") == "" and mask_key_for("data.adc", None) == ""
    assert mask_key_for("data.adc", "data.badpix") == "data.badpix"


def test_per_train_mean_ignores_nan():
    t, m = per_train_mean([5, 5, 6, 6], np.array([1.0, np.nan, np.nan, np.nan]))
    assert list(t) == [5, 6] and m[0] == 1.0 and np.isnan(m[1])


def _jf_fake_run(n=4, energy=9.0):
    tids = np.arange(2000, 2000 + n)
    data = np.zeros((n, 2, 4, 6), np.float32)        # 2 memory cells of a 4x6 "Jungfrau"
    data[:, 0, 0, 0], data[:, 1, 0, 0] = 9.2, 17.5   # 1 and 2 photons -> mean 1.5
    data[:, :, 1, 1] = -3.0                          # negative: NaN
    data[:, 0, 2, 2], data[:, 1, 2, 2] = 27.1, -1.0  # 3 photons, NaN -> mean 3
    mask = np.zeros(data.shape, np.uint32)
    mask[:, :, 3, 5] = 1
    data[:, :, 3, 5] = 90.0
    src = "MID_EXP_JF500K1/CORR/JNGFR01:daqOutput"
    keys = {(src, "data.adc"): _KD(np.repeat(tids, 1), data), (src, "data.mask"): _KD(tids, mask),
            ("MID_XTD1_UND/DOOCS/ENERGY", "actualPosition.value"): _KD(tids, np.full(n, energy)),
            ("PUMP", "v"): _KD(tids, np.ones(n)), ("PROBE", "v"): _KD(tids, np.ones(n))}
    return _Run(keys)


@pytest.mark.parametrize("fixed", [0.0, 4.5])
def test_extra_data_photonizes_jungfrau(monkeypatch, fixed):
    from xtg_mid_gui.config import Detector, Source
    fake = types.SimpleNamespace(open_run=lambda prop, run, data="proc": _jf_fake_run(), by_index=_ByIndex(), by_id=_ById())
    monkeypatch.setitem(sys.modules, "extra_data", fake)
    cfg = SourceConfig(proposal=11456, spec_enabled=False, crl_enabled=False,
                       detectors={"jf1": Detector("Jungfrau 1", Source("MID_EXP_JF500K1/CORR/JNGFR01:daqOutput", "data.adc"))},
                       pump=Source("PUMP", "v"), probe=Source("PROBE", "v"), delay=Source("", ""))
    cfg.photonize, cfg.photon_energy_keV = True, fixed
    d = ExtraDataBackend(cfg).load_scan(1)
    im = d.images["jf1"]
    if fixed:                                         # 4.5 keV: 9.2 -> 2, 17.5 -> 4
        assert d.photon_energy == 4.5 and np.allclose(im[:, 0, 0], 3.0)
    else:                                             # 9 keV from the undulator
        assert d.photon_energy == 9.0 and np.allclose(im[:, 0, 0], 1.5)
        assert np.allclose(im[:, 2, 2], 3.0)          # the negative cell does not count
    assert np.isnan(im[:, 1, 1]).all() and np.isnan(im[:, 3, 5]).all()   # negative / masked
    assert np.allclose(im[:, 0, 1], 0.0)
    assert any("photonized" in n for n in d.notes)
    assert any(n.startswith("WARNING") and "2 memory cells" in n for n in d.notes)   # 2 cells: flagged


def test_extra_data_photonize_off_keeps_keV(monkeypatch):
    from xtg_mid_gui.config import Detector, Source
    fake = types.SimpleNamespace(open_run=lambda prop, run, data="proc": _jf_fake_run(), by_index=_ByIndex(), by_id=_ById())
    monkeypatch.setitem(sys.modules, "extra_data", fake)
    cfg = SourceConfig(proposal=11456, spec_enabled=False, crl_enabled=False,
                       detectors={"jf1": Detector("Jungfrau 1", Source("MID_EXP_JF500K1/CORR/JNGFR01:daqOutput", "data.adc"))},
                       pump=Source("PUMP", "v"), probe=Source("PROBE", "v"), delay=Source("", ""))
    d = ExtraDataBackend(cfg).load_scan(1)
    assert np.isnan(d.photon_energy) and np.allclose(d.images["jf1"][:, 0, 0], (9.2 + 17.5) / 2)


def test_mask_read_once_for_the_experiment(monkeypatch):
    from xtg_mid_gui import maskstore
    run = _masked_fake_run()
    fake = types.SimpleNamespace(open_run=lambda prop, r, data="proc": run, by_index=_ByIndex(), by_id=_ById())
    monkeypatch.setitem(sys.modules, "extra_data", fake)
    cfg = config_for("7016")
    d1 = ExtraDataBackend(cfg).load_scan(42)
    assert maskstore.path_for(cfg.proposal).exists()
    for k in [k for k in run.keys if k[1].endswith(".mask")]:      # masks gone from the "data"
        del run.keys[k]
    d2 = ExtraDataBackend(cfg).load_scan(43)
    assert np.isnan(d2.images["agipd_sub"][:, 0, 0]).all() and np.allclose(d2.pump, d1.pump)
    assert any("masks_7016.npz" in str(v) for v in d2.masks.values())
    cfg.mask_per_train = True                                       # every frame: reads the (now missing) mask
    d3 = ExtraDataBackend(cfg).load_scan(44)
    assert np.allclose(d3.pump, 100, atol=1)
