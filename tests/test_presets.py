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
    assert st[1, 0, 7] == per[8][1, 7, 0]           # tile drawn transposed, ascending tile number
    assert np.isnan(st[0, 130]).all()
    assert agipd.parse_tiles("15, 8 3-4 99") == [3, 4, 8, 15]


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
        keep = np.unique(self.tids)[sel]
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
    fake = types.SimpleNamespace(open_run=lambda prop, run, data="proc": _fake_run(), by_index=_ByIndex())
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
