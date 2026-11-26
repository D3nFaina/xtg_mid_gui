"""Analysis, config and simulated-backend tests (no Qt needed)."""
import numpy as np
import pytest

from xtg_mid_gui import analysis as an
from xtg_mid_gui.backends import SimulatedBackend, parse_scans, reduce_pulses, xtg_response
from xtg_mid_gui.config import Source, SourceConfig


def test_parse_scans():
    assert parse_scans("101-103, 110 102") == [101, 102, 103, 110]


def test_ellipse_mask_area_and_rotation():
    m = an.ellipse_mask((200, 200), 100, 100, 40, 20)
    assert abs(m.sum() - np.pi * 40 * 20) / (np.pi * 800) < 0.02
    r = an.ellipse_mask((200, 200), 100, 100, 40, 20, angle_deg=90)
    assert m[100, 135] and not r[100, 135] and r[135, 100]


def test_roi_signal_background_subtraction():
    imgs = np.full((5, 50, 60), 3.0)
    imgs[:, 20:30, 20:30] += np.arange(5)[:, None, None]
    g = (25, 25, 4, 4, 0)
    s, npx = an.roi_stats(imgs, g)
    sig = an.roi_signal(imgs, g, [(50, 8, 5, 5, 0)])
    assert np.allclose(sig, np.arange(5) * npx)
    assert np.allclose(s, sig + 3 * npx)


def test_ratio_of_means_and_delay_series():
    rng = np.random.default_rng(0)
    den = [rng.gamma(9, 1 / 9, 400) for _ in range(3)]
    num = [2.0 * d for d in den]
    t, y, e, n = an.delay_series([100, -50, 100], num, den)
    assert list(t) == [-50, 100] and list(n) == [400, 800]   # same delay -> merged
    assert np.allclose(y, 2.0) and np.all(e < 1e-9)


def test_clean_scan_drops_outliers_and_beam_off():
    pump = np.r_[1 + 0.05 * np.random.default_rng(1).standard_normal(100), 50.0, 0.01]
    probe = np.ones(102)
    keep = an.clean_scan(pump, probe, nmad=5, min_frac=0.1)
    assert keep[:100].all() and not keep[100] and not keep[101]


def test_reduce_pulses():
    a = np.arange(2 * 4).reshape(2, 4).astype(float)
    assert np.allclose(reduce_pulses(a, Source("s", "k", pulses="1:3", reduce="sum")), [3, 11])
    img = np.ones((3, 2, 5, 6))
    assert reduce_pulses(img, Source("s", "k"), image_ndim=2).shape == (3, 5, 6)


def test_config_roundtrip(tmp_path):
    cfg = SourceConfig()
    cfg.pump = Source("MID_REAL/DIODE/PUMP", "data.value", scale=2.0)
    cfg.detectors["jf1"].label = "JF signal"
    for ext in ("yaml", "json"):
        p = tmp_path / f"c.{ext}"
        cfg.save(p)
        back = SourceConfig.load(p)
        assert back.pump == cfg.pump and back.detectors["jf1"].label == "JF signal"
    assert "pump" not in back.placeholders() and "probe" in back.placeholders()
    with pytest.raises(KeyError):
        SourceConfig.from_dict({"nonsense": 1})


def test_simulated_backend_one_delay_per_scan():
    b = SimulatedBackend(n_shots=30)
    scans = b.scans()
    assert len(scans) == len(set(b.delays.values()))
    d = b.load_scan(scans[0])
    assert d.images["jf1"].shape == (30, 100, 160) and d.images["jf2"].shape == (30, 100, 160)
    assert d.pump.shape == d.probe.shape == (30,)
    assert set(d.crl) == set(SourceConfig.CRL_KEYS)
    # repeatable
    assert np.array_equal(b.load_scan(scans[0]).pump, d.pump)


def test_simulated_signal_follows_response():
    b = SimulatedBackend(n_shots=60, delays=[-300, 100])
    geom = (0.25 * 160, 0.5 * 100, 12, 12, 0)          # TG +1 on jf1
    bkg = [(80, 15, 30, 7, 0)]
    out = {}
    for s in b.scans():
        d = b.load_scan(s)
        sig = an.roi_signal(d.images["jf1"], geom, bkg)
        out[d.delay] = an.ratio_of_means(sig, d.pump ** 2 * d.probe)[0]
    assert out[100] > 5 * abs(out[-300])
    assert xtg_response(-300) < 0.01 < xtg_response(100)


def test_spectrum_integral_baseline_and_window():
    x = np.arange(200)
    spec = np.tile(5.0 + 10 * np.exp(-0.5 * ((x - 100) / 3) ** 2), (4, 1))
    integ, cen = an.spectrum_integral(spec, n_edge=20)
    assert np.allclose(integ, 10 * 3 * np.sqrt(2 * np.pi), rtol=1e-3)   # baseline 5 removed
    assert np.allclose(cen, 100, atol=0.01)
    iw, _ = an.spectrum_integral(spec, n_edge=20, window=(97, 104))
    assert np.all(iw < integ) and np.all(iw > 0.7 * integ)


def test_simulated_self_seeded_spectra():
    b = SimulatedBackend(n_shots=200, delays=[0])
    d = b.load_scan(b.scans()[0])
    assert set(d.spectra) == {"spec1"} and d.spectra["spec1"].shape == (200, 400)   # one, before the SDL
    full, _ = an.spectrum_integral(d.spectra["spec1"])
    seed, _ = an.spectrum_integral(d.spectra["spec1"], window=(190, 212))
    # diodes see the seed line: correlate better with the seed window than with the full integral
    assert an.pearson(seed, d.pump) > an.pearson(full, d.pump) + 0.2


def test_config_spectrometers_roundtrip(tmp_path):
    cfg = SourceConfig()
    cfg.spectrometers["spec1"].px = (100, 900)
    cfg.spectrometers["spec1"].eV0 = 9000.0
    cfg.spectrometers["spec1"].dE = 0.01
    cfg.save(tmp_path / "c.yaml")
    back = SourceConfig.load(tmp_path / "c.yaml")
    assert back.spectrometers["spec1"].px == (100, 900) and back.spectrometers["spec1"].eV0 == 9000.0
    assert "spec1 (Upstream spectrometer 1)" in back.placeholders()


def test_rect_roi_mask_and_signal():
    m = an.rect_mask((100, 100), 50, 50, 10, 5)
    assert m.sum() == 20 * 10
    r = an.rect_mask((100, 100), 50, 50, 10, 5, angle_deg=90)
    assert r.sum() == 200 and r[58, 50] and not m[58, 50]
    imgs = np.ones((3, 100, 100))
    s, npx = an.roi_stats(imgs, (50, 50, 10, 5, 0, "rectangle"))
    se, npe = an.roi_stats(imgs, (50, 50, 10, 5, 0, "ellipse"))
    assert npx == 200 and np.allclose(s, 200) and npe < npx
