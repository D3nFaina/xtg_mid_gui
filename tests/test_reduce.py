import numpy as np
from xtg_mid_gui import analysis as an
from xtg_mid_gui.reduce import ImageReducer, reduce_images


def _stack(n=50, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.normal(5, 1, (n, 40, 60)).astype(np.float32)
    a[:, 20, 30] += rng.uniform(50, 100, n)
    a[3, 5:8, 5:8] = np.nan
    return a


SIG = [(30.0, 20.0, 6.0, 6.0, 0.0, "ellipse"), (10.0, 10.0, 8.0, 5.0, 20.0, "rect")]
BKG = [(50.0, 30.0, 8.0, 6.0, 0.0, "ellipse")]


def test_reduced_signal_equals_full():
    a, t = _stack(), np.arange(100, 150)
    st = reduce_images(a, t, SIG, BKG, chunk=7)
    for g in SIG:
        assert np.allclose(st.signal(g, BKG), an.roi_signal(a, g, BKG), equal_nan=True)
        assert np.allclose(st.signal(g), an.roi_signal(a, g), equal_nan=True)
    assert np.allclose(st.mean, np.nanmean(a, axis=0), equal_nan=True)
    assert st.signal((1.0, 1.0, 2.0, 2.0, 0.0, "ellipse")) is None


def test_positions_and_missing_trains():
    a, t = _stack(), np.arange(100, 150)
    st = reduce_images(a, t, SIG, BKG)
    x, y = st.position(SIG[0], "max")
    assert np.allclose(x, 30.5) and np.allclose(y, 20.5)        # pixel centres
    r = ImageReducer(SIG, BKG)
    r.add(t[10:], a[10:])
    r.add(t[:5], a[:5])
    st2, _ = r.finish(t)
    s = st2.signal(SIG[0], BKG)
    assert np.isnan(s[5:10]).all() and np.isfinite(s[:5]).all() and np.isfinite(s[10:]).all()


def test_load_many_in_parallel():
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from _par_backend import ParBackend
    from xtg_mid_gui.backends import load_many
    from xtg_mid_gui.config import SourceConfig
    be = ParBackend(SourceConfig())
    rois = {"jf1": ([(40.0, 50.0, 6.0, 6.0, 0.0, "ellipse")], [])}
    out = dict(load_many(be, [(101, rois), (102, rois), (999, rois)], workers=3))
    assert isinstance(out[999], RuntimeError)
    assert out[101].stats["jf1"].signal(rois["jf1"][0][0]) is not None
    assert {out[101].notes[-1], out[102].notes[-1]} != {f"pid {os.getpid()}"}     # read in other processes
