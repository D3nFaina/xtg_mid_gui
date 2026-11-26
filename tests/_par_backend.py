"""A picklable stand-in for the real-data backend, for the parallel-loading test."""
import os

from xtg_mid_gui.backends import ExtraDataBackend, SimulatedBackend


class ParBackend(ExtraDataBackend):
    def load_scan(self, scan, rois=None):
        if scan == 999:
            raise RuntimeError("no such run")
        d = SimulatedBackend(self.cfg, n_shots=10).load_scan(scan, rois=rois)
        d.notes.append(f"pid {os.getpid()}")
        return d
