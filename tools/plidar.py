"""Python access to the C library, through ctypes.

The sweeps and the figures are written in Python because that is where
numpy and matplotlib are, but none of the physics is reimplemented here.
Every number this module produces comes out of the same object file the
tests run against, which is the point: a figure in the README cannot drift
away from the library by being generated from a second implementation.

Build the shared object first::

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --parallel
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import POINTER, byref, c_double, c_int, c_size_t, c_uint32, c_uint64
from pathlib import Path

C_LIGHT = 299792458.0

IRF_GAUSSIAN = 0
IRF_EMG = 1

SIM_IDEAL = 0
SIM_FIRST_PHOTON = 1

_STATUS = {
    0: "ok",
    1: "invalid argument",
    2: "value outside the model domain",
    3: "every cycle detected, inverse undefined",
    4: "iteration did not converge",
}


class PlidarError(RuntimeError):
    def __init__(self, code: int, where: str):
        super().__init__(f"{where}: {_STATUS.get(code, code)}")
        self.code = code


def _check(code: int, where: str) -> None:
    if code != 0:
        raise PlidarError(code, where)


class Irf(ctypes.Structure):
    _fields_ = [("kind", c_int), ("sigma", c_double), ("tau", c_double)]


class Scene(ctypes.Structure):
    _fields_ = [
        ("period", c_double),
        ("bin_width", c_double),
        ("nbins", c_size_t),
        ("t0", c_double),
        ("signal", c_double),
        ("background", c_double),
        ("efficiency", c_double),
        ("irf", Irf),
    ]


class Detector(ctypes.Structure):
    _fields_ = [
        ("dead_time", c_double),
        ("paralyzable", c_int),
        ("first_photon_only", c_int),
        ("afterpulse_prob", c_double),
        ("afterpulse_tau", c_double),
    ]


class Estimate(ctypes.Structure):
    _fields_ = [
        ("t0", c_double),
        ("range", c_double),
        ("signal", c_double),
        ("background", c_double),
        ("loglik", c_double),
        ("iterations", c_uint32),
    ]


class CrbResult(ctypes.Structure):
    _fields_ = [
        ("var_tof_known", c_double),
        ("var_tof_joint", c_double),
        ("information", c_double),
    ]


class Rng(ctypes.Structure):
    _fields_ = [("state", c_uint64), ("inc", c_uint64)]


def _library_name() -> str:
    if sys.platform == "darwin":
        return "libplidar.dylib"
    if sys.platform == "win32":
        return "plidar.dll"
    return "libplidar.so"


def _find_library(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("PLIDAR_LIB")
    if env:
        return Path(env)

    root = Path(__file__).resolve().parent.parent
    name = _library_name()
    for candidate in sorted(root.glob(f"build*/**/{name}")):
        return candidate
    raise FileNotFoundError(
        f"{name} not found under {root}. Configure and build first, or set "
        "PLIDAR_LIB to the shared object."
    )


_D = POINTER(c_double)
_U64 = POINTER(c_uint64)


class Plidar:
    """Thin wrapper holding the loaded shared object and its signatures."""

    def __init__(self, path: str | None = None):
        self.path = _find_library(path)
        self.lib = ctypes.CDLL(str(self.path))
        f = self.lib

        f.plidar_rng_seed.argtypes = [POINTER(Rng), c_uint64, c_uint64]
        f.plidar_rng_seed.restype = None

        f.plidar_scene_check.argtypes = [POINTER(Scene)]
        f.plidar_scene_check.restype = c_int

        f.plidar_scene_rate.argtypes = [POINTER(Scene), c_double, _D, c_size_t]
        f.plidar_scene_rate.restype = c_int

        f.plidar_sim_analytic.argtypes = [
            POINTER(Scene), c_int, c_double, _D, c_size_t
        ]
        f.plidar_sim_analytic.restype = c_int

        f.plidar_sim_mc.argtypes = [
            POINTER(Scene), POINTER(Detector), c_uint64, POINTER(Rng), _U64,
            c_size_t, _U64,
        ]
        f.plidar_sim_mc.restype = c_int

        f.plidar_coates_invert.argtypes = [_U64, c_uint64, _D, c_size_t]
        f.plidar_coates_invert.restype = c_int

        f.plidar_est_centroid.argtypes = [
            POINTER(Scene), _D, c_size_t, c_double, POINTER(Estimate)
        ]
        f.plidar_est_centroid.restype = c_int

        f.plidar_est_matched.argtypes = [
            POINTER(Scene), _D, c_size_t, _D, POINTER(Estimate)
        ]
        f.plidar_est_matched.restype = c_int

        f.plidar_est_mle.argtypes = [
            POINTER(Scene), _D, c_double, c_size_t, _D, POINTER(Estimate)
        ]
        f.plidar_est_mle.restype = c_int

        f.plidar_crb.argtypes = [
            POINTER(Scene), c_double, _D, c_size_t, POINTER(CrbResult)
        ]
        f.plidar_crb.restype = c_int

        f.plidar_tof_to_range.argtypes = [c_double]
        f.plidar_tof_to_range.restype = c_double

    # -- construction helpers -------------------------------------------

    @staticmethod
    def gaussian(fwhm_s: float) -> Irf:
        return Irf(IRF_GAUSSIAN, fwhm_s / 2.35482004503094938202, 0.0)

    @staticmethod
    def emg(sigma_s: float, tau_s: float) -> Irf:
        return Irf(IRF_EMG, sigma_s, tau_s)

    @staticmethod
    def scene(
        *,
        period: float,
        bin_width: float,
        t0: float,
        signal: float,
        background: float,
        efficiency: float,
        irf: Irf,
    ) -> Scene:
        nbins = int(round(period / bin_width))
        return Scene(period, bin_width, nbins, t0, signal, background,
                     efficiency, irf)

    @staticmethod
    def detector(
        *,
        dead_time: float = 0.0,
        paralyzable: bool = False,
        first_photon_only: bool = True,
        afterpulse_prob: float = 0.0,
        afterpulse_tau: float = 0.0,
    ) -> Detector:
        return Detector(dead_time, int(paralyzable), int(first_photon_only),
                        afterpulse_prob, afterpulse_tau)

    # -- the library ----------------------------------------------------

    def seed(self, seed: int, stream: int = 0) -> Rng:
        rng = Rng()
        self.lib.plidar_rng_seed(byref(rng), c_uint64(seed), c_uint64(stream))
        return rng

    def check(self, scene: Scene) -> None:
        _check(self.lib.plidar_scene_check(byref(scene)), "scene_check")

    def rate(self, scene: Scene, t0: float):
        import numpy as np

        out = np.zeros(scene.nbins, dtype=np.float64)
        _check(
            self.lib.plidar_scene_rate(
                byref(scene), c_double(t0),
                out.ctypes.data_as(_D), c_size_t(scene.nbins)),
            "scene_rate")
        return out

    def analytic(self, scene: Scene, cycles: float, mode: int = SIM_IDEAL):
        import numpy as np

        out = np.zeros(scene.nbins, dtype=np.float64)
        _check(
            self.lib.plidar_sim_analytic(
                byref(scene), c_int(mode), c_double(cycles),
                out.ctypes.data_as(_D), c_size_t(scene.nbins)),
            "sim_analytic")
        return out

    def simulate(self, scene: Scene, detector: Detector, cycles: int, rng: Rng):
        import numpy as np

        hist = np.zeros(scene.nbins, dtype=np.uint64)
        recorded = c_uint64(0)
        _check(
            self.lib.plidar_sim_mc(
                byref(scene), byref(detector), c_uint64(cycles), byref(rng),
                hist.ctypes.data_as(_U64), c_size_t(scene.nbins),
                byref(recorded)),
            "sim_mc")
        return hist, recorded.value

    def coates(self, hist, cycles: int):
        import numpy as np

        out = np.zeros(hist.size, dtype=np.float64)
        _check(
            self.lib.plidar_coates_invert(
                hist.ctypes.data_as(_U64), c_uint64(cycles),
                out.ctypes.data_as(_D), c_size_t(hist.size)),
            "coates_invert")
        return out

    def centroid(self, scene: Scene, counts, half_window: float) -> Estimate:
        est = Estimate()
        _check(
            self.lib.plidar_est_centroid(
                byref(scene), counts.ctypes.data_as(_D), c_size_t(counts.size),
                c_double(half_window), byref(est)),
            "est_centroid")
        return est

    def matched(self, scene: Scene, counts, scratch) -> Estimate:
        est = Estimate()
        _check(
            self.lib.plidar_est_matched(
                byref(scene), counts.ctypes.data_as(_D), c_size_t(counts.size),
                scratch.ctypes.data_as(_D), byref(est)),
            "est_matched")
        return est

    def mle(self, scene: Scene, counts, cycles: float, scratch) -> Estimate:
        est = Estimate()
        _check(
            self.lib.plidar_est_mle(
                byref(scene), counts.ctypes.data_as(_D), c_double(cycles),
                c_size_t(counts.size), scratch.ctypes.data_as(_D), byref(est)),
            "est_mle")
        return est

    def crb(self, scene: Scene, cycles: float, scratch) -> CrbResult:
        out = CrbResult()
        _check(
            self.lib.plidar_crb(
                byref(scene), c_double(cycles), scratch.ctypes.data_as(_D),
                c_size_t(scene.nbins), byref(out)),
            "crb")
        return out


def tof_to_range(tof_s: float) -> float:
    return 0.5 * C_LIGHT * tof_s


def range_to_tof(range_m: float) -> float:
    return 2.0 * range_m / C_LIGHT
