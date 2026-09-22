# photon-counting-lidar

[![ci](https://github.com/andrealo20/photon-counting-lidar/actions/workflows/ci.yml/badge.svg)](https://github.com/andrealo20/photon-counting-lidar/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![language: C99](https://img.shields.io/badge/language-C99-00599C.svg)](https://en.wikipedia.org/wiki/C99)
[![tests: 46 passing](https://img.shields.io/badge/tests-46%20passing-brightgreen.svg)](tests/)
[![sanitizers: ASan + UBSan](https://img.shields.io/badge/sanitizers-ASan%20%2B%20UBSan-brightgreen.svg)](.github/workflows/ci.yml)
[![no malloc](https://img.shields.io/badge/allocation-none-informational.svg)](#memory)

**Depth estimation from single photon arrival times, with the detector modelled
as it actually behaves and every estimator measured against the Cramer Rao
bound rather than against itself.**

A lidar that counts single photons never sees a waveform. For each laser pulse
it gets, at best, one timestamp, and usually nothing at all. The distance
exists only as a statistic accumulated over millions of pulses, which makes the
question interesting: given a histogram of arrival times, how well can the
distance be read out, and how close does a given estimator get to that limit?

![depth error against photon budget](docs/images/photons.png)

With a 152 ps instrument response, worth 23 mm of range on its own, the maximum
likelihood estimate reaches 0.17 mm from ten thousand photons and sits on the
bound over three decades. Below about ten photons it stops doing that, for a
reason worth understanding rather than hiding.

## The model

Within each repetition period the detected photons form an inhomogeneous
Poisson process of rate

```
lambda(t) = eta * ( S * irf(t - t0) + B )
```

with t0 = 2d/c the time of flight, S the signal photons a cycle delivers, B the
rate of ambient light and dark counts, and eta the detection efficiency. Counts
in bin i over M cycles are Poisson with mean M*mu_i. Four things about a real
instrument are in the forward model and change the answer:

**The response is not symmetric.** A single photon avalanche diode has a
Gaussian core from avalanche and electronics jitter, plus an exponential tail
from carriers that reach the multiplication region by diffusion. The pair is an
exponentially modified Gaussian. This is why the centroid curve above flattens
at 19.5 mm and stays there no matter how many photons arrive: that is a bias,
not noise, and it is close to the 21 mm the tail constant is worth.

**One record per cycle.** Classic timing electronics keep the first photon and
discard the rest, so early photons mask late ones and the histogram leans
early. The distance comes out short. The effect has an exact inverse, published
by Coates in 1968, and the library implements it.

**Dead time.** After a detection the diode is blind for tens of nanoseconds,
which is comparable to the repetition period, so the blindness carries across
cycle boundaries. Non paralyzable and paralyzable quenching are both modelled.

**Afterpulsing.** Trapped carriers released later produce spurious counts
correlated with an earlier detection.

## Verification

The library is checked against closed forms, against its own independent
reimplementation, and against a bound. Nothing in the results above rests on an
estimator being compared with itself.

**The simulator exists twice.** One path evaluates the response distribution
function and applies the first photon expression in closed form. The other
draws a Poisson count of photons per cycle and places each one from the way the
exponentially modified Gaussian is constructed, a normal jitter plus a
diffusion delay, then runs them through the detector model. The two share the
response parameters and nothing else: the second never evaluates the
distribution function or the density. A Pearson statistic over about a thousand
bins, accumulated over two million cycles, must land within five standard
deviations of its degrees of freedom, and it does for both response shapes and
with pile up on or off.

**The pile up correction round trips exactly.** Distorting a known set of rates
and inverting the distortion returns them to twelve significant figures, at
detection rates up to a quarter per cycle where the distortion is severe.

**The bound is computed twice.** The Fisher information comes from the analytic
derivative of the binned response, and is checked against the curvature of the
expected log likelihood taken by finite difference, which uses none of those
derivatives. They agree to within one percent, which is the truncation error of
the difference.

**The estimators are measured against the bound, not against each other.** Four
hundred independent realisations per photon budget, with the realisations that
landed on the wrong peak counted separately rather than averaged in.

| signal photons | bound | maximum likelihood | matched filter | centroid |
|---|---|---|---|---|
| 10 | 5.30 mm | 5.98 mm (0.5% lost) | 16.58 mm (0.5% lost) | 27.77 mm (7.5% lost) |
| 100 | 1.68 mm | 1.60 mm | 1.99 mm | 19.62 mm |
| 1000 | 0.53 mm | 0.53 mm | 0.68 mm | 19.55 mm |
| 10000 | 0.17 mm | 0.17 mm | 0.21 mm | 19.53 mm |

Errors are root mean square over the realisations that found the return; "lost"
is the fraction that did not. At 100 photons the measured value is 4% under the
bound, which is within the sampling error of 400 realisations and not a claim
that the bound was beaten.

### Why the likelihood curve leaves the bound

Below about ten photons the estimate is sometimes metres out rather than
millimetres. The likelihood surface has grown a second maximum on a background
fluctuation and the search took it. This is the threshold effect familiar from
delay estimation, it is not an implementation defect, and no unbiased estimator
avoids it: the Cramer Rao bound describes local curvature and says nothing
about a competing peak elsewhere. The coarse stage of the search deliberately
scans the whole repetition period so that this can be observed instead of
being designed away.

### Pile up, as a distance

Exact expected histograms, no noise, so what is left is the systematic error
alone. Generated by `tools/sweep_pileup.py`.

| detections per cycle | uncorrected | after Coates |
|---|---|---|
| 4.4% | -0.063 mm | +0.14 um |
| 12.6% | -0.189 mm | +0.14 um |
| 25.9% | -0.610 mm | +0.14 um |
| 41.7% | -1.197 mm | +0.14 um |
| 59.3% | -2.307 mm | +0.14 um |
| 80.8% | -4.332 mm | +0.14 um |

The residual after correction is the tolerance of the search for the peak, not
anything left of the physics. The customary advice to keep the detection rate
under a few percent per pulse is visible in the first row: it buys a bias below
a tenth of a millimetre.

## Cost

Release build, one core.

| step | work | time | rate |
|---|---|---|---|
| simulate | 2 000 000 cycles | 29.7 ms | 67.3 Mcycle/s |
| correct pile up | 2000 bins | 6.2 us | 322 Mbin/s |
| matched filter | 2000 bins | 35 us | 28 900 estimates/s |
| maximum likelihood | 2000 bins, 18 amplitude updates | 202 us | 4950 estimates/s |
| Fisher information | 2000 bins | 60 us | 16 700 evaluations/s |

Regenerate with `./build/bench/plidar_bench`.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The sweeps and figures are Python, and they drive the same shared object the
tests are built from rather than reimplementing anything:

```bash
python -m pip install numpy matplotlib
cd tools && python sweep_photons.py --trials 400 && python make_figures.py
```

## Layout

```
include/plidar/   public headers, one per concern
src/              the library, C99, no allocation
tests/            46 tests over six modules
tools/            ctypes binding, sweeps, figures
bench/            the cost report above
docs/design.md    the derivations and the choices behind them
docs/data/        the csv files the tables and figures come from
```

## Memory

Nothing in the library allocates. Buffers are caller owned, and the routines
that need working space take it as an argument and document how much. The
Monte Carlo path holds one cycle of arrivals on the stack and reports a domain
error rather than growing if a scene is bright enough to exceed it.

## Limitations

Worth stating plainly, since none of them are hidden in the code:

- Everything here is simulated. The forward model is built from the physics and
  cross checked against itself, but it has not been compared with a recorded
  measurement. Validating it against a public time correlated single photon
  counting trace, with a documented response and known parameters, is the
  obvious next step.
- One surface per pixel. Fog, foliage or a window in front of the target put
  two returns in the same histogram, which turns the problem into deciding how
  many surfaces there are before estimating where they sit.
- Returns beyond the unambiguous range are dropped rather than folded back.
  Both simulator paths drop them identically, so the parity argument is
  unaffected, but a scene with the target near the end of the period is not
  meaningful.
- The estimators assume the response is known. A response fitted from a
  calibration measurement carries its own uncertainty, which is not propagated.

## Licence

MIT, see [LICENSE](LICENSE).
