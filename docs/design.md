# Design notes

Why the code is shaped the way it is, and where the formulas in it come from.
The README says what the library does; this says what had to be decided to make
it do that.

## 1. What is being modelled

One pixel of a direct time of flight lidar that counts single photons. A laser
fires at repetition rate f_rep. Light travels to a surface at distance d and
back, so it arrives delayed by t0 = 2d/c. A single photon avalanche diode
detects individual photons and a time to digital converter timestamps them
relative to the laser sync. Over M cycles the timestamps accumulate into a
histogram of N_bins bins of width delta.

The detected photons within a cycle are an inhomogeneous Poisson process of
rate

    lambda(t) = eta * ( S * h(t - t0) + B )

where h is the normalised instrument response, S the mean signal photons a
cycle would deliver to a perfect detector, B the flat rate of ambient light
plus dark counts, and eta the detection efficiency. Writing

    mu_i = eta * ( S * g_i(t0) + B * delta ),   g_i(t0) = H(hi_i - t0) - H(lo_i - t0)

for the expected detected photons per cycle in bin i, with H the distribution
function of the response, the counts are Poisson with mean M*mu_i. Everything
in the library is expressed in terms of mu_i and g_i.

Bin masses are taken as differences of the distribution function rather than
the density at the bin centre. The result is then exact at any bin width and
the masses of a grid covering the line sum to one, which the midpoint rule does
not give.

## 2. The instrument response

A Gaussian is the usual idealisation and it is wrong in a way that matters.
Carriers generated outside the multiplication region of the diode reach it by
diffusion and arrive late, which puts an exponential tail on the right of an
otherwise Gaussian core. The convolution of a normal of width sigma with an
exponential of mean tau is the exponentially modified Gaussian.

Writing a = t/sigma and v = sigma/tau, and Phi for the standard normal
distribution function, the density and the distribution function are

    f(t) = (1/tau) * exp( v^2/2 - t/tau ) * Phi(a - v)
    F(t) = Phi(a) - tau * f(t)

The second line is an identity, not a shortcut. Differentiating
F(t) = Phi(a) - exp(v^2/2 - t/tau) * Phi(a - v) gives three terms, and two of
them cancel exactly, because

    phi(a - v) = phi(a) * exp(a*v - v^2/2)   and   a*v = t/tau

Using it means one expression is evaluated instead of two, and it gives the
tests something real to check: differentiating F numerically has to reproduce
f, which is a statement about the algebra above rather than a comparison of two
copies of the same code.

The mean is tau and the variance is sigma^2 + tau^2. Both are used as test
targets against moments computed from the binned shape.

### Numerics

The factor exp(v^2/2 - t/tau) diverges as t goes to minus infinity while
Phi(a - v) vanishes, and their product is bounded by one. Evaluating them
separately overflows for perfectly ordinary parameters. The library assembles
the whole thing in the exponent:

    tau * f(t) = exp( v^2/2 - t/tau + log Phi(a - v) )

which needs log Phi, and log Phi needs care of its own because Phi underflows
long before its logarithm does. Above x = -30 the logarithm of
0.5*erfc(-x/sqrt(2)) is accurate and erfc is nowhere near its own underflow.
Below it, the Mills ratio expansion

    log Phi(x) = -x^2/2 - log(sqrt(2 pi)) - log(-x)
                 + log( 1 - 1/x^2 + 3/x^4 - 15/x^6 + 105/x^8 - 945/x^10 )

is used. At the threshold the first dropped term is 10395/x^12, which is 2e-14
relative, and an asymptotic series is bounded by its first omitted term.

One more cancellation hides in the same expression and is worse than it looks.
For a tail time constant far below the jitter, v^2/2 appears once with a plus
sign and once inside log Phi with a minus sign, and the two should cancel
exactly. Adding them and subtracting again throws away every digit they share:
by v = 1e5 the fifth significant figure is gone, by v = 1e8 the answer is 40
percent high, and past v = 1e154 it is a not a number. No detector has such a
ratio, but a sweep that drives tau towards zero to watch the response become
Gaussian passes through all of them. Doing the cancellation on paper leaves

    v^2/2 - t/tau + log Phi(a - v) = -a^2/2 - log(sqrt(2 pi)) - log(v - a) + log(S)

using a*v = t/tau, with S the Mills series, and nothing large is left in it.

### Bin masses in the far tail

Taking a bin mass as F(hi) - F(lo) works while the two values are away from
one. In the right tail they are not: once the remaining mass drops below the
spacing of doubles near one, both ends round to the same number and the mass
comes out as exactly zero. The derivative of that same mass is a difference of
two densities, which keeps its precision for another fifteen orders of
magnitude.

A bin with zero mass and a nonzero derivative is not a rounding nuisance. The
Fisher information divides the square of the derivative by the mass, so with a
small background in the denominator instead, those bins invent information that
is not there, and the bound becomes discontinuous as the background goes to
zero.

So the mass is taken from the distribution function left of the centre and from
the survival function right of it, with the one bin that straddles the centre
paying for the changeover. Subtracting two small numbers keeps their relative
precision; subtracting two numbers near one does not. The survival function has
no cancellation of its own: for the Gaussian it is erfc read from the other
side, and for the EMG it is Phi(-a) plus the tail term, both positive.

## 3. Pile up and its inverse

Classic time correlated single photon counting electronics record the first
photon of a cycle and ignore any that follow. The probability that the recorded
photon of a cycle falls in bin i is then

    P_i = exp( - sum_{j<i} mu_j ) * ( 1 - exp( -mu_i ) )

which is the probability that nothing arrived before bin i, times the
probability that at least one arrived in it. Exact, not a small signal
expansion. Early bins shadow late ones, the histogram leans early, and a
distance read off it comes out short.

Inverting is a matter of reading that relation backwards. With R_i the number
of cycles still undetected when bin i begins,

    R_0 = M,   R_{i+1} = R_i - N_i,   mu_i = log( R_i / R_{i+1} )

This is Coates' estimator, and it is the maximum likelihood estimate of mu_i
under the first photon rule rather than a fitted correction. The library
computes the logarithm as log1p(N_i / R_{i+1}) because the ratio is within
about 1e-6 of one in the regime these instruments run at.

It fails in exactly one way, and the failure is real: if every cycle has
produced a detection by some bin, nothing was ever observed past it and no
finite rate is consistent with that bin. The function reports it instead of
returning an infinity.

## 4. Two simulators, on purpose

A forward model that only exists once is a model nobody has checked. There are
two here and they are built from different facts about the same distribution.

The analytic path evaluates g_i from the distribution function and applies the
expression for P_i above. The Monte Carlo path never touches either the
distribution function or the density. It uses two properties of Poisson
processes instead:

- **Superposition.** A process of rate a*h(t) + b is the union of two
  independent processes, one of rate b and one of rate a*h(t). The background
  is drawn as a Poisson count with times uniform over the cycle.
- **Marking.** Thinning a Poisson process leaves a Poisson process. The signal
  photon count is Poisson with mean eta*S over the whole line, and each time is
  drawn as t0 plus a normal variate times sigma plus an exponential variate of
  mean tau, which is the construction the exponentially modified Gaussian is
  defined by. Photons landing outside the cycle are dropped with no correction
  needed.

The merged stream then goes through the detector: a blind interval after each
detection, tracked in absolute time so it carries across cycle boundaries,
optionally paralyzable, optionally with an afterpulse drawn from its own
generator stream so that turning afterpulsing on does not change which photons
arrive.

An error in the closed forms of section 2 or in the expression of section 3
makes the two disagree. The parity test is a Pearson statistic over the bins
with at least five expected counts, gated at five standard deviations of its
degrees of freedom. Seeds are fixed, so the test passes every time or fails
every time. A statistical test that is flaky in continuous integration is worse
than no test at all.

The gate is only worth what it can catch, and that depends on the scene rather
than on the number of cycles. At six percent of cycles recording, deleting the
pile up term from the analytic model outright moves the statistic by 4.3
standard deviations, which passes. So one case is run at 36 percent, where the
same deliberate error moves it by 987, and the test asserts both the agreement
with the right model and the disagreement with the wrong one. Without the
second half, the first is a comparison whose power is unknown.

Thinning against a constant envelope was the first approach and was discarded.
It works, but it evaluates the density pointwise, which is exactly the code the
parity test is supposed to be checking, and with a narrow response in a long
period it rejects around fifty candidates per cycle.

## 5. Estimators

Three, differing in how much of the model they use. A fourth, which fits the
first photon model to the counts as recorded, came with the recorded data and
is in section 8.

**Centroid.** Fullest bin, window around it, background taken from everything
outside the window, weighted mean of the bin centres. Biased late by close to
tau, because a mean does not know the shape is asymmetric. Left uncorrected on
purpose: subtracting the known mean of the response would remove most of that
bias, and a system that does not model the tail does not know the mean either.

**Matched filter.** Cross correlation with the binned response, peak located to
sub bin resolution by fitting a parabola to the three correlation values around
it. Uses the shape, so the tail no longer biases it. Optimal when the noise is
additive and Gaussian, which this noise is not, and the figure in the README
shows the resulting gap of about a quarter in standard deviation.

**Maximum likelihood.** With counts Poisson of mean A*g_i(t0) + B, maximise

    L(t0, A, B) = sum_i [ N_i log(A g_i + B) - (A g_i + B) ]

over all three. The amplitudes are moved by expectation maximisation, which for
this model has a two line update. Splitting each bin's counts into a signal
part and a background part, with the expected split

    w_i = A g_i / (A g_i + B)

the maximisation step is

    A <- sum_i N_i w_i / sum_i g_i
    B <- ( N_total - sum_i N_i w_i ) / N_bins

Both steps increase the likelihood, so the pair converges monotonically. The
time of flight is moved by golden section at fixed amplitudes, and the two
alternate. Coordinate ascent rather than a joint Newton step: the objective is
cheap, the bracket is small, and there is no Hessian to keep positive definite.

### The search, in three stages

1. A matched filter, used only to place the window the first amplitude guess
   is read off.
2. A scan of the whole repetition period with the statistic the likelihood
   reduces to at fixed amplitudes, which is a correlation with
   log(1 + (A/B) g). The logarithms depend only on the template, so they are
   taken once, which is what makes a full scan affordable.
3. Coordinate ascent from the winner.

Stage two covers every candidate whose window fits inside the record, rather
than a neighbourhood of stage one, and that is deliberate. A local search would
be faster and would hide the behaviour this repository exists to measure: below
a certain photon budget a background fluctuation carries a higher likelihood
than the return does, and the estimate lands metres away. An estimator that is
not allowed to make that mistake cannot be shown making it.

It is not the whole period, and the README says so as well. Candidates run from
bin h to bin nbins-h-1, so a return within a template half width of either end
is unreachable: 1.575 ns at each end here, about half a metre of the fifteen
the period allows. That is a property of a finite record rather than of the
search, but it is a limit and not a rounding.

### Cost

The response has finite effective support, so the template is truncated at ten
standard deviations either side, and every per candidate operation is linear in
the template width rather than in the number of bins. The full scan is then one
pass over the histogram per candidate offset, and a maximum likelihood estimate
over two thousand bins takes around two hundred microseconds.

Truncating the template puts a small amount of signal mass outside the window,
where the model treats it as background. For the response and bins used here
the half width is 31 bins, or 1.575 ns, and the mass beyond it is 1.4e-5 on the
tail side and 6e-154 on the other. Both are well under the statistical error at
any photon budget the bound allows.

## 6. The bound

For a Poisson model the Fisher information is

    I_jk = M * sum_i ( d mu_i / d theta_j ) ( d mu_i / d theta_k ) / mu_i

with theta = (t0, A, B) and

    d mu_i / d t0 = A * ( h(lo_i - t0) - h(hi_i - t0) )
    d mu_i / d A  = g_i
    d mu_i / d B  = 1

The derivative with respect to t0 is the difference of two density values by
the fundamental theorem of calculus, so nothing here is finite differenced and
the bound does not inherit a step size.

Two bounds are reported. The one usually quoted takes the amplitudes as known
and is 1/I_00. The one that describes what the estimator in this library
actually does takes all three as unknown and is the top left entry of the
inverse.

They turn out to be the same number to the last bit, and the reason is a
symmetry rather than a coincidence. The derivative of a symmetric response is
odd about the peak while the response itself is even, so the (t0, A) entry is a
sum of an odd quantity against an even weight and vanishes; the (t0, B) entry
is a sum of the derivative alone, which telescopes to nothing across a grid
that covers the response. An orthogonal nuisance parameter is free. The tailed
response breaks the first argument and the answer does not move, because the
second still holds over a record this long.

Both are still computed and reported, because the symmetry is a property of
these scenes and not of the formula.

### The zero background case

With B exactly zero the joint bound takes the background as known and inverts
the two by two block of t0 and A. This holds for `plidar_crb` and for the
first photon bound of section 8. Every bin the return does not reach has an
expected count of zero, so observing nothing there determines the background
exactly: its information is infinite, and a background known exactly leaves
two parameters.

The rule tests B itself, before any inversion, because the arithmetic cannot
be relied on to show the limit. A bin with a rate of zero is skipped so as not
to divide by it. With a Gaussian or an EMG the far tail leaves rates next to
those bins small enough that the information overflows and the three parameter
cofactor fails, or pins B so well that the cofactor agrees with the pair to
rounding. A table is exactly zero outside its support, so nothing overflows,
the skipped bins take their information with them, and the cofactor would
return a finite three parameter bound that is too pessimistic. With any
background the three parameter inverse is used, and the pair only where that
inverse does not come out finite and positive.

Because the derivative of the response integrates to zero, t0 and A are close
to orthogonal, and the answer lands near the bound with everything known,
which is what the background free limit should give.

## 7. What the tests actually establish

| test module | what it would catch |
|---|---|
| rng | a generator whose streams are not distinct, or whose variates have the wrong moments |
| irf | an error in the closed forms of section 2, an overflow in the tails, a derivative that does not match the function it claims to differentiate, a cancellation that eats the answer as the tail narrows |
| coates | an inverse that does not invert, a saturation reported as a number |
| parity | an error anywhere in either simulator, since they would have to be wrong in the same way to agree, with a control that measures how much the comparison could catch |
| estimate | an estimator that does not recover the parameters from its own noiseless forward model, and a first photon fit whose spread under pile up misses its bound |
| crb | an information matrix that does not match the curvature of the likelihood it came from, and a three by three inverse that does not survive being taken a second way |

The tolerances in the response tests are set by the finite differences in the
tests, not by the library. A central difference carries a truncation error of
order (step/scale)^2 times the ratio of the third derivative to the first, and
in the tails that ratio is an order of magnitude above one over the width
squared, which puts the achievable agreement around one part in ten thousand.
Shrinking the step trades that error for a cancellation error and does not
help.

For a tabulated response the parity test checks less than it does for the
EMG. Its Monte Carlo draws arrival times by rejection against the interpolated
density, so both paths start from the same density, and what they compare is
the integration of that density into bins and the pile up. The Monte Carlo of
the EMG builds each time from a normal and an exponential variate and never
evaluates the density, so an error in the density formula itself shows up as a
disagreement.

## 8. Real data

Up to here the model is checked against itself and against closed forms. This
section runs the library on a recorded sensor: sample 1 of the DepthEstimate
track of SP-TransientBench (Dong et al., 2026, arXiv:2606.18952), recorded by
an Adaps ADS6311. The data stay outside the repository. `tools/fetch_stb.py`
downloads one sample and verifies its hashes, `tools/real_stb.py` runs
everything below, and the files it derives sit in `docs/real/` under the
dataset's licence, CC BY-NC-SA 4.0.

### What the data hold

The sensor has 256 by 192 pixels. Each pixel is a histogram of 672 bins of
750 ps, a window of 504 ns, and sums 3 by 3 SPADs on the chip. Away from the
return the bins are empty: the median bin holds no counts, so there is next to
no background.

Pile up is extreme. More than half the pixels the precision is measured on
hold 6000 photons or more out of 7000 cycles. In such a pixel most cycles
record on the rising edge of the return, so the recorded shape changes with
flux. Averaged within each of the 5 flux classes and aligned on the rising
edge, the class shapes differ by 0.226 in mean L1 distance as recorded and
still by 0.152 after Coates (`docs/real/shapes.png`). That distance spans all
5 classes; the collapse curve of the cycle count below uses only the 4 under
6400 photons.

The response the dataset publishes, `config/IRF_global.csv`, is 1.5 ns wide
at half height and dips slightly negative. The returns here are about 4
times as wide, so the response is fitted to the histograms instead.
`config/offset.txt` gives an offset per pixel in metres, calibrated for the
dataset's own conversion of the fullest bin p to a range, (p - 17) c delta / 2
minus the offset. A Livox lidar recorded the same scene, 232548 points for this
frame.

### A response given as a table

Neither model of section 2 describes this pulse. It rises from 10 to 90
percent in about 3 bins, holds a flat top and keeps a long tail, and a sum of
parametric shapes fitted to it would mix its fitting error into everything
downstream. So the library also takes the response as a table,
`PLIDAR_IRF_TABLE`.

The table holds the density at nodes t_k = t_first + k dt. Between nodes the
density is linear and outside the first and last node it is zero. A piecewise
linear density has a piecewise quadratic distribution function in closed form,
and the trapezoidal rule gives its area exactly, so the samples are normalised
by it. Both end samples must be zero. A density that jumped at the end of its
support would put a kink in the bin masses, and `plidar_irf_bin_dt0`, which
takes the density at the two edges of a bin, would stop being the derivative
of the mass `plidar_irf_bin` returns.

The caller provides 3n doubles of working space for the density at the nodes,
the mass left of each node and the mass right of each node, the last summed
from the right end. Bins right of the centre are differenced on the survival
function, as for the two models in section 2, so the far tail stays a sum of
small terms and keeps its relative precision. The centre is t = 0, so the
origin of a table belongs inside the bulk of the response. For the recorded
pulse it is the half height point of the rising edge, and a distance read from
that origin needs a calibration offset, as the fullest bin does.

The peak bound the Monte Carlo needs is the largest node, exact for a
piecewise linear density. Mean and variance are integrated cell by cell in
closed form. The template of the estimators covers the whole support instead
of ten standard deviations.

### The first photon bound

Section 6 bounds a histogram of Poisson counts, which is what Coates gives
back in the mean. A recorded histogram is multinomial instead: each cycle puts
its one record in one bin or in none. With mu_i = a g_i(t0) + b photons per
cycle in bin i, a = eta S and b = eta B delta, and Lambda_i the rate summed
over the bins before i, a cycle records in bin i with probability

    P_i = exp( -Lambda_i ) * ( 1 - exp( -mu_i ) )

and records nothing with probability P_none = exp( -Lambda_n ).

The precision is read from groups of a pixel's counts (below), and a group is
the histogram thinned: each count survives with probability q, the keep. A
thinned histogram is multinomial too, with categories q P_i and one more,
1 - q (1 - P_none), for the cycles that left no kept record. Its Fisher
information for theta = (t0, a, b) is

    I = M * [ q * sum_i dP_i dP_i^T / P_i
              + q^2 * dP_none dP_none^T / (1 - q + q P_none) ]

    dP_i = exp( -Lambda_i ) * [ exp( -mu_i ) dmu_i - ( 1 - exp( -mu_i ) ) dLambda_i ]
    dP_none = -P_none dLambda_n

and running sums of Lambda and of its derivatives give it in one pass over
the bins. At q = 1 it is the ordinary multinomial. It is inverted as in
section 6, which also gives the rule for a background of exactly zero.

The tests check it against the Poisson bound at low flux, and check that the
timing information scales exactly as q while the support lies inside the
window, where the g_i sum to one at every t0. The curvature and second
inversion checks of the Poisson bound (section 7) run on it as well.

### The first photon fit

Coates then Poisson, the route of sections 3 and 5, recovers the mean rates
and then weighs every bin as if every cycle could still record in it. Once
most cycles record on the rising edge, the late bins are seen only by the few
cycles still listening. Coates divides by those few, so its noise grows, and
the Poisson likelihood gives the late bins more weight than they carry.
`plidar_est_mle_first_photon` fits the multinomial model to the counts as
recorded,

    L(t0, a, b) = sum_i n_i log( q P_i ) + n_none log( 1 - q (1 - P_none) )
    n_none = M - sum_i n_i

with M and q known and a, b >= 0. It needs no correction in front, weighs each
bin by the cycles that could still record in it, and with q < 1 takes one group
of a split as it is.

The search keeps the three stages of section 5. Its amplitudes have no closed
form update here. Expectation maximisation over the photons the first photon
rule hides would keep the Poisson form, but at these fluxes most photons are
hidden, and EM slows with the share that is missing. So the amplitudes move by
Newton steps on (a, b), with the gradient and Hessian from the same pass as L,
each step halved until L does not drop. A background of zero is a legitimate
answer: a step that would take b below zero stops at zero, and from there b
can stay while a moves alone. Everything in L is concave in (a, b) except the
empty category, which for q < 1 is convex in Lambda_n. Where that makes the
Hessian indefinite it is left out of the curvature, and the step still climbs.

Under pile up t0 and a are coupled, since a brighter return records earlier
and a later t0 makes up for it. In the test scene their squared correlation in
the bound is 0.42 at q = 1 and 0.67 at q = 1/4. Alternating one golden section
in t0 with one amplitude fit then closes the gap by a constant factor per
round, and on noiseless histograms it stopped short of the maximum. So the
golden section runs on the profile likelihood, with the amplitudes refitted at
every t0 it tries.

The tests check that on the noiseless expected histogram it returns the true
parameters, with no background and with a table, and that at low flux it
agrees with `plidar_est_mle`. On 300 Monte Carlo histograms of a scene shaped
like the recorded one, a flat topped table on 750 ps bins with 97 percent of
7000 cycles recording, its spread is 1.03 times the first photon bound, and
the spread of Coates then Poisson on the same histograms 1.84 times. On groups
thinned with q = 1/4 it stays consistent and reaches the bound for that keep,
and a histogram in which every cycle recorded still gives a finite estimate.

### Two halves of the frame

The pixels are split by column. The even columns calibrate: M, the response
and the shift of the Livox projection come from them. The odd columns are
measured, with the response the even ones gave.

A pixel enters when the dataset gives it an offset (under 10 m; the map holds
1000 elsewhere), its total is over 1500, its fullest bin lies between bins 26
and 639, and its total is under M. A pixel of the odd columns must also hold a
single return on its whole histogram after Coates. The bound is the bound of
one return, and on a pixel with two surfaces the spread would measure how
wrong that model is. The criterion is fixed: no bin more than 6 bins before
the rising edge, and no 5 bin average from 14 bins after it on, may exceed 10
percent of the peak. It sets aside 1663 odd pixels and leaves 19808.

### The cycle count

The method was designed to take M from the collapse of the shapes. For each M
on a grid from 6500 to 9000, Coates corrects the even pixels, each of the 4
flux classes under 6400 photons is averaged into one shape aligned on the
rising edge, and M is where the class shapes agree best: too large an M
under-corrects, too small an M over-corrects. The limit of 6400 leaves every
pixel some cycles at the smallest M tried. On sample 1 the minimum is at 6900.
The design also asked for M to be checked on a second sample before anything
was published, and that check fired. On sample 2 the curve falls to the lower
edge of the grid, so its minimum is at 6500 or below, although about a quarter
of the valid pixels there have totals above 6500 and no M can sit below a
total. Both frames come from the same sensor and differ in the scene, and
sample 2 has 6751 even pixels in the faintest class against 1697 in sample 1.
Where the shapes agree depends on which pixels the scene holds.

So M is now the ceiling of the totals: the smallest multiple of 100 at or
above the 99.9th percentile of the totals of the valid even pixels. No pixel
can record more photons than there were cycles, so the totals bound M from
below, and on a sensor whose brightest pixels record in nearly every cycle
they crowd up just under it. The percentile keeps the few pixels above the
rest from setting M: 4 even pixels hold 7060 to 7835 counts, more than one
record per cycle allows. Pixels at or above M are set aside, those 4 and 3
odd ones, the highest at 8424. Both samples give 7000, from 99.9th
percentiles of 6938 and 6951. The change came after the check on sample 2
fired and after a first run on sample 1, and the bands of the synthetic check
stayed as they were. The collapse stays as a diagnostic: every run prints its
curve, and `docs/real/collapse.csv` holds the one of sample 1.

The ceiling falls short when no pixel comes near saturation. The synthetic
check shows it: its flux tops out at 97.3 percent of cycles detected, and the
ceiling gives 6900 for a true 7000. That 1.4 percent follows from how bright
the synthetic pixels are and says nothing about the error on the recorded
frame.

### The response

The response is fitted on the even pixels that hold a single return, 19765 of
them, after Coates with M. Each is aligned on the half height point of its
rising edge, found to a fraction of a bin by linear interpolation. Averaging
the aligned windows would give the response already integrated over one bin,
and the library would integrate it a second time, so the density is fitted:
its values at nodes every eighth of a bin, from 6 bins before the edge to 40
after, with both ends held at zero. The pixels are grouped by the phase of
their edge within a bin, and the mean window of each phase is fitted by least
squares to the bin integrals of the density, each phase weighted by its share
of the pixels. A penalty on the second differences of the density holds down
the oscillations one bin of integration cannot see, and its weight was fixed
on synthetic pixels before the method saw recorded data. Negative values are
then set to zero, which removes 3.44e-4 of the area.

The fitted pulse is 8.11 bins (6.08 ns) wide at half height and rises from 10
to 90 percent in 3.32 bins. Its density is 23.8, 2.5 and 2.0 percent of the
peak at 10, 20 and 30 bins after the edge, and the fit ends at 40.

### Precision on groups

A recorded pixel is a single acquisition, so its spread cannot be seen on it.
Each odd histogram is split at random into 4 groups, every count going to one
of them. Each group is the histogram thinned with q = 1/4, and the groups add
up to it. Every estimator runs on the whole histogram and on every group. The
first photon fit takes the counts as they are, with M and q. The matched filter
and Coates then Poisson first apply Coates with M q trials, since a group is on
average the record of M q cycles at the same rates. A group counts as a
failure when its counts reach M q and Coates has no inverse, when an estimator
refuses it, as the matched filter does with a return beyond the reach of its
template, or when its estimate lies more than 5 bins from the same estimator's
estimate on the whole histogram. Where an estimator refuses the whole
histogram, all 4 groups fail.

The bound for a pixel is the first photon bound with keep q, at the t0 and
amplitudes the first photon fit finds on its whole histogram. For each flux
class and estimator the ratio is

    sqrt( sum_p s_p^2 / sum_p v_p )

with s_p^2 the variance of the pixel's group estimates that did not fail and
v_p its bound, over the pixels with at least 2 groups left. It is the precision
of one group, a quarter of the counts, against the bound for one group. The
precision of a whole histogram is not measured. The ratio is also a spread
within each pixel, so a bias that stays constant within a pixel does not show
in it. The bound holds for unbiased estimators, and some of these are not.

The groups of a split share the cycles of one acquisition, and a thinned group
differs from an acquisition with fewer cycles, so the fit and the bound both
use the thinned model. On synthetic pixels the first photon spread over the
groups of a split is 1.004 times its spread over groups thinned from
independent histograms.

### The synthetic check

`tools/real_stb.py --synthetic` runs in CI. The library simulates 3000 pixels
with a flat topped table, first photon detection, M = 7000, no background and
fluxes across the classes of the recorded frame, and the same steps run on
them, the filter for a single return included. The bands were fixed before
any recorded data went through the method and have not moved: M within 3
percent of the truth, the fitted response within 0.05 of the true one in L1,
the first photon ratio between 0.90 and 1.15 in every class, and the split
against independent groups between 0.90 and 1.10. With seed 1 the filter sets
aside none of the 1500 odd pixels under M, and the check gives an L1 of 0.0133
and first photon ratios from 1.012 to 1.062. The other estimators are printed
without a band.

### What came out

Where pile up is strong the first photon fit is at the bound and Coates then
Poisson at 1.9 to 2.7 times it, as in the Monte Carlo test; the README gives
every class. The centroid sits between 1.010 and 1.517 times the bound, and
the fullest bin, which reads whole bins, between 5.611 and 11.016.

The matched filter on Coates refuses the whole histogram of 3695 odd pixels,
3585 of them in the brightest class. Their returns sit 35 to 41 bins from the
start of the record, at or beyond the reach of a template that spans 40 bins
either side of a candidate, the length of the response's tail (section 5).
That takes 73.6 percent of the brightest class's groups, of the 74.1 percent
lost, and its ratio of 2.706 comes from the 1283 pixels left.

### The Livox reference

The Livox lidar gives a second, coarser reference. Its points go into the
frame of the single photon sensor through the published extrinsic and are
projected with the dataset's intrinsics, radial distortion included. The
published rotation R is not exactly orthogonal, with
R^T R = diag(1, 0.99848, 0.99848). Its transpose would shorten two coordinates
by 0.15 percent, so the exact inverse is used. The projection still needs a
shift in whole pixels, chosen on the even columns as the one that minimises the
spread of the fullest bin's residual: -5 columns and -5 rows. That search is
nearly flat in the row direction, so the shift is only loosely pinned. Only
pixels with a compact reference enter, at least 4 points whose quartiles lie
within 5 cm of each other.

The residuals are taken on 8922 odd pixels, and on 8446 for the matched
filter, which refuses the rest. The fullest bin is the only estimator in the
dataset's own conversion: its median is 4.70 cm and its residuals spread by
12.08 cm. The other four are each offset from that conversion, so each one's
median, 184 to 220 cm, is removed first, and the residuals then spread by 9.06
to 10.34 cm. The residual has a structure by rows that every estimator shares:
for the first photon fit, medians over bands of 24 rows go from -14.7 to
+10.2 cm. That points at calibration and registration. A spread of about a
decimetre cannot rank estimators whose bound for one group is 6.9 to 12.5 mm.

### Limits

- In the two faintest classes both likelihood fits sit 1.28 to 1.40 times the
  bound, close to each other, while on synthetic pixels the first photon fit
  stays within 1.012 to 1.062 in every class. The two handle pile up in
  different ways and share the response, M and the first photon model of a
  single detector, applied to a sum of 9 SPADs, so the excess likely comes
  from one of these. The response is a strong candidate. Its density is still
  2.0 percent of the peak 30 bins after the edge and the fit cuts it at 40
  bins, and one response serves every flux, although the class shapes still
  differ after Coates. M moves these two classes too (below).
- In the brightest class the first photon fit sits at 0.975. Counting 3
  degrees of freedom for each of its 4870 pixels, the standard error of that
  ratio is about 0.6 percent, which would put it about 4 standard errors
  under 1. That standard error ignores the spread of the bound across pixels
  and the dependence between the groups of one pixel, so it is only a rough
  guide, and at M = 6900 the same class reads 0.996. The bound, computed with
  the shared response, may be slightly pessimistic there, which would fit a
  pulse narrower than the response in the brightest pixels: after Coates that
  class still peaks higher and narrower than the others
  (`docs/real/shapes.png`).
- Each pixel sums 9 SPADs. The sum follows the first photon model of a single
  detector only if the 9 share one profile of rates.
- One M serves every pixel of the frame, read off the totals and rounded up to
  a hundred, and Coates and the first photon fit both take it as exact. Run
  again at M = 6900 and 7100 with `--cycles`, the first photon ratio ranges
  over 1.238 to 1.295, 1.312 to 1.380, 1.110 to 1.139, 1.009 to 1.014 and
  0.972 to 0.996 from the faintest class to the brightest, and Coates then
  Poisson over 1.245 to 1.298, 1.363 to 1.403, 1.447 to 1.489, 1.787 to 2.120
  and 2.444 to 2.936. The first photon fit moves most in the two faintest
  classes, by up to 0.068, and Coates then Poisson in the two brightest, by up
  to 0.49.
- One scene. Everything above comes from one frame, and a second frame entered
  only to check M.
- Only pixels with a single return are measured. The 1663 odd pixels the
  criterion sets aside are the hardest ones in the scene, and a second surface
  within about 14 bins (1.6 m) of the first merges with it and passes.
- The Livox reference is too coarse to rank the estimators (see its section).
- At exactly zero background the bins past the ends of a table have a rate of
  zero and are skipped, while a bin holding only the first sliver of the
  support keeps a fixed share of the timing information however thin it is,
  because the density rises linearly from zero there. So the bound includes
  information carried by the hard ends of a tabulated response, about 1.5
  percent of it for the test fixture, a property of a response with finite
  support. A pixel whose fit returns no background gets that bound.

## 9. Not here yet

- More than one surface per pixel, which is the case of fog, foliage or a
  window in front of the target. It turns the problem into model selection,
  deciding how many returns there are before estimating where they are, by
  likelihood ratio or an information criterion.
- Range folding. Returns past the unambiguous range are dropped rather than
  aliased back.
- A bound that survives the threshold region. The Cramer Rao bound describes
  local curvature and says nothing about a competing peak, which is why the
  measured error leaves it below ten photons. The Ziv Zakai family of bounds
  does cover that regime.
