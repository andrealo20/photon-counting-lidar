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

Three, differing in how much of the model they use.

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

With B exactly zero the three parameter system stops being invertible in
floating point. Every bin the return does not reach has an expected count of
zero, so observing nothing there determines the background exactly, the
information about it runs away, and the products in the cofactor overflow.

That is a property of the model rather than a numerical accident, and it has a
clean limit: a background known exactly leaves two parameters, not three. The
library detects the overflow and drops B from the system, inverting the
remaining two by two block. Because the derivative of the response integrates
to zero, t0 and A are close to orthogonal and the answer lands near the bound
with everything known, which is what the background free limit should give.

## 7. What the tests actually establish

| test module | what it would catch |
|---|---|
| rng | a generator whose streams are not distinct, or whose variates have the wrong moments |
| irf | an error in the closed forms of section 2, an overflow in the tails, a derivative that does not match the function it claims to differentiate, a cancellation that eats the answer as the tail narrows |
| coates | an inverse that does not invert, a saturation reported as a number |
| parity | an error anywhere in either simulator, since they would have to be wrong in the same way to agree, with a control that measures how much the comparison could catch |
| estimate | an estimator that does not recover the parameters from its own noiseless forward model |
| crb | an information matrix that does not match the curvature of the likelihood it came from, and a three by three inverse that does not survive being taken a second way |

The tolerances in the response tests are set by the finite differences in the
tests, not by the library. A central difference carries a truncation error of
order (step/scale)^2 times the ratio of the third derivative to the first, and
in the tails that ratio is an order of magnitude above one over the width
squared, which puts the achievable agreement around one part in ten thousand.
Shrinking the step trades that error for a cancellation error and does not
help.

## 8. Not here yet

- Validation against a recorded time correlated single photon counting trace.
  The forward model is checked against itself and against closed forms, which
  is not the same as being checked against an instrument.
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
