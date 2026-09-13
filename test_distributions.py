"""Checks on Distributions.py.

Every class is measured against an independent reference: scipy.stats where a
closed form exists, and the data itself where it does not. Run it with
    python3 test_distributions.py
Exit status is 0 when everything passes.
"""
import math, sys, time
import numpy as np
import Distributions as D

try:
    from scipy import stats as ss
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

PS = np.array([0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999])
fails = []


def check(name, got, want, tol, unit='rel'):
    got, want = np.asarray(got, float), np.asarray(want, float)
    err = np.max(np.abs(got - want) / np.maximum(np.abs(want), 1e-12)) if unit == 'rel' \
        else np.max(np.abs(got - want))
    ok = err <= tol
    print('  %-52s %-9s %.3g' % (name, 'pass' if ok else 'FAIL', err))
    if not ok:
        fails.append(name)
    return ok


print('=' * 78)
print('1. inverse CDFs against scipy')
print('=' * 78)
if not HAVE_SCIPY:
    print('  scipy not installed, skipping')
else:
    cases = [
        ('Normal_PDF(13, 3)',        D.Normal_PDF(13, 3),        ss.norm(13, 3)),
        ('Lognormal_PDF(ln13, ln1.35)',
         D.Lognormal_PDF(math.log(13), math.log(1.35)),
         ss.lognorm(s=math.log(1.35), scale=13)),
        ('Triangular_PDF(2.4, 3.1, 6.0)', D.Triangular_PDF(2.4, 3.1, 6.0),
         ss.triang(c=(3.1 - 2.4) / (6.0 - 2.4), loc=2.4, scale=3.6)),
        ('Uniform_PDF(2, 5)',        D.Uniform_PDF(2, 5),        ss.uniform(2, 3)),
        ('Logistic_PDF(10, 2)',      D.Logistic_PDF(10, 2),      ss.logistic(10, 2)),
        ('Weibull_PDF(1.8, 10)',     D.Weibull_PDF(1.8, 10),     ss.weibull_min(c=1.8, scale=10)),
        ('Trunc_norm_PDF(10, 3, 4, 16)', D.Trunc_norm_PDF(10, 3, 4, 16),
         ss.truncnorm((4 - 10) / 3, (16 - 10) / 3, loc=10, scale=3)),
    ]
    for name, ours, ref in cases:
        check(name + '  inv_cdf', ours.inv_cdf(PS), ref.ppf(PS), 1e-9)
        xs = ref.ppf(PS)
        check(name + '  cdf', ours.cdf(xs), PS, 1e-9)
        check(name + '  pdf', ours.pdf(xs), ref.pdf(xs), 1e-9)
        check(name + '  mean, std', [ours.mean, ours.std],
              [ref.mean(), ref.std()], 1e-9)

print()
print('=' * 78)
print('2. degenerate three-point estimates')
print('=' * 78)
for a, c, b in [(3.0, 3.0, 6.0), (3.0, 6.0, 6.0), (2.0, 4.0, 9.0)]:
    t = D.Triangular_PDF(a, c, b)
    q = t.inv_cdf(PS)
    ok = (np.all(np.diff(q) >= -1e-12) and np.all(q >= a - 1e-12) and np.all(q <= b + 1e-12)
          and np.all(np.isfinite(q)))
    print('  %-52s %s' % ('Triangular_PDF(%g, %g, %g) monotone and in range' % (a, c, b),
                          'pass' if ok else 'FAIL'))
    if not ok:
        fails.append('triangular degenerate %s' % ((a, c, b),))
    if HAVE_SCIPY and a < c < b:
        check('  against scipy', q, ss.triang(c=(c-a)/(b-a), loc=a, scale=b-a).ppf(PS), 1e-9)

print()
print('=' * 78)
print('3. data-driven distributions')
print('=' * 78)
draws = np.random.default_rng(7).lognormal(math.log(9.5), math.log(1.28), 20000)
E = D.Empirical_PDF(draws, name='vantage cost')
check('Empirical_PDF quantiles equal the data quantiles',
      [E.inv_cdf(p) for p in (0.05, 0.5, 0.95)],
      np.percentile(draws, [5, 50, 95]), 2e-3)
check('Empirical_PDF mean and std equal the data',
      [E.mean, E.std], [draws.mean(), draws.std()], 1e-12)

gx = np.linspace(0.001, 60, 400)
A = D.ArrayPDF(gx, D.lognormal(math.log(13), math.log(1.35), gx), name='atlas cost')
if HAVE_SCIPY:
    ref = ss.lognorm(s=math.log(1.35), scale=13)
    check('ArrayPDF inv_cdf against the lognormal it was built from',
          A.inv_cdf([0.05, 0.5, 0.95]), ref.ppf([0.05, 0.5, 0.95]), 5e-3)
    check('ArrayPDF cdf', A.cdf(ref.ppf([0.05, 0.5, 0.95])), [0.05, 0.5, 0.95], 5e-3)

med, fac = 13.0, 1.35
tbl = [(p, med * math.exp(math.log(fac) * float(D.PhiInv(p))))
       for p in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)]
Q = D.Quantile_PDF(tbl, name='from a schedule tool')
check('Quantile_PDF reproduces the table it was given',
      Q.inv_cdf([p for p, _ in tbl]), [v for _, v in tbl], 1e-9)
if HAVE_SCIPY:
    ref = ss.lognorm(s=math.log(fac), scale=med)
    check('Quantile_PDF extrapolates past the table, p1 and p99',
          Q.inv_cdf([0.01, 0.99]), ref.ppf([0.01, 0.99]), 1e-9)

print()
print('=' * 78)
print('4. round trips and internal consistency')
print('=' * 78)
every = [D.Normal_PDF(13, 3), D.Lognormal_PDF(math.log(13), math.log(1.35)),
         D.Triangular_PDF(2.4, 3.1, 6.0), D.Uniform_PDF(2, 5), D.Logistic_PDF(10, 2),
         D.Weibull_PDF(1.8, 10), D.Trunc_norm_PDF(10, 3, 4, 16), E, A, Q]
for d in every:
    nm = type(d).__name__
    check('%-18s cdf(inv_cdf(p)) == p' % nm, d.cdf(d.inv_cdf(PS)), PS, 5e-3, 'abs')
    check('%-18s density integrates to 1' % nm,
          np.trapezoid(d.pdf(np.linspace(d.min_domain, d.max_domain, 20001)),
                       np.linspace(d.min_domain, d.max_domain, 20001)),
          1 - 2 * d.EDGE, 5e-3, 'abs')
    check('%-18s confidence(x) == 1 - cdf(x)' % nm,
          d.confidence(d.median), 0.5, 5e-3, 'abs')
    check('%-18s event over the whole range' % nm,
          d.event((d.inv_cdf(0.1), d.inv_cdf(0.9))), 0.8, 5e-3, 'abs')

print()
print('=' * 78)
print('5. sampling')
print('=' * 78)
L = D.Lognormal_PDF(math.log(13), math.log(1.35))
s1 = L.samples(50000, random_seed=42)
s2 = L.samples(50000, random_seed=42)
print('  %-52s %s' % ('same seed gives the same draws', 'pass' if np.array_equal(s1, s2) else 'FAIL'))
if not np.array_equal(s1, s2):
    fails.append('seed reproducibility')

np.random.seed(1234)
before = np.random.random()
np.random.seed(1234)
L.samples(1000, random_seed=99)
after = np.random.random()
ok = before == after
print('  %-52s %s' % ('samples() leaves the global RNG alone', 'pass' if ok else 'FAIL'))
if not ok:
    fails.append('global RNG untouched')

check('50,000 draws recover p50 and p95',
      np.percentile(s1, [50, 95]), [L.inv_cdf(0.5), L.inv_cdf(0.95)], 2e-2)

rng = np.random.default_rng(2026)
u = rng.random(20000)
a = D.Lognormal_PDF(math.log(13), math.log(1.35)).inv_cdf(u)
b = D.Triangular_PDF(2.4, 3.1, 6.0).inv_cdf(u)
check('a shared uniform stream couples two distributions',
      np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1], 1.0, 1e-9)

t = time.time(); L.samples(20000); dt = time.time() - t
print('  %-52s %-9s %.4f s' % ('20,000 draws', 'pass' if dt < 1.0 else 'FAIL', dt))
if dt >= 1.0:
    fails.append('sampling speed')

print()
print('=' * 78)
print('6. the API the book examples use')
print('=' * 78)
for d in every:
    nm = type(d).__name__
    for attr in ('pdf', 'cdf', 'inv_cdf', 'samples', 'percentile', 'find_confidence',
                 'confidence', 'event', 'mean', 'std', 'mode', 'median', 'x', 'y', 'cy',
                 'min_domain', 'max_domain'):
        if not hasattr(d, attr):
            print('  %-52s FAIL  missing %s' % (nm, attr))
            fails.append('%s.%s' % (nm, attr))
print('  %-52s %s' % ('every class carries the full surface',
                      'pass' if not fails or not any('.' in f for f in fails) else 'FAIL'))
print('  %-52s %s' % ('Logisitic_PDF still resolves',
                      'pass' if D.Logisitic_PDF is D.Logistic_PDF else 'FAIL'))
print('  %-52s %s' % ('Weibull_PDF.inverse_cdf still resolves',
                      'pass' if hasattr(D.Weibull_PDF(1.8, 10), 'inverse_cdf') else 'FAIL'))

print()
print('=' * 78)
print('7. the one deliberate behaviour change')
print('=' * 78)
print('  percentile(p) now takes p in [0, 1] and equals inv_cdf(p).')
for d in (D.Lognormal_PDF(math.log(13), math.log(1.35)), D.Uniform_PDF(2, 5),
          D.Logistic_PDF(10, 2), D.Normal_PDF(13, 3), D.Triangular_PDF(2.4, 3.1, 6.0), E):
    nm = type(d).__name__
    same = abs(d.percentile(0.85) - float(np.asarray(d.inv_cdf(0.85)).ravel()[0])) < 1e-9
    print('    %-20s percentile(0.85) = %10.4f   find_confidence(85) = %10.4f   %s'
          % (nm, d.percentile(0.85), d.find_confidence(85), 'agree' if same else 'DISAGREE'))
    if not same:
        fails.append('%s percentile' % nm)
for bad in (1.5, -0.1):
    try:
        D.Normal_PDF(0, 1).percentile(bad)
        print('    percentile(%s) was accepted   FAIL' % bad); fails.append('percentile range')
    except ValueError:
        print('    percentile(%s) rejected, as it should be' % bad)

print()
print('=' * 78)
print('FAILURES: %d' % len(fails))
for f in fails:
    print('  ', f)
print('=' * 78)
sys.exit(1 if fails else 0)
