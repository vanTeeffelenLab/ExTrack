#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests of the moment matched fusion.

1) fuse_gaussians reproduces the 2 first moments of the mixture it replaces,
   computed here by brute force from a large sample.
2) the legacy fusion reproduces the mean but not the variance.
3) fusing a single branch, or branches that are all identical, is the identity.
4) the fusion is exact (matches the mixture) when all the branch means coincide.
"""

import numpy as np

from common import tracking

rng = np.random.default_rng(0)
fails = []


def check(name, ok, detail=''):
    print(('  OK   ' if ok else '  FAIL ') + name + ('   ' + detail if detail else ''))
    if not ok:
        fails.append(name)


print('fuse_gaussians against the brute force moments of the mixture')

nb_branches, nb_dims = 7, 3
m = rng.normal(0, 0.05, (4, nb_branches, nb_dims))
s2 = rng.uniform(1e-4, 4e-4, (4, nb_branches, 1))
LP = rng.normal(0, 3, (4, nb_branches))

w = np.exp(LP - LP.max(1, keepdims=True))
w = w / w.sum(1, keepdims=True)
exact_mu = np.sum(w[:, :, None] * m, 1)
exact_var = np.sum(w[:, :, None] * (s2 + (m - exact_mu[:, None]) ** 2), 1)
exact_LP = np.log(np.sum(np.exp(LP - LP.max(1, keepdims=True)), 1)) + LP.max(1)

tracking.set_fusion_method(moment_matching=True, isotropic_variance=False)
mu, var, lp = tracking.fuse_gaussians(m, s2, LP, 1)
check('mean', np.allclose(mu, exact_mu), 'max err %.2e' % np.abs(mu - exact_mu).max())
check('variance (per dimension)', np.allclose(var, exact_var),
      'max err %.2e' % np.abs(var - exact_var).max())
check('log weight', np.allclose(lp, exact_LP), 'max err %.2e' % np.abs(lp - exact_LP).max())

# a Monte Carlo draw from the mixture, as an independent check of the algebra
comp = np.array([rng.choice(nb_branches, size=400000, p=w[t]) for t in range(4)])
sample = m[np.arange(4)[:, None], comp] + rng.normal(size=(4, 400000, nb_dims)) * np.sqrt(
    s2[np.arange(4)[:, None], comp])
check('mean vs 4e5 samples', np.abs(sample.mean(1) - mu).max() < 5e-4,
      'max err %.2e' % np.abs(sample.mean(1) - mu).max())
rel = np.abs(sample.var(1) - var) / var
check('variance vs 4e5 samples', rel.max() < 0.02, 'max rel err %.3f' % rel.max())

print('')
print('the legacy fusion keeps the mean but loses the spread')
tracking.set_fusion_method(moment_matching=False)
mu0, var0, lp0 = tracking.fuse_gaussians(m, s2, LP, 1)
check('same mean as moment matching', np.allclose(mu0, mu))
check('same log weight', np.allclose(lp0, lp))
missing = np.sum(w[:, :, None] * (m - exact_mu[:, None]) ** 2, 1)
check('variance is short by exactly the spread of the means',
      np.allclose(var0 + missing, exact_var))
check('legacy variance is always the smaller one', np.all(var0 <= var + 1e-18),
      'median deficit %.1f %%' % (100 * np.median(missing / exact_var)))

print('')
print('isotropic variant')
tracking.set_fusion_method(moment_matching=True, isotropic_variance=True)
mu2, var2, lp2 = tracking.fuse_gaussians(m, s2, LP, 1)
check('keeps a single variance', var2.shape[-1] == 1)
check('matches the trace of the second moment',
      np.allclose(var2[..., 0] * nb_dims, exact_var.sum(-1)))

print('')
print('degenerate cases')
tracking.set_fusion_method(moment_matching=True, isotropic_variance=False)
mu1, var1, lp1 = tracking.fuse_gaussians(m[:, :1], s2[:, :1], LP[:, :1], 1)
check('a single branch is left untouched (mean)', np.allclose(mu1, m[:, 0]))
check('a single branch is left untouched (variance)', np.allclose(var1, s2[:, 0]))
check('a single branch is left untouched (log weight)', np.allclose(lp1, LP[:, 0]))

m_same = np.repeat(m[:, :1], nb_branches, 1)
mu3, var3, lp3 = tracking.fuse_gaussians(m_same, s2, LP, 1)
check('identical means give no spread', np.allclose(var3, np.sum(w[:, :, None] * s2, 1)))

# fusing along several axes at once, as fuse_tracks_general does
m4 = m.reshape(4, 7, 1, nb_dims)
s4 = s2.reshape(4, 7, 1, 1)
mu4, var4, lp4 = tracking.fuse_gaussians(m4, s4, LP.reshape(4, 7, 1), (1, 2))
check('fusion over a tuple of axes', np.allclose(mu4, mu) and np.allclose(var4, var))

# invariance to a constant offset of all the log weights (only the ratios matter)
mu5, var5, lp5 = tracking.fuse_gaussians(m, s2, LP + 1234.5, 1)
check('invariant to a shift of the log weights',
      np.allclose(mu5, mu) and np.allclose(var5, var) and np.allclose(lp5, lp + 1234.5))

tracking.set_fusion_method()  # back to the default

print('')
print('%d/%d checks passed' % (13 + 5 - len(fails), 18))
if fails:
    print('FAILED: ' + ', '.join(fails))
    raise SystemExit(1)
print('all good')
