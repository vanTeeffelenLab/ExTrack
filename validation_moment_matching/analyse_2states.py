#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tables for the 2 state sweep."""

import json
import sys

import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else 'results_2states_well_specified.json'
res = json.load(open(PATH))
modes = ['legacy', 'moment_matching']
LAB = {'legacy': 'legacy', 'moment_matching': 'moment m.', 'exact': 'exact'}

d1s = sorted(set(r['d1'] for r in res))
ps = sorted(set(r['p'] for r in res))
print('%s : %d points (%d d1 x %d p x %d replicates), %d tracks each'
      % (PATH, len(res), len(d1s), len(ps), len(res) // (len(d1s) * len(ps)), res[0]['nb_tracks']))
print('')


def get(r, mode, *keys):
    x = r['modes'][mode]
    for k in keys:
        x = x[k]
    return x


# ---------------------------------------------------------------------------
print('=' * 92)
print('1. LOG LIKELIHOOD ERROR at the true parameters, against the fusion free likelihood')
print('   (total over the data set, and rms per track)')
print('=' * 92)
print('  d1     p    |  total dlogL legacy   moment m.   |  rms/track legacy   moment m.   gain')
tot = {m: [] for m in modes}
rms = {m: [] for m in modes}
for d1 in d1s:
    for p in ps:
        sub = [r for r in res if r['d1'] == d1 and r['p'] == p]
        a = {m: np.mean([get(r, m, 'logL_error_at_truth') for r in sub]) for m in modes}
        b = {m: np.mean([get(r, m, 'logL_rms_error_at_truth') for r in sub]) for m in modes}
        for m in modes:
            tot[m].append(a[m])
            rms[m].append(b[m])
        print('  %-5.3f  %-4.2f |  %+14.3f %+12.3f   |  %11.3e %11.3e  %5.2f x'
              % (d1, p, a['legacy'], a['moment_matching'],
                 b['legacy'], b['moment_matching'], b['legacy'] / b['moment_matching']))
print('  ' + '-' * 88)
print('  mean       |  %+14.3f %+12.3f   |  %11.3e %11.3e  %5.2f x'
      % (np.mean(np.abs(tot['legacy'])), np.mean(np.abs(tot['moment_matching'])),
         np.mean(rms['legacy']), np.mean(rms['moment_matching']),
         np.mean(rms['legacy']) / np.mean(rms['moment_matching'])))
print('  (first two columns are mean |total error| over the 20 models)')

# ---------------------------------------------------------------------------
print('')
print('=' * 92)
print('2. FITTED PARAMETERS, mean over the 3 replicates, all 20 models pooled')
print('=' * 92)


def rel_err(r, mode, key, ref):
    v = get(r, mode, 'fitted', key)
    t = r['truth'].get(key, 0.0) if ref == 'truth' else get(r, ref, 'fitted', key)
    return v - t, (v - t) / t if t != 0 else np.nan


for ref, title in [('truth', 'distance to the SIMULATION TRUTH'),
                   ('exact', 'distance to the FUSION FREE fit (same data, same start)')]:
    print('')
    print('  ' + title)
    print('  param     |    legacy bias    legacy |err|  |  moment bias   moment |err|  | gain on |err|')
    for key, unit in [('d1', '%'), ('LocErr', '%'), ('T01', '%'), ('T10', '%'), ('F0', '%'),
                      ('d0', 'abs')]:
        line = '  %-9s |' % key
        vals = {}
        for m in modes:
            if unit == '%':
                e = np.array([rel_err(r, m, key, ref)[1] for r in res]) * 100
            else:
                e = np.array([rel_err(r, m, key, ref)[0] for r in res])
            vals[m] = e
            line += ' %+12.4f %13.4f  |' % (np.mean(e), np.mean(np.abs(e)))
        g = np.mean(np.abs(vals['legacy'])) / np.mean(np.abs(vals['moment_matching']))
        print(line + '  %6.2f x   %s' % (g, '(um)' if unit == 'abs' else '(%)'))
    if ref == 'truth':
        print('  (d0 truth is 0, so it is reported in um rather than in %)')

# ---------------------------------------------------------------------------
print('')
print('=' * 92)
print('3. STATE PREDICTIONS')
print('=' * 92)
for when, title in [('states_at_true_params', 'at the TRUE parameters'),
                    ('states_at_fitted_params', "at each mode's own FITTED parameters")]:
    print('')
    print('  ' + title)
    print('  metric                       |     legacy      moment m.   |  gain      exact')
    for key, better, fmt in [('mean_abs_dev_from_exact', 'low', '%12.3e'),
                             ('rms_dev_from_exact', 'low', '%12.3e'),
                             ('max_dev_from_exact', 'low', '%12.3e'),
                             ('log_loss', 'low', '%12.6f'),
                             ('brier', 'low', '%12.6f'),
                             ('accuracy', 'high', '%12.6f')]:
        v = {m: np.mean([get(r, m, when, key) for r in res]) for m in modes + ['exact']}
        if key.endswith('from_exact'):
            gain = '%6.2f x' % (v['legacy'] / v['moment_matching'])
            ex = '   0 (ref)'
        else:
            gain = '%+7.4f%%' % (100 * (v['moment_matching'] - v['legacy']) / v['legacy'])
            ex = fmt % v['exact']
        print(('  %-28s | ' + fmt + ' ' + fmt + '   |  %s  %s')
              % (key, v['legacy'], v['moment_matching'], gain, ex))

print('')
print('=' * 92)
print('4. HOW OFTEN IS MOMENT MATCHING CLOSER, point by point (60 fits per mode)')
print('=' * 92)
for ref in ['truth', 'exact']:
    print('  vs %-6s :' % ref, end='')
    for key in ['d1', 'LocErr', 'T01', 'T10', 'F0', 'd0']:
        wins = 0
        for r in res:
            el = abs(rel_err(r, 'legacy', key, ref)[0])
            em = abs(rel_err(r, 'moment_matching', key, ref)[0])
            wins += em < el
        print('  %s %2d/%d' % (key, wins, len(res)), end='')
    print('')
w = sum(get(r, 'moment_matching', 'states_at_true_params', 'mean_abs_dev_from_exact')
        < get(r, 'legacy', 'states_at_true_params', 'mean_abs_dev_from_exact') for r in res)
print('  state posterior closer to exact : %d/%d points' % (w, len(res)))

print('')
print('  mean fit time (s) : ' + '   '.join(
    '%s %.0f' % (LAB[m], np.mean([get(r, m, 'fit_seconds') for r in res]))
    for m in ['exact'] + modes))
print('  mean nfev         : ' + '   '.join(
    '%s %.0f' % (LAB[m], np.mean([get(r, m, 'nfev') for r in res]))
    for m in ['exact'] + modes))
