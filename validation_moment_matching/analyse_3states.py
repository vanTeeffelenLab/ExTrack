#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tables for the 3 state sweep: what the coarser (age, state) buffer costs."""

import json
import sys

import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else 'results_3states.json'
res = json.load(open(PATH))
S = ['sequences', 'ages']
LAB = {'sequences': 'sequences', 'ages': 'ages'}
ks = sorted(set(r['k'] for r in res))
PAIRS = ['01', '02', '10', '12', '20', '21']

print('%s : %d points (%d k x %d replicates), %d tracks of 12 points, frame_len = 4'
      % (PATH, len(res), len(ks), len(res) // len(ks), res[0]['nb_tracks']))
print('')
print('=' * 100)
print('1. COST : one likelihood evaluation and one full fit')
print('=' * 100)
print('   scheme     | hypotheses  branches/step |  s per likelihood   s per fit    nfev')
for s in S:
    n_h = '3**4 = 81 max' if s == 'sequences' else '4*3 = 12'
    n_b = '3**5 = 243 max' if s == 'sequences' else '4*3**2 = 36'
    print('   %-10s | %-11s %-14s |  %14.3f %11.0f %7.0f'
          % (LAB[s], n_h, n_b,
             np.mean([r['schemes'][s]['eval_seconds'] for r in res]),
             np.mean([r['schemes'][s]['fit_seconds'] for r in res]),
             np.mean([r['schemes'][s]['nfev'] for r in res])))
sp_e = np.mean([r['schemes']['sequences']['eval_seconds'] for r in res]) / \
    np.mean([r['schemes']['ages']['eval_seconds'] for r in res])
sp_f = np.mean([r['schemes']['sequences']['fit_seconds'] / r['schemes']['ages']['fit_seconds']
                for r in res])
print('   speed-up of ages : %.2f x per likelihood, %.2f x per fit (medians of the ratios: %.2f x)'
      % (sp_e, sp_f, np.median([r['schemes']['sequences']['fit_seconds'] /
                                r['schemes']['ages']['fit_seconds'] for r in res])))

print('')
print('=' * 100)
print('2. LOG LIKELIHOOD at the true parameters (the two schemes on identical data)')
print('=' * 100)
print('     k   | d1/LocErr d2/LocErr |   sequences        ages       difference   per track')
for k in ks:
    sub = [r for r in res if r['k'] == k]
    a = np.mean([r['schemes']['sequences']['logL'] for r in sub])
    b = np.mean([r['schemes']['ages']['logL'] for r in sub])
    print('   %-5.1f |   %-7.1f   %-7.1f   | %12.2f %12.2f %11.3f  %+9.2e'
          % (k, k, 3 * k, a, b, b - a, (b - a) / sub[0]['nb_tracks']))

print('')
print('=' * 100)
print('3. FITTED PARAMETERS, error against the simulation truth (mean over 3 replicates)')
print('=' * 100)


def errs(r, s):
    f, t = r['schemes'][s]['fitted'], r['truth']
    le = np.array(t['LocErr'])
    out = {}
    out['d0'] = f['ds'][0] - t['ds'][0]                       # truth 0, absolute (um)
    out['d1'] = 100 * (f['ds'][1] - t['ds'][1]) / t['ds'][1]  # %
    out['d2'] = 100 * (f['ds'][2] - t['ds'][2]) / t['ds'][2]
    out['LocErr'] = 100 * (f['LocErr'] - t['LocErr']) / t['LocErr']
    out['F'] = 100 * np.mean(np.abs(np.array(f['Fs']) - np.array(t['Fs'])))
    out['T'] = 100 * np.mean([abs(f['T'][p] - 0.05) / 0.05 for p in PAIRS])
    out['Tsigned'] = 100 * np.mean([(f['T'][p] - 0.05) / 0.05 for p in PAIRS])
    return out


print('     k   |        d1 error (%)        |        d2 error (%)        |     LocErr error (%)')
print('         |  sequences     ages        |  sequences     ages        |  sequences     ages')
for k in ks:
    sub = [r for r in res if r['k'] == k]
    e = {s: [errs(r, s) for r in sub] for s in S}
    row = '   %-5.1f |' % k
    for key in ['d1', 'd2', 'LocErr']:
        for s in S:
            row += ' %+11.3f' % np.mean([x[key] for x in e[s]])
        row += '  |'
    print(row)

print('')
print('     k   |    d0 fitted (um, truth 0) |   transition rates |err| (%) |  fractions |err| (abs %)')
print('         |  sequences     ages        |  sequences     ages        |  sequences     ages')
for k in ks:
    sub = [r for r in res if r['k'] == k]
    e = {s: [errs(r, s) for r in sub] for s in S}
    row = '   %-5.1f |' % k
    for key in ['d0', 'T', 'F']:
        for s in S:
            row += ' %+11.4f' % np.mean([x[key] for x in e[s]])
        row += '  |'
    print(row)

print('')
print('  pooled over the 30 fits :')
print('  parameter |  mean bias seq.  mean bias ages  |  mean |err| seq.  mean |err| ages  |  ratio')
for key, unit in [('d1', '%'), ('d2', '%'), ('LocErr', '%'), ('Tsigned', '%'), ('d0', 'um')]:
    v = {s: np.array([errs(r, s)[key] for r in res]) for s in S}
    r_ = np.mean(np.abs(v['ages'])) / np.mean(np.abs(v['sequences']))
    print('  %-9s | %15.4f %15.4f  | %16.4f %17.4f  | %6.2f x  (%s)'
          % (key, v['sequences'].mean(), v['ages'].mean(),
             np.abs(v['sequences']).mean(), np.abs(v['ages']).mean(), r_, unit))
print('  (ratio > 1 means the coarser (age, state) buffer is that much further from the truth)')

print('')
print('=' * 100)
print('4. STATE PREDICTIONS, against the simulated hidden states')
print('=' * 100)
print('     k   |     accuracy              |      log loss             |       Brier')
print('         |  sequences     ages       |  sequences     ages       |  sequences     ages')
for k in ks:
    sub = [r for r in res if r['k'] == k]
    row = '   %-5.1f |' % k
    for key in ['accuracy', 'log_loss', 'brier']:
        for s in S:
            row += ' %11.5f' % np.mean([r['schemes'][s]['states'][key] for r in sub])
        row += '  |'
    print(row)
print('')
print('  pooled : ' + '   '.join(
    '%s %s %.5f' % (key, LAB[s], np.mean([r['schemes'][s]['states'][key] for r in res]))
    for key in ['accuracy', 'log_loss'] for s in S))
acc = {s: np.array([r['schemes'][s]['states']['accuracy'] for r in res]) for s in S}
ll = {s: np.array([r['schemes'][s]['states']['log_loss'] for r in res]) for s in S}
print('  accuracy lost by the coarser buffer : %+.4f points (%.3f %% relative)'
      % (100 * (acc['ages'] - acc['sequences']).mean(),
         100 * (acc['ages'] - acc['sequences']).mean() / acc['sequences'].mean()))
print('  log loss  : %+.5f (%.3f %% relative);  ages better at %d/%d points'
      % ((ll['ages'] - ll['sequences']).mean(),
         100 * (ll['ages'] - ll['sequences']).mean() / ll['sequences'].mean(),
         (ll['ages'] < ll['sequences']).sum(), len(res)))
