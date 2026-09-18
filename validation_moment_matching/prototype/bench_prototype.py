#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the C++ prototype, check it reproduces the numpy likelihood, and time it.

The point is not to ship a kernel but to measure the headroom that is left once
the python-level grouping loop is out of the way, so that "would C++ help?" has a
number attached.
"""

import ctypes
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import common                      # noqa: E402
import profile_likelihood as PL    # noqa: E402
from common import tracking        # noqa: E402

ZIG = r'C:\Users\Franc\anaconda3\envs\PyCGP\bin\zig.exe'
ENV_BIN = [r'C:\Users\Franc\anaconda3\envs\PyCGP\bin',
           r'C:\Users\Franc\anaconda3\envs\PyCGP\Library\bin']
DLL = os.path.join(HERE, 'extrack_ages.dll')
SRC = os.path.join(HERE, 'extrack_ages.cpp')


def build():
    env = dict(os.environ)
    env['PATH'] = os.pathsep.join(ENV_BIN + [env.get('PATH', '')])
    cmd = [ZIG, 'c++', '-target', 'x86_64-windows-gnu', '-O2', '-std=c++17',
           '-shared', SRC, '-o', DLL]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=HERE)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit('build failed')
    print('built extrack_ages.dll in %.1f s' % (time.time() - t0))


def load():
    lib = ctypes.CDLL(DLL)
    f = lib.extrack_ages_loglik
    d = ctypes.POINTER(ctypes.c_double)
    f.argtypes = [d, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                  ctypes.c_int, ctypes.c_double, d, d, d, ctypes.c_int, d]
    f.restype = None
    return f


def cpp_loglik(f, Cs, ds, Fs, TrMat, frame_len, loc_err, nthreads):
    Cs = np.ascontiguousarray(Cs, dtype=np.float64)
    S = len(ds)
    log_tr = np.ascontiguousarray(np.log(TrMat), dtype=np.float64)
    log_fs = np.ascontiguousarray(np.log(Fs), dtype=np.float64)
    d2 = np.asarray(ds, dtype=np.float64) ** 2
    pair = np.ascontiguousarray((d2[:, None] + d2[None, :]) / 2.0)
    out = np.empty(Cs.shape[0], dtype=np.float64)
    p = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    f(p(Cs), Cs.shape[0], Cs.shape[1], Cs.shape[2], S, frame_len, loc_err ** 2,
      p(log_tr), p(log_fs), p(pair), nthreads, p(out))
    return out


def timed(fn, n=5):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    build()
    f = load()
    common.set_mode('moment_matching')
    LOC = 0.02
    LocErr = np.array(LOC)[None, None, None]

    cases = [('2 states, frame_len 6', 2, 6, 2000, 20),
             ('3 states, frame_len 4', 3, 4, 2000, 12),
             ('3 states, frame_len 6', 3, 6, 2000, 12),
             ('4 states, frame_len 4', 4, 4, 2000, 12),
             ('2 states, 50 tracks', 2, 6, 50, 20)]

    print('')
    print('agreement with the numpy (age, state) recursion')
    for label, S, L, N, T in cases:
        Cs, ds, Fs, TrMat = PL.make(S, N, T)
        ref = np.asarray(tracking.Proba_Cs(Cs, LocErr, ds, Fs, TrMat, 1e-8, 0,
                                           common.BIG_FOV, 1, L, T, 0.2, 200, 'ages'))
        got = cpp_loglik(f, Cs, ds, Fs, TrMat, L, LOC, 1)
        err = np.abs(got - ref).max()
        rel = err / np.abs(ref).mean()
        print('  %-24s  max |cpp - numpy| = %.3e   (relative %.1e)  %s'
              % (label, err, rel, 'OK' if rel < 1e-12 else 'MISMATCH'))

    print('')
    hdr = ('  case                     tracks x len |  numpy sequences   numpy ages   '
           'cpp 1 thread   cpp 16 threads |  vs sequences   vs numpy ages')
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for label, S, L, N, T in cases:
        Cs, ds, Fs, TrMat = PL.make(S, N, T)
        t_seq = timed(lambda: tracking.Proba_Cs(Cs, LocErr, ds, Fs, TrMat, 1e-8, 0,
                                                common.BIG_FOV, 1, L, T, 0.2, 200, 'sequences'))
        t_age = timed(lambda: tracking.Proba_Cs(Cs, LocErr, ds, Fs, TrMat, 1e-8, 0,
                                                common.BIG_FOV, 1, L, T, 0.2, 200, 'ages'))
        t_c1 = timed(lambda: cpp_loglik(f, Cs, ds, Fs, TrMat, L, LOC, 1), 20)
        t_c16 = timed(lambda: cpp_loglik(f, Cs, ds, Fs, TrMat, L, LOC, 16), 20)
        print('  %-24s %5d x %-3d |  %13.4f %12.4f %14.5f %15.5f |  %8.0f x %13.1f x'
              % (label, N, T, t_seq, t_age, t_c1, t_c16,
                 t_seq / t_c16, t_age / t_c16))

    print('')
    print('  thread scaling, 3 states / frame_len 4 / 2000 tracks x 12')
    Cs, ds, Fs, TrMat = PL.make(3, 2000, 12)
    base = None
    line = '   '
    for nt in [1, 2, 4, 8, 16, 24, 32]:
        t = timed(lambda nt=nt: cpp_loglik(f, Cs, ds, Fs, TrMat, 4, LOC, nt), 20)
        base = base or t
        line += '  %d thr %.2fx' % (nt, base / t)
    print(line)


if __name__ == '__main__':
    main()
