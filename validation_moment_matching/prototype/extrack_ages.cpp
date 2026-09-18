// Prototype: the (segment age, current state) recursion of ExTrack, in C++.
//
// This exists to answer one question with a measurement instead of an
// extrapolation: once the python-level grouping loop is gone, is there still a
// large win left in a compiled kernel?
//
// It reproduces P_Cs_inter_bound_stats_ages exactly for the case the benchmark
// uses: one localization error for every peak and dimension, one diffusion
// length per state, nb_substeps = 1, isBL = 0, and no field-of-view term (the
// benchmark sets cell_dims large and min_len = track_len, so Lp_stay never
// enters the loop). Everything else -- the moment matched fusion, the two pass
// mean/variance, the log-sum-exp bookkeeping -- is the same arithmetic in the
// same order.
//
// Build:  zig c++ -target x86_64-windows-gnu -O2 -std=c++17 -shared
//                 extrack_ages.cpp -o extrack_ages.dll

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr double LOG_2PI = 1.8378770664093454835606594728112;

struct Problem {
    int nb_tracks, track_len, nb_dims, nb_states, frame_len;
    const double* Cs;        // (nb_tracks, track_len, nb_dims)
    const double* log_tr;    // (nb_states, nb_states)  [from][to]
    const double* log_fs;    // (nb_states,)
    const double* pair_d2;   // (nb_states, nb_states)  (d2[s] + d2[j]) / 2
    double l2;               // localization variance
};

// One track. Scratch buffers are caller-owned so the inner loop allocates nothing.
double run_track(const Problem& P, int track,
                 std::vector<double>& LP, std::vector<double>& m,
                 std::vector<double>& s2, std::vector<double>& nLP,
                 std::vector<double>& nm, std::vector<double>& ns2,
                 std::vector<double>& bLP, std::vector<double>& bm,
                 std::vector<double>& bs2, std::vector<double>& w) {
    const int S = P.nb_states, D = P.nb_dims, L = P.frame_len;
    const double* C = P.Cs + (size_t)track * P.track_len * D;

    // time 0: one hypothesis per state, carrying the posterior of r_0 given c_0
    int n_act = 1;
    for (int s = 0; s < S; ++s) {
        LP[s] = P.log_fs[s];
        for (int d = 0; d < D; ++d) {
            m[s * D + d] = C[d];
            s2[s * D + d] = P.l2;
        }
    }

    for (int step = 0; step < P.track_len - 1; ++step) {
        const int P_now = n_act * S;

        // ---- fold observation c_step, except at the very first transition ----
        if (step > 0) {
            const double* Ci = C + (size_t)step * D;
            for (int c = 0; c < P_now; ++c) {
                double K = 0.0;
                for (int d = 0; d < D; ++d) {
                    const double sv = s2[c * D + d], mv = m[c * D + d];
                    const double tot = sv + P.l2;
                    const double dv = Ci[d] - mv;
                    K += -0.5 * (LOG_2PI + std::log(tot)) - dv * dv / (2.0 * tot);
                    m[c * D + d] = (mv * P.l2 + Ci[d] * sv) / tot;
                    s2[c * D + d] = sv * P.l2 / tot;
                }
                LP[c] += K;
            }
        }

        // ---- branch every hypothesis over the nb_states next states ----
        for (int c = 0; c < P_now; ++c) {
            const int s = c % S;
            for (int j = 0; j < S; ++j) {
                const int b = c * S + j;
                bLP[b] = LP[c] + P.log_tr[s * S + j];
                const double add = P.pair_d2[s * S + j];
                for (int d = 0; d < D; ++d) {
                    bm[b * D + d] = m[c * D + d];
                    bs2[b * D + d] = s2[c * D + d] + add;
                }
            }
        }

        const int n_new = std::min(n_act + 1, L);

        // ---- newborns: every source arriving in j is fused into (age 0, j) ----
        for (int j = 0; j < S; ++j) {
            int nsrc = 0;
            double mx = -std::numeric_limits<double>::infinity();
            for (int a = 0; a < n_act; ++a)
                for (int s = 0; s < S; ++s) {
                    if (s == j) continue;
                    const int b = (a * S + s) * S + j;
                    w[nsrc] = bLP[b];
                    mx = std::max(mx, bLP[b]);
                    ++nsrc;
                }
            double W = 0.0;
            for (int k = 0; k < nsrc; ++k) { w[k] = std::exp(w[k] - mx); W += w[k]; }
            const double inv = 1.0 / W;

            // pass 1: the mean
            for (int d = 0; d < D; ++d) nm[j * D + d] = 0.0;
            int k = 0;
            for (int a = 0; a < n_act; ++a)
                for (int s = 0; s < S; ++s) {
                    if (s == j) continue;
                    const int b = (a * S + s) * S + j;
                    const double wk = w[k] * inv;
                    for (int d = 0; d < D; ++d) nm[j * D + d] += wk * bm[b * D + d];
                    ++k;
                }
            // pass 2: the within-branch variance plus the spread of the means
            for (int d = 0; d < D; ++d) ns2[j * D + d] = 0.0;
            k = 0;
            for (int a = 0; a < n_act; ++a)
                for (int s = 0; s < S; ++s) {
                    if (s == j) continue;
                    const int b = (a * S + s) * S + j;
                    const double wk = w[k] * inv;
                    for (int d = 0; d < D; ++d) {
                        const double dm = bm[b * D + d] - nm[j * D + d];
                        ns2[j * D + d] += wk * (bs2[b * D + d] + dm * dm);
                    }
                    ++k;
                }
            nLP[j] = mx + std::log(W);
        }

        // ---- stays: the age advances; the two oldest slabs merge when full ----
        const int n_copy = (n_act < L) ? n_act : L - 2;
        for (int a = 0; a < n_copy; ++a)
            for (int s = 0; s < S; ++s) {
                const int src = (a * S + s) * S + s;
                const int dst = (a + 1) * S + s;
                nLP[dst] = bLP[src];
                for (int d = 0; d < D; ++d) {
                    nm[dst * D + d] = bm[src * D + d];
                    ns2[dst * D + d] = bs2[src * D + d];
                }
            }
        if (n_act >= L) {
            for (int s = 0; s < S; ++s) {
                const int b0 = ((L - 2) * S + s) * S + s, b1 = ((L - 1) * S + s) * S + s;
                const int dst = (L - 1) * S + s;
                const double mx = std::max(bLP[b0], bLP[b1]);
                const double w0 = std::exp(bLP[b0] - mx), w1 = std::exp(bLP[b1] - mx);
                const double W = w0 + w1, a0 = w0 / W, a1 = w1 / W;
                for (int d = 0; d < D; ++d) {
                    const double mu = a0 * bm[b0 * D + d] + a1 * bm[b1 * D + d];
                    const double d0 = bm[b0 * D + d] - mu, d1 = bm[b1 * D + d] - mu;
                    nm[dst * D + d] = mu;
                    ns2[dst * D + d] = a0 * (bs2[b0 * D + d] + d0 * d0)
                                     + a1 * (bs2[b1 * D + d] + d1 * d1);
                }
                nLP[dst] = mx + std::log(W);
            }
        }

        std::swap(LP, nLP); std::swap(m, nm); std::swap(s2, ns2);
        n_act = n_new;
    }

    // ---- last observation, then log-sum-exp over the buffer ----
    const double* Ci = C + (size_t)(P.track_len - 1) * D;
    const int P_now = n_act * S;
    double mx = -std::numeric_limits<double>::infinity();
    for (int c = 0; c < P_now; ++c) {
        double K = 0.0;
        for (int d = 0; d < D; ++d) {
            const double tot = s2[c * D + d] + P.l2;
            const double dv = Ci[d] - m[c * D + d];
            K += -0.5 * (LOG_2PI + std::log(tot)) - dv * dv / (2.0 * tot);
        }
        LP[c] += K;
        mx = std::max(mx, LP[c]);
    }
    double W = 0.0;
    for (int c = 0; c < P_now; ++c) W += std::exp(LP[c] - mx);
    return mx + std::log(W);
}

void run_block(const Problem& P, int lo, int hi, double* out) {
    const int cap = P.frame_len * P.nb_states;
    const int bcap = cap * P.nb_states;
    std::vector<double> LP(cap), m(cap * P.nb_dims), s2(cap * P.nb_dims);
    std::vector<double> nLP(cap), nm(cap * P.nb_dims), ns2(cap * P.nb_dims);
    std::vector<double> bLP(bcap), bm(bcap * P.nb_dims), bs2(bcap * P.nb_dims);
    std::vector<double> w(bcap);
    for (int t = lo; t < hi; ++t)
        out[t] = run_track(P, t, LP, m, s2, nLP, nm, ns2, bLP, bm, bs2, w);
}

}  // namespace

EXPORT void extrack_ages_loglik(const double* Cs, int nb_tracks, int track_len, int nb_dims,
                                int nb_states, int frame_len, double locerr2,
                                const double* log_tr, const double* log_fs,
                                const double* pair_d2, int nthreads, double* out) {
    Problem P{nb_tracks, track_len, nb_dims, nb_states, std::max(frame_len, 2),
              Cs, log_tr, log_fs, pair_d2, locerr2};
    if (nthreads <= 0) nthreads = (int)std::thread::hardware_concurrency();
    nthreads = std::max(1, std::min(nthreads, nb_tracks));
    if (nthreads == 1) { run_block(P, 0, nb_tracks, out); return; }

    std::vector<std::thread> pool;
    pool.reserve(nthreads);
    const int chunk = (nb_tracks + nthreads - 1) / nthreads;
    for (int i = 0; i < nthreads; ++i) {
        const int lo = i * chunk, hi = std::min(nb_tracks, lo + chunk);
        if (lo >= hi) break;
        pool.emplace_back([&P, lo, hi, out]() { run_block(P, lo, hi, out); });
    }
    for (auto& th : pool) th.join();
}
