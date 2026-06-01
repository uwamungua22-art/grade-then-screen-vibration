# -*- coding: utf-8 -*-
"""
WS6: Trend-screening statistical robustness (JVET supplementary analysis)
=========================================================================
Self-contained script implementing three statistical procedures for the
grade-then-screen vibration monitoring paper:

  1. BH-FDR & BY correction on Spearman trend tests (12 named points)
  2. Leave-one-out (LOO) stability analysis for significant points
  3. Effective sample size (n_eff) adjustment with autocorrelation diagnostics
  4. No-grading ablation comparison (spikes included)

Reads only the local parquet feature table; does not touch raw waveforms.
"""
import sys
import argparse
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm, theilslopes
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import acf as sm_acf

SEED = 42
np.random.seed(SEED)

# Named-point regex (matches reversing, commutator, ball, screw, guide, rail,
# seat, conn, lift -- identical to ws1_4_methods.py convention)
NAMED_RE = r'reversing|commutator|ball|screw|guide|rail|seat|conn|lift'
MIN_SESSIONS = 8


# ──────────────────────────────────────────────────────────────────────
#  Data loading (self-contained, mirrors ws1_4_methods.load)
# ──────────────────────────────────────────────────────────────────────
def _parse_timestamp(m):
    g = re.search(r'(\d\d)-(\d\d)-(\d\d) (\d\d)-(\d\d)-(\d\d)', m)
    if g:
        return pd.Timestamp(2000 + int(g[1]), int(g[2]), int(g[3]),
                            int(g[4]), int(g[5]), int(g[6]))
    return pd.NaT


def load(path):
    df = pd.read_parquet(path)
    df['adc_fs'] = df['range_to'].astype(float) * df['scale_g_per_count']
    df['pr'] = df['td_peak'] / df['adc_fs']
    df['is_empty'] = df['n_samples'].fillna(0) < 8000
    df['hard_spike'] = (~df['is_empty']) & (df['pr'] >= 0.98)
    df['usable'] = (~df['is_empty']) & (~df['hard_spike']) & (df['sample_rate'] == 8000)
    df['t'] = df['measurement'].map(_parse_timestamp)
    return df


def named_usable(df):
    u = df[df['usable']].copy()
    return u[u.point_name.str.contains(NAMED_RE, case=False, na=False)]


def named_points(named):
    return sorted([p for p, c in named.point_name.value_counts().items()
                   if c >= MIN_SESSIONS])


# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────
def lag_acf(x, nlags=3):
    """Biased ACF via statsmodels (consistent with DW approximation)."""
    x = np.asarray(x, float)
    if len(x) < nlags + 2 or np.allclose(x, x.mean()):
        return [np.nan] * nlags
    ac = sm_acf(x, nlags=nlags, fft=False)
    return list(ac[1:nlags + 1])


def durbin_watson(resid):
    resid = np.asarray(resid, float)
    return float(np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2))


def n_eff_from_r1(n, r1):
    """Effective sample size: n_eff = n(1-r1)/(1+r1)."""
    if np.isnan(r1) or abs(r1) >= 1:
        return np.nan
    return n * (1 - r1) / (1 + r1)


def fisher_z_ci(rho, n_eff, alpha=0.05):
    """Fisher z-transform 95% CI for Spearman rho, using n_eff."""
    if np.isnan(n_eff) or n_eff < 4 or abs(rho) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(rho)
    se = 1.03 / np.sqrt(n_eff - 3)   # Spearman correction factor 1.03
    zc = norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))


# ──────────────────────────────────────────────────────────────────────
#  1. BH-FDR & BY correction
# ──────────────────────────────────────────────────────────────────────
def section1_fdr(named, pts):
    print('=' * 80)
    print('SECTION 1: BH-FDR and BY correction on Spearman trend tests (q = 0.05)')
    print('=' * 80)

    rows = []
    for p in pts:
        s = named[named.point_name == p].dropna(subset=['td_rms']).sort_values('t')
        n = len(s)
        y = s['td_rms'].values.astype(float)
        rho, pval = spearmanr(np.arange(n), y)
        rows.append({'point': p, 'n': n, 'rho': round(rho, 4),
                     'p_raw': pval})

    tbl = pd.DataFrame(rows)
    _, p_bh, _, _ = multipletests(tbl['p_raw'], alpha=0.05, method='fdr_bh')
    _, p_by, _, _ = multipletests(tbl['p_raw'], alpha=0.05, method='fdr_by')
    tbl['p_adj_BH'] = p_bh
    tbl['p_adj_BY'] = p_by
    tbl['sig_BH'] = tbl['p_adj_BH'] < 0.05
    tbl['sig_BY'] = tbl['p_adj_BY'] < 0.05
    tbl = tbl.sort_values('p_raw').reset_index(drop=True)

    # Print table
    hdr = f"{'Point':42s} {'n':>3s} {'rho':>7s} {'p_raw':>11s} {'p_adj_BH':>11s} {'p_adj_BY':>11s} {'BH':>3s} {'BY':>3s}"
    print(hdr)
    print('-' * len(hdr))
    for _, r in tbl.iterrows():
        bh = 'Y' if r.sig_BH else 'N'
        by = 'Y' if r.sig_BY else 'N'
        print(f"{r.point:42s} {r.n:3.0f} {r.rho:+7.4f} {r.p_raw:11.4e} "
              f"{r.p_adj_BH:11.4e} {r.p_adj_BY:11.4e} {bh:>3s} {by:>3s}")

    n_bh = int(tbl['sig_BH'].sum())
    n_by = int(tbl['sig_BY'].sum())
    print(f"\nSignificant (BH q<0.05): {n_bh}/{len(tbl)}")
    print(f"Significant (BY q<0.05): {n_by}/{len(tbl)}")
    print()
    return tbl


# ──────────────────────────────────────────────────────────────────────
#  2. Leave-one-out (LOO) stability
# ──────────────────────────────────────────────────────────────────────
def section2_loo(named, sig_points):
    print('=' * 80)
    print('SECTION 2: Leave-one-out (LOO) stability for BH-significant points')
    print('=' * 80)

    hdr = f"{'Point':42s} {'n':>3s} {'rho_full':>9s} {'rho_min':>9s} {'rho_max':>9s} {'range':>9s} {'sign_flip':>10s}"
    print(hdr)
    print('-' * len(hdr))

    for p in sig_points:
        s = named[named.point_name == p].dropna(subset=['td_rms']).sort_values('t')
        n = len(s)
        y = s['td_rms'].values.astype(float)
        rho_full, _ = spearmanr(np.arange(n), y)

        rhos = []
        for i in range(n):
            y_loo = np.delete(y, i)
            rho_i, _ = spearmanr(np.arange(len(y_loo)), y_loo)
            rhos.append(rho_i)

        rho_min = min(rhos)
        rho_max = max(rhos)
        sign_flip = any(np.sign(r) != np.sign(rho_full) for r in rhos)
        print(f"{p:42s} {n:3d} {rho_full:+9.4f} {rho_min:+9.4f} "
              f"{rho_max:+9.4f} {rho_max - rho_min:9.4f} "
              f"{'YES' if sign_flip else 'no':>10s}")

    print()


# ──────────────────────────────────────────────────────────────────────
#  3. n_eff effective sample size adjustment
# ──────────────────────────────────────────────────────────────────────
def section3_neff(named, sig_points):
    print('=' * 80)
    print('SECTION 3: Autocorrelation diagnostics and n_eff adjustment')
    print('=' * 80)

    # 3a. Autocorrelation table
    print('\n--- 3a. Theil-Sen residual autocorrelation ---')
    hdr = (f"{'Point':42s} {'n':>3s} {'rho':>7s} {'lag1':>7s} {'lag2':>7s} "
           f"{'lag3':>7s} {'DW':>6s} {'n_eff':>6s}")
    print(hdr)
    print('-' * len(hdr))

    neff_dict = {}
    for p in sig_points:
        s = named[named.point_name == p].dropna(subset=['td_rms']).sort_values('t')
        n = len(s)
        idx = np.arange(n)
        y = s['td_rms'].values.astype(float)
        rho, _ = spearmanr(idx, y)

        slope, intercept, _, _ = theilslopes(y, idx)
        resid = y - (slope * idx + intercept)

        lags = lag_acf(resid, nlags=3)
        dw = durbin_watson(resid)
        ne = n_eff_from_r1(n, lags[0])
        neff_dict[p] = ne

        print(f"{p:42s} {n:3d} {rho:+7.4f} {lags[0]:7.4f} {lags[1]:7.4f} "
              f"{lags[2]:7.4f} {dw:6.2f} {ne:6.1f}")

    # 3b. Fisher z-transform adjusted CI
    print('\n--- 3b. Fisher z-transform CI (using n_eff) ---')
    hdr2 = f"{'Point':42s} {'n':>3s} {'n_eff':>6s} {'rho':>7s} {'CI_lo':>7s} {'CI_hi':>7s} {'excludes_0':>11s}"
    print(hdr2)
    print('-' * len(hdr2))

    for p in sig_points:
        s = named[named.point_name == p].dropna(subset=['td_rms']).sort_values('t')
        n = len(s)
        rho, _ = spearmanr(np.arange(n), s['td_rms'].values.astype(float))
        ne = neff_dict[p]
        ci_lo, ci_hi = fisher_z_ci(rho, ne)
        if np.isnan(ci_lo):
            ci_str = '  n_eff too small for CI'
            excludes = 'NO'
        else:
            ci_str = f"{ci_lo:+7.4f} {ci_hi:+7.4f}"
            excludes = 'yes' if (ci_lo > 0 or ci_hi < 0) else 'NO'
        print(f"{p:42s} {n:3d} {ne:6.1f} {rho:+7.4f} {ci_str} {excludes:>11s}")

    # 3c. Piecewise model for CH19
    print('\n--- 3c. Piecewise (single change-point) model for CH19 ---')
    ch19_name = [p for p in sig_points if 'CH19' in p]
    if ch19_name:
        p = ch19_name[0]
        s = named[named.point_name == p].dropna(subset=['td_rms']).sort_values('t')
        n = len(s)
        idx = np.arange(n)
        y = s['td_rms'].values.astype(float)

        # Search for optimal changepoint (minimise total Theil-Sen SSE)
        best_cp, best_sse = None, np.inf
        for cp in range(3, n - 3):
            left_idx, right_idx = idx[:cp + 1], idx[cp:]
            sl, il, _, _ = theilslopes(y[:cp + 1], left_idx)
            sr, ir, _, _ = theilslopes(y[cp:], right_idx)
            pred = np.empty_like(y)
            pred[:cp + 1] = sl * left_idx + il
            pred[cp + 1:] = sr * idx[cp + 1:] + ir
            sse = np.sum((y - pred) ** 2)
            if sse < best_sse:
                best_sse, best_cp = sse, cp

        # Compute piecewise residuals at optimal cp
        sl, il, _, _ = theilslopes(y[:best_cp + 1], idx[:best_cp + 1])
        sr, ir, _, _ = theilslopes(y[best_cp:], idx[best_cp:])
        pred = np.empty_like(y)
        pred[:best_cp + 1] = sl * idx[:best_cp + 1] + il
        pred[best_cp + 1:] = sr * idx[best_cp + 1:] + ir
        resid_pw = y - pred

        lags_pw = lag_acf(resid_pw, nlags=3)
        dw_pw = durbin_watson(resid_pw)
        ne_pw = n_eff_from_r1(n, lags_pw[0])

        # Linear comparison
        slope, intercept, _, _ = theilslopes(y, idx)
        resid_lin = y - (slope * idx + intercept)
        lags_lin = lag_acf(resid_lin, nlags=3)
        ne_lin = n_eff_from_r1(n, lags_lin[0])

        rho, _ = spearmanr(idx, y)
        ci_lin = fisher_z_ci(rho, ne_lin)
        ci_pw = fisher_z_ci(rho, ne_pw)

        print(f"  Optimal changepoint: index {best_cp}")
        print(f"  Left slope  = {sl:.6f}")
        print(f"  Right slope = {sr:.6f}")
        print(f"  {'Model':<12s} {'lag1':>7s} {'lag2':>7s} {'lag3':>7s} {'DW':>6s} {'n_eff':>6s} {'CI_lo':>7s} {'CI_hi':>7s}")
        print(f"  {'Linear':<12s} {lags_lin[0]:7.4f} {lags_lin[1]:7.4f} {lags_lin[2]:7.4f} "
              f"{durbin_watson(resid_lin):6.2f} {ne_lin:6.1f} {ci_lin[0]:+7.4f} {ci_lin[1]:+7.4f}")
        print(f"  {'Piecewise':<12s} {lags_pw[0]:7.4f} {lags_pw[1]:7.4f} {lags_pw[2]:7.4f} "
              f"{dw_pw:6.2f} {ne_pw:6.1f} {ci_pw[0]:+7.4f} {ci_pw[1]:+7.4f}")

    print()


# ──────────────────────────────────────────────────────────────────────
#  4. No-grading ablation comparison
# ──────────────────────────────────────────────────────────────────────
def section4_ablation(df, named_graded, pts_graded, tbl_graded):
    print('=' * 80)
    print('SECTION 4: No-grading ablation (all non-empty 8 kHz, spikes included)')
    print('=' * 80)

    # Grading counts
    ne = ~df['is_empty']
    n_trust = int((ne & (df['td_peak'] <= 50)).sum())
    n_over = int((ne & (df['td_peak'] > 50) & (df['pr'] < 0.98)).sum())
    n_spike = int((ne & (df['pr'] >= 0.98)).sum())
    n_empty = int(df['is_empty'].sum())
    print(f"Grading counts: trustworthy={n_trust}, over-range={n_over}, "
          f"spike={n_spike}, empty={n_empty}  (total={n_trust+n_over+n_spike+n_empty})")

    # No-grading: all non-empty 8kHz
    allc = df[(df['sample_rate'] == 8000) & (~df['is_empty'])].copy()
    named_all = allc[allc.point_name.str.contains(NAMED_RE, case=False, na=False)]
    pts_all = sorted([p for p, c in named_all.point_name.value_counts().items()
                      if c >= MIN_SESSIONS])

    rows = []
    for p in pts_all:
        s = named_all[named_all.point_name == p].dropna(subset=['td_rms']).sort_values('t')
        n = len(s)
        y = s['td_rms'].values.astype(float)
        rho, pval = spearmanr(np.arange(n), y)
        rows.append({'point': p, 'n': n, 'rho': round(rho, 4), 'p_raw': pval})

    tbl_ng = pd.DataFrame(rows)
    _, p_bh, _, _ = multipletests(tbl_ng['p_raw'], alpha=0.05, method='fdr_bh')
    tbl_ng['p_adj_BH'] = p_bh
    tbl_ng['sig_BH'] = tbl_ng['p_adj_BH'] < 0.05
    tbl_ng = tbl_ng.sort_values('p_raw').reset_index(drop=True)

    print(f"\nNo-grading: {len(allc)} non-empty 8 kHz records "
          f"({len(named_all)} named-point records)\n")

    hdr = f"{'Point':42s} {'n':>3s} {'rho':>7s} {'p_raw':>11s} {'p_adj_BH':>11s} {'BH':>3s}"
    print(hdr)
    print('-' * len(hdr))
    for _, r in tbl_ng.iterrows():
        bh = 'Y' if r.sig_BH else 'N'
        print(f"{r.point:42s} {r.n:3.0f} {r.rho:+7.4f} {r.p_raw:11.4e} "
              f"{r.p_adj_BH:11.4e} {bh:>3s}")

    # Ranking comparison
    print('\n--- Ranking comparison (by |rho|, graded vs no-grading) ---')
    rank_g = tbl_graded.copy()
    rank_g['abs_rho'] = rank_g['rho'].abs()
    rank_g = rank_g.sort_values('abs_rho', ascending=False).reset_index(drop=True)
    rank_g['rank_graded'] = rank_g.index + 1

    rank_ng = tbl_ng.copy()
    rank_ng['abs_rho'] = rank_ng['rho'].abs()
    rank_ng = rank_ng.sort_values('abs_rho', ascending=False).reset_index(drop=True)
    rank_ng['rank_nograding'] = rank_ng.index + 1

    merged = rank_g[['point', 'rank_graded']].merge(
        rank_ng[['point', 'rank_nograding']], on='point', how='outer')
    merged = merged.sort_values('rank_graded')

    n_sig_g = int(tbl_graded['sig_BH'].sum())
    n_sig_ng = int(tbl_ng['sig_BH'].sum())

    hdr3 = f"{'Point':42s} {'Rank(graded)':>13s} {'Rank(no-grading)':>17s}"
    print(hdr3)
    print('-' * len(hdr3))
    for _, r in merged.iterrows():
        rg = f"{r.rank_graded:.0f}" if not pd.isna(r.rank_graded) else '-'
        rn = f"{r.rank_nograding:.0f}" if not pd.isna(r.rank_nograding) else '-'
        print(f"{r.point:42s} {rg:>13s} {rn:>17s}")

    print(f"\nBH-significant: graded={n_sig_g}/{len(tbl_graded)}, "
          f"no-grading={n_sig_ng}/{len(tbl_ng)}")
    print()


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='WS6: Trend-screening statistical robustness')
    parser.add_argument('--features', default='./features_all_public.parquet',
                        help='Path to feature table (parquet)')
    args = parser.parse_args()

    df = load(args.features)
    named = named_usable(df)
    pts = named_points(named)

    print(f"Feature table: {args.features}")
    print(f"Total records: {len(df)}, usable: {df['usable'].sum()}, "
          f"named usable: {len(named)}, named points (n>={MIN_SESSIONS}): {len(pts)}")
    print()

    # Section 1
    tbl = section1_fdr(named, pts)

    # Section 2
    sig_bh = tbl[tbl['sig_BH']]['point'].tolist()
    section2_loo(named, sig_bh)

    # Section 3
    section3_neff(named, sig_bh)

    # Section 4
    section4_ablation(df, named, pts, tbl)


if __name__ == '__main__':
    main()
