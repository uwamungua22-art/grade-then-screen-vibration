# -*- coding: utf-8 -*-
"""WS5: fills statistical gaps raised in review -- autocorrelation + effect-size 95% CI + Sen slope CI.
Reuses the load()/usable filtering and sorting conventions of ws1_4_methods; reads only the local
parquet feature table, does not touch any raw-waveform store."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kendalltau, norm, theilslopes

import ws1_4_methods as W

TARGETS = ['CH6 reversing/commutator unit', 'CH19 ball screw R-low',
           'CH16 lift screw', 'CH10 screw R-mid']


def fisher_ci(stat, n, spearman=False, alpha=0.05):
    """Fisher z-transform CI for Pearson/Spearman r."""
    if abs(stat) >= 1 or n < 5:
        return (np.nan, np.nan)
    z = np.arctanh(stat)
    se = (1.03 / np.sqrt(n - 3)) if spearman else (1.0 / np.sqrt(n - 3))
    zc = norm.ppf(1 - alpha / 2)
    return np.tanh(z - zc * se), np.tanh(z + zc * se)


def kendall_ci(tau, n, alpha=0.05):
    """Normal-approx CI for Kendall tau via its standard error."""
    if n < 5:
        return (np.nan, np.nan)
    se = np.sqrt(2 * (2 * n + 5) / (9.0 * n * (n - 1)))
    zc = norm.ppf(1 - alpha / 2)
    return tau - zc * se, tau + zc * se


def lag1_autocorr(x):
    x = np.asarray(x, float)
    x = x - x.mean()
    if len(x) < 3 or np.allclose(x, 0):
        return np.nan
    return np.sum(x[:-1] * x[1:]) / np.sum(x * x)


def durbin_watson(resid):
    resid = np.asarray(resid, float)
    return np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2)


def main():
    df = W.load()
    u = df[df['usable']].copy()
    named = u[u.point_name.str.contains('reversing|commutator|ball|screw|guide|rail|seat|conn|lift', case=False, na=False)]

    print('point | n | rho[95%CI] | tau[95%CI] | r[95%CI] | Sen[95%CI] | '
          'lag1_raw | lag1_resid(Sen) | DW_resid')
    print('-' * 120)
    for p in TARGETS:
        s = named[named.point_name == p].dropna(subset=['td_rms']).sort_values('t')
        n = len(s)
        idx = np.arange(n)
        y = s.td_rms.values.astype(float)

        rho, _ = spearmanr(idx, y)
        tau, _ = kendalltau(idx, y)
        r, _ = pearsonr(idx, y)
        sen, b, sen_lo, sen_hi = theilslopes(y, idx, 0.95)

        rho_ci = fisher_ci(rho, n, spearman=True)
        tau_ci = kendall_ci(tau, n)
        r_ci = fisher_ci(r, n, spearman=False)

        resid = y - (sen * idx + b)
        l1_raw = lag1_autocorr(y)
        l1_res = lag1_autocorr(resid)
        dw = durbin_watson(resid)

        print(f'{p[:14]:14s} | n={n:2d} | '
              f'rho={rho:.3f}[{rho_ci[0]:.2f},{rho_ci[1]:.2f}] | '
              f'tau={tau:.3f}[{tau_ci[0]:.2f},{tau_ci[1]:.2f}] | '
              f'r={r:.3f}[{r_ci[0]:.2f},{r_ci[1]:.2f}] | '
              f'Sen={sen:.4f}[{sen_lo:.4f},{sen_hi:.4f}] | '
              f'l1raw={l1_raw:.3f} | l1resid={l1_res:.3f} | DW={dw:.2f}')


if __name__ == '__main__':
    main()
