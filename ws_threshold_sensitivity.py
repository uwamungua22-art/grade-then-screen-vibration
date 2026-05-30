# -*- coding: utf-8 -*-
"""
F5 data-quality grading threshold sensitivity analysis (JVET journal extended version)
==================================================
The swept threshold theta = "electrical-spike decision threshold" = the critical value of the
full-scale ratio pr (baseline 0.98 = 98% of full scale).

Four-level data-quality grading convention (all 888 single-channel feature records):
  - empty (empty/truncated)        : n_samples < 8000 (<1s). **Fixed at 144, not affected by theta.**
  - trustworthy (amplitude credible): non-empty & td_peak <= 50 (g). The boundary is set by the
                                      ENDEVCO sensor +/-50g physical range, **not the 98% threshold,
                                      and does not change with theta.**
  - spike (electrical spike)        : non-empty & pr >= theta. Physically impossible for a 50g sensor,
                                      judged an acquisition-chain artefact.
  - over-range                      : non-empty & td_peak > 50 & pr < theta.

==> theta only moves the boundary between over-range <-> spike;
    empty (144) and trustworthy (412) do not change with theta.

Note: td_peak / td_rms etc. are already in physical units g (already multiplied by scale_g_per_count).
      adc_fs = range_to * scale_g_per_count, the acquisition-chain full scale in g (typically ~101g).
      pr = td_peak / adc_fs.

Downstream stability (following the F3/WS3a convention of ws1_4_methods.py):
  candidate set = (sample_rate==8000) & (~is_empty), take the Top20 by largest td_kurtosis.
  This candidate set and these 20 Top20 records themselves do not change with theta;
  only "how many of the Top20 are judged hard_spike (pr>=theta)" changes with theta.
  The baseline conclusion is that spikes make up ~50% (10/20) of the Top20; F5 tests whether this is robust.

Environment: reads only the local parquet feature table; does not touch any raw-waveform store.
  Does not modify ws_raw_trajectory.py / ws1_4_methods.py.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = r'.'
FEAT = os.path.join(BASE, '00_inputs', 'features_all.parquet')
OUT = os.path.join(BASE, '04_jvet_results')
os.makedirs(OUT, exist_ok=True)

# sweep thresholds: baseline 0.98 plus denser sampling + 95~99% boundary
THETAS = [0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99]


def load():
    df = pd.read_parquet(FEAT)
    df['adc_fs'] = df['range_to'].astype(float) * df['scale_g_per_count']
    df['pr'] = df['td_peak'] / df['adc_fs']
    df['is_empty'] = df['n_samples'].fillna(0) < 8000
    return df


def classify_counts(df, theta):
    """Recompute the four-class counts at a given theta."""
    ne = ~df['is_empty']
    n_empty = int(df['is_empty'].sum())                       # fixed, does not change with theta
    n_trust = int((ne & (df['td_peak'] <= 50)).sum())         # set by the 50g physical range, does not change with theta
    n_spike = int((ne & (df['pr'] >= theta)).sum())
    n_over = int((ne & (df['td_peak'] > 50) & (df['pr'] < theta)).sum())
    return n_trust, n_over, n_spike, n_empty


def kurt_top20_spike_frac(df, theta):
    """F3/WS3a downstream: fraction of the kurtosis Top20 (8kHz non-empty records) judged electrical spikes (pr>=theta).
    The candidate set and these 20 Top20 records do not change with theta, only the decision label changes with theta."""
    cand = df[(df['sample_rate'] == 8000) & (~df['is_empty'])]
    top20 = cand.nlargest(20, 'td_kurtosis')
    n_spike_in_top = int((top20['pr'] >= theta).sum())
    return n_spike_in_top, n_spike_in_top / 20.0


def main():
    df = load()

    rows = []
    for th in THETAS:
        n_trust, n_over, n_spike, n_empty = classify_counts(df, th)
        n_in_top, frac = kurt_top20_spike_frac(df, th)
        rows.append({
            'theta': th,
            'n_trustworthy': n_trust,
            'n_overrange': n_over,
            'n_spike': n_spike,
            'n_empty': n_empty,
            'spike_frac_in_kurt_top20': round(frac, 4),
            'n_spike_in_kurt_top20': n_in_top,
        })
    res = pd.DataFrame(rows)

    # baseline 0.98 check
    base = res[np.isclose(res['theta'], 0.98)].iloc[0]
    print('=== theta=0.98 recomputation ===')
    print(f"trustworthy={base.n_trustworthy} over-range={base.n_overrange} "
          f"spike={base.n_spike} empty={base.n_empty} (expected 412/143/189/144)")
    expected = (412, 143, 189, 144)
    got = (int(base.n_trustworthy), int(base.n_overrange), int(base.n_spike), int(base.n_empty))
    print('reproduces expected counts:' , got == expected)

    # over-range<->spike migration magnitude (95% vs 99%)
    lo = res.iloc[0]
    hi = res.iloc[-1]
    print(f"\n=== theta {lo.theta:.3f} -> {hi.theta:.3f} migration ===")
    print(f"spike: {int(lo.n_spike)} -> {int(hi.n_spike)} (Delta={int(hi.n_spike-lo.n_spike)})")
    print(f"over : {int(lo.n_overrange)} -> {int(hi.n_overrange)} (Delta={int(hi.n_overrange-lo.n_overrange)})")
    print(f"trustworthy constant throughout: {res['n_trustworthy'].nunique()==1} (={int(res.n_trustworthy.iloc[0])})")
    print(f"empty constant throughout: {res['n_empty'].nunique()==1} (={int(res.n_empty.iloc[0])})")
    print(f"\n=== kurtosis Top20 spike fraction ===")
    print(res[['theta', 'spike_frac_in_kurt_top20', 'n_spike_in_kurt_top20']].to_string(index=False))

    csv_path = os.path.join(OUT, 'F5_threshold_sensitivity.csv')
    res.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print('\nCSV ->', csv_path)

    # ---- plotting ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) four-class counts vs theta
    ax = axes[0]
    ax.plot(res['theta'], res['n_trustworthy'], 'o-', label='trustworthy (fixed)', color='#2ca02c')
    ax.plot(res['theta'], res['n_overrange'], 's-', label='over-range', color='#ff7f0e')
    ax.plot(res['theta'], res['n_spike'], '^-', label='electrical spike', color='#d62728')
    ax.plot(res['theta'], res['n_empty'], 'd--', label='empty/truncated (fixed)', color='#7f7f7f')
    ax.axvline(0.98, color='k', ls=':', lw=1, alpha=0.6)
    ax.text(0.98, ax.get_ylim()[1]*0.5, ' baseline theta=0.98', rotation=90, va='center', fontsize=9)
    ax.set_xlabel('electrical-spike decision threshold theta (full-scale ratio)')
    ax.set_ylabel('number of records')
    ax.set_title('(a) four-class quality-grading counts vs threshold theta')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (b) kurtosis Top20 spike fraction vs theta
    ax = axes[1]
    ax.plot(res['theta'], res['spike_frac_in_kurt_top20']*100, 'o-', color='#d62728')
    for _, r in res.iterrows():
        ax.annotate(f"{int(r.n_spike_in_kurt_top20)}/20",
                    (r.theta, r.spike_frac_in_kurt_top20*100),
                    textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8)
    ax.axhline(50, color='gray', ls='--', lw=1, alpha=0.7)
    ax.axvline(0.98, color='k', ls=':', lw=1, alpha=0.6)
    ax.set_ylim(0, 100)
    ax.set_xlabel('electrical-spike decision threshold theta (full-scale ratio)')
    ax.set_ylabel('electrical-spike fraction in kurtosis Top20 (%)')
    ax.set_title('(b) 8kHz non-empty kurtosis Top20 spike fraction vs threshold theta')
    ax.grid(alpha=0.3)

    fig.suptitle('F5 data-quality grading threshold sensitivity analysis', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = os.path.join(OUT, 'F5_threshold_sensitivity.png')
    fig.savefig(png_path, dpi=180, bbox_inches='tight')
    print('PNG ->', png_path)


if __name__ == '__main__':
    main()
