# -*- coding: utf-8 -*-
"""R-class analysis (re-reads the raw waveforms, run after the external drive is connected).
R1 implemented: full-record sliding-window RMS/kurtosis trajectories for key channels
(within-record time variation, source of the largest increments).
R2/R3/R4 are stubs (spec in each docstring), to be completed.
Depends on vibe_core (same directory). Interpreter: python -X utf8
Usage:
  python ws_raw_trajectory.py --check         # only verify the raw store is readable + enumerate waveform files
  python ws_raw_trajectory.py --r1            # run R1 sliding-window trajectories
"""
import sys, io, os, re, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd

import vibe_core as V

# ---- Config: raw store root directory (verify the path after connecting the drive) ----
RAW_ROOT = r'./data/raw'
OUT = r'./results'

# R1 key channels (matched by point_name substring; reuses the channels from the conference version)
TARGET_SUBSTR = ['reversing', 'commutator', 'ball', 'screw']  # CH6/CH19/CH16/CH10
WIN_S = 1.0      # sliding-window length (seconds)
HOP_S = 1.0      # hop (seconds); equal to WIN_S means no overlap


def ts(m):
    g = re.search(r'(\d\d)-(\d\d)-(\d\d) (\d\d)-(\d\d)-(\d\d)', m)
    return pd.Timestamp(2000 + int(g[1]), int(g[2]), int(g[3]),
                        int(g[4]), int(g[5]), int(g[6])) if g else pd.NaT


def check_drive():
    if not os.path.isdir(RAW_ROOT):
        print(f'[FAIL] raw store root directory not found: {RAW_ROOT}\n  -> edit RAW_ROOT at the top of this file to the correct path')
        return []
    stss = V.find_all_sts(RAW_ROOT)
    print(f'[OK] enumerated {len(stss)} raw waveform files')
    for p in stss[:3]:
        print('   ', p)
    return stss


def sliding_trajectory(sts_path, scale, fs, win_s=WIN_S, hop_s=HOP_S, max_windows=None):
    """Compute full-record sliding-window RMS/kurtosis trajectory for a single record.
    Sequential read, constant memory."""
    total = os.path.getsize(sts_path) // 4
    win = int(win_s * fs)
    hop = int(hop_s * fs)
    if win < 16 or total < win:
        return np.empty((0, 3))
    rows = []
    off = 0
    while off + win <= total:
        x = np.fromfile(sts_path, dtype='<i4', count=win, offset=off * 4).astype(np.float64) * scale
        m = x.mean()
        s = x.std()
        rms = np.sqrt(np.mean(x * x))
        kurt = np.mean((x - m) ** 4) / s ** 4 if s > 0 else 0.0
        rows.append((off / fs, rms, kurt))
        off += hop
        if max_windows and len(rows) >= max_windows:
            break
    return np.array(rows)  # columns: [t_sec, rms, kurtosis]


def run_r1():
    os.makedirs(OUT, exist_ok=True)
    stss = check_drive()
    if not stss:
        return
    # Build index: meta for each record (use 8 kHz, optional empty/spike filtering applied later)
    recs = []
    for p in stss:
        meta = V.channel_meta(p)
        if meta['sample_rate'] != 8000:
            continue
        if not any(sub in meta['point_name'] for sub in TARGET_SUBSTR):
            continue
        recs.append(meta)
    print(f'records matching key channels: {len(recs)}')
    summary = []
    for meta in recs:
        traj = sliding_trajectory(meta['sts_path'], meta['scale_g_per_count'], meta['sample_rate'])
        if traj.size == 0:
            continue
        sess = meta['measurement']
        pt = meta['point_name']
        # Store per-window trajectory
        fn = f"{sess}__{pt}".replace(' ', '_').replace('/', '_').replace('\\', '_')[:120]
        np.savetxt(os.path.join(OUT, f'R1_traj__{fn}.csv'), traj,
                   delimiter=',', header='t_sec,rms,kurtosis', comments='', fmt='%.6f')
        # Session-level summary: window-level distribution rather than a single point
        # (paper point: reveals within-record non-stationarity)
        rms_w, kurt_w = traj[:, 1], traj[:, 2]
        summary.append({
            'measurement': sess, 'point_name': pt, 't': ts(sess),
            'n_windows': len(traj),
            'rms_mean': rms_w.mean(), 'rms_std': rms_w.std(),
            'rms_p05': np.percentile(rms_w, 5), 'rms_p50': np.percentile(rms_w, 50),
            'rms_p95': np.percentile(rms_w, 95), 'rms_max': rms_w.max(),
            'kurt_mean': kurt_w.mean(), 'kurt_p95': np.percentile(kurt_w, 95),
            'kurt_max': kurt_w.max(),
        })
    df = pd.DataFrame(summary).sort_values(['point_name', 't'])
    df.to_csv(os.path.join(OUT, 'R1_window_level_summary.csv'), index=False, encoding='utf-8-sig')
    print(f'[done] R1 trajectories and window-level summary -> {OUT}')
    print(df.to_string(index=False))


# ---- R2/R3/R4 stubs (spec, to be completed) ----
def run_r2():
    """R2 cross-sensor spatial consistency (alternative validation):
    take neighboring measurement points on the same component (e.g. multiple points on the
    reversing/commutator unit / same-side ball-screw upper-mid-lower), compute the pairwise
    correlation of their cross-session RMS trends; genuine degradation should be co-directional
    across multiple points (high correlation), whereas a single-channel artifact is isolated.
    Input: can directly use the local features_all.parquet td_rms (no need to re-read the raw!),
    so R2 can in fact be done locally first -- see load() in ws1_4_methods. Output:
    point-by-point correlation heatmap + synchronization metric."""
    raise NotImplementedError('R2 to be done; hint: can use the local parquet first, see docstring')


def run_r3():
    """R3 degradation trajectory modeling + change-point detection:
    fit linear/exponential to the cross-session RMS series (ordinal axis) for CH6/CH19,
    give bootstrap CI; detect degradation knees with ruptures/PELT or a simple CUSUM.
    Emphasize ordinal axis, not calibrated time.
    Can use the local parquet, no need to re-read the raw."""
    raise NotImplementedError('R3 to be done')


def run_r4():
    """R4 shaft/screw order analysis (needs the raw waveforms):
    recompute fine-resolution Welch + envelope spectrum for key channels at key sessions,
    compare against 50 Hz (motor), 1.83/4.58 Hz (screw rotation) and their harmonics/sidebands;
    matches are labeled as shaft-order sources, bearing BPFO is left as a limitation due to
    missing explicit geometry.
    Use vibe_core.read_window + frequency_features/envelope_features (tunable band)."""
    raise NotImplementedError('R4 to be done')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--r1', action='store_true')
    a = ap.parse_args()
    if a.check:
        check_drive()
    elif a.r1:
        run_r1()
    else:
        print(__doc__)
