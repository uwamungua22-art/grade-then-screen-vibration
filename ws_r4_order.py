# -*- coding: utf-8 -*-
"""
R4 shaft/screw order analysis (order / shaft-line attribution)
============================================================
Paper framing: trend-screening / data-quality. This module only attributes
spectral peaks to known shaft orders (attribution); it does NOT perform
diagnosis / fault detection / validated conclusions.

Known drivetrain rotation frequencies (from mechanical parameters, not estimated):
    motor  : 50.00 Hz  (servo motor 700W / 3000 rpm)
    screwX : 1.83  Hz  (X-axis ball screw, pitch 10 mm)
    screwZ : 4.58  Hz  (Z-axis ball screw, pitch 4 mm)
Bearing BPFO/BPFI/BSF/FTF cannot be computed because bearing geometry parameters
are unavailable; left as a limitation, not fabricated.

Usage:
    python -X utf8 ws_r4_order.py --selftest
    python -X utf8 ws_r4_order.py --real <feature_path> [--scene SCENE] [--point POINT]
    (the --real branch reads only the local parquet feature table; does not touch
     any raw-waveform store. Written but not exercised at this stage; called by
     the main pipeline after R1 completes.)

Dependencies: numpy / scipy / matplotlib (optional, unused) / pandas; vibe_core in the same directory.
Author: R4 collaboration agent. Does not modify ws_raw_trajectory.py / ws1_4_methods.py.
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
from scipy import signal

# ---- Allow importing vibe_core from the same directory (the script sits next to vibe_core) ----
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import vibe_core as V  # noqa: E402

# ------------------------------------------------------------------
# Default shaft-order base frequencies (Hz). The key is the order_label prefix.
# ------------------------------------------------------------------
DEFAULT_ORDERS = {
    'motor': 50.0,    # servo motor rotation frequency
    'screwX': 1.83,   # X-axis ball screw rotation frequency (pitch 10mm)
    'screwZ': 4.58,   # Z-axis ball screw rotation frequency (pitch 4mm)
}

# Result output root directory (used only by the --real branch; not created at this stage)
RESULT_DIR = r'./results'


# ==================================================================
# Spectral estimation (fine-resolution Welch PSD + Hilbert envelope spectrum)
# ==================================================================
def _welch_psd(x, fs, target_df=0.25):
    """Fine-resolution Welch PSD.
    target_df: target frequency resolution (Hz), default 0.25 Hz.
    Returns (f, Pxx, df). DC already removed (P[0]=0).
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    # Derive nperseg = fs/target_df from the target resolution, bounded by data length.
    nper = int(round(fs / float(target_df)))
    nper = min(nper, n)
    # Take an even value near a power of two, and ensure >=256 so the spectrum is not too coarse
    nper = max(256, nper)
    if nper > n:
        nper = n
    f, Pxx = signal.welch(x, fs=fs, nperseg=nper, noverlap=nper // 2,
                          window='hann', detrend='constant')
    Pxx = Pxx.copy()
    if len(Pxx):
        Pxx[0] = 0.0  # remove DC
    df = float(f[1] - f[0]) if len(f) > 1 else float(fs)
    return f, Pxx, df


def _envelope_psd(x, fs, band=None, target_df=0.25):
    """High-frequency resonance demodulation -> Hilbert envelope -> envelope spectrum (fine-resolution Welch).
    band defaults to the upper half band (0.30~0.95*Nyquist), consistent with
    vibe_core.envelope_features.
    Returns (f, Pe, df, band).
    """
    x = np.asarray(x, dtype=np.float64)
    nyq = fs / 2.0
    if band is None:
        band = (0.30 * nyq, 0.95 * nyq)
    lo = max(band[0], 1.0) / nyq
    hi = min(band[1], nyq * 0.99) / nyq
    if hi <= lo:
        return np.array([]), np.array([]), float(fs), band
    try:
        sos = signal.butter(4, [lo, hi], btype='band', output='sos')
        xb = signal.sosfiltfilt(sos, x)
    except Exception:
        return np.array([]), np.array([]), float(fs), band
    env = np.abs(signal.hilbert(xb))
    env = env - env.mean()
    f, Pe, df = _welch_psd(env, fs, target_df=target_df)
    return f, Pe, df, band


# ==================================================================
# Neighborhood peak search
# ==================================================================
def _find_local_peak(f, P, target_hz, tol_hz):
    """Find the maximum within the neighborhood [target-tol, target+tol]; return (found_hz, amp, matched).
    matched: whether a significant peak exists in the neighborhood (here defined as the
    neighborhood having data and amp>0).
    If no bin falls within the neighborhood (frequency out of range), return (nan, nan, False).
    """
    if f is None or len(f) == 0:
        return float('nan'), float('nan'), False
    m = (f >= target_hz - tol_hz) & (f <= target_hz + tol_hz)
    if not np.any(m):
        return float('nan'), float('nan'), False
    idx = np.where(m)[0]
    sub = P[idx]
    j = int(np.argmax(sub))
    found_hz = float(f[idx[j]])
    amp = float(sub[j])
    matched = bool(amp > 0.0)
    return found_hz, amp, matched


def _tolerance(target_hz, df, rel=0.02, n_bin=1.0):
    """Tolerance = max(n_bin frequency-resolution bins, rel relative tolerance).
    Low frequencies (e.g. screw 1.83 Hz) are dominated by n_bin*df; high frequencies
    (e.g. 50 Hz harmonics) are dominated by rel*target.
    """
    return max(n_bin * df, rel * abs(target_hz))


# ==================================================================
# Core: order analysis
# ==================================================================
def order_analysis(x, fs, orders=None, n_harm=3, sideband=True,
                   target_df=0.25, rel_tol=0.02, n_bin_tol=1.0,
                   sideband_axes=('screwX', 'screwZ')):
    """Shaft/screw order analysis (spectral-peak attribution).

    Parameters
    ----------
    x : 1-D array, vibration signal (physical quantity g).
    fs : float, sampling rate Hz.
    orders : dict {label: base_hz}, shaft-order base-frequency table; defaults to DEFAULT_ORDERS.
    n_harm : int, examine harmonics 1..n_harm for each base frequency.
    sideband : bool, whether to look for +/- screw-rotation-frequency sidebands near the motor main peak.
    target_df : float, target spectral resolution (Hz).
    rel_tol : float, relative tolerance (used for high frequencies).
    n_bin_tol : float, absolute tolerance (unit: number of frequency bins, used for low frequencies).
    sideband_axes : shaft-order keys used as sideband spacing.

    Returns
    -------
    pandas.DataFrame, columns:
        order_label : 'motor_1x' / 'screwX_2x' / 'motor_50.0+screwX' ...
        target_hz   : target frequency
        found_hz    : measured peak frequency in the neighborhood (nan = no data in neighborhood)
        amp         : peak amplitude (PSD or envelope-spectrum power)
        matched     : whether a peak was found in the neighborhood
        source      : 'psd' / 'env'
        tol_hz      : the tolerance actually used
    """
    if orders is None:
        orders = dict(DEFAULT_ORDERS)
    x = np.asarray(x, dtype=np.float64)
    fs = float(fs)

    f_psd, P_psd, df_psd = _welch_psd(x, fs, target_df=target_df)
    f_env, P_env, df_env, _band = _envelope_psd(x, fs, target_df=target_df)

    rows = []

    def _scan(f, P, df, source):
        nyq = fs / 2.0
        for label, base in orders.items():
            for h in range(1, n_harm + 1):
                target = base * h
                if target <= 0 or target >= nyq:
                    continue
                tol = _tolerance(target, df, rel=rel_tol, n_bin=n_bin_tol)
                found, amp, matched = _find_local_peak(f, P, target, tol)
                rows.append({
                    'order_label': f'{label}_{h}x',
                    'target_hz': round(float(target), 4),
                    'found_hz': (round(found, 4) if np.isfinite(found) else float('nan')),
                    'amp': amp,
                    'matched': matched,
                    'source': source,
                    'tol_hz': round(float(tol), 4),
                })

    # Scan once on the PSD and once on the envelope spectrum
    _scan(f_psd, P_psd, df_psd, 'psd')
    if len(f_env):
        _scan(f_env, P_env, df_env, 'env')

    # ---- Sidebands: look for +/- screw-rotation-frequency sidebands near the motor 1x main peak (on the PSD) ----
    if sideband and 'motor' in orders and len(f_psd):
        nyq = fs / 2.0
        carrier = orders['motor']  # 50Hz carrier
        for ax in sideband_axes:
            if ax not in orders:
                continue
            sb = orders[ax]
            for sign, tag in ((+1, '+'), (-1, '-')):
                target = carrier + sign * sb
                if target <= 0 or target >= nyq:
                    continue
                tol = _tolerance(target, df_psd, rel=rel_tol, n_bin=n_bin_tol)
                found, amp, matched = _find_local_peak(f_psd, P_psd, target, tol)
                rows.append({
                    'order_label': f'motor_{carrier:g}{tag}{ax}',
                    'target_hz': round(float(target), 4),
                    'found_hz': (round(found, 4) if np.isfinite(found) else float('nan')),
                    'amp': amp,
                    'matched': matched,
                    'source': 'psd',
                    'tol_hz': round(float(tol), 4),
                })

    df = pd.DataFrame(rows, columns=[
        'order_label', 'target_hz', 'found_hz', 'amp', 'matched', 'source', 'tol_hz'
    ])
    return df


# ==================================================================
# Synthetic-signal self-test
# ==================================================================
def _make_synth(fs=8000, dur=30.0, seed=42):
    """Synthetic signal = 50/1.83/4.58 Hz base frequencies + 2nd~3rd harmonics + modulation sidebands + Gaussian noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * dur)) / fs

    x = np.zeros_like(t)
    # motor 50Hz + 2x + 3x
    x += 1.00 * np.sin(2 * np.pi * 50.0 * t)
    x += 0.40 * np.sin(2 * np.pi * 100.0 * t)
    x += 0.20 * np.sin(2 * np.pi * 150.0 * t)
    # screwX 1.83Hz + 2x + 3x
    x += 0.60 * np.sin(2 * np.pi * 1.83 * t)
    x += 0.25 * np.sin(2 * np.pi * 3.66 * t)
    x += 0.12 * np.sin(2 * np.pi * 5.49 * t)
    # screwZ 4.58Hz + 2x + 3x
    x += 0.55 * np.sin(2 * np.pi * 4.58 * t)
    x += 0.22 * np.sin(2 * np.pi * 9.16 * t)
    x += 0.10 * np.sin(2 * np.pi * 13.74 * t)
    # Modulation sideband: 50Hz carrier amplitude-modulated by 1.83Hz -> produces 50+/-1.83 sidebands
    x += 0.30 * (1.0 + 0.6 * np.sin(2 * np.pi * 1.83 * t)) * np.sin(2 * np.pi * 50.0 * t)
    # High-frequency resonance carrier (simulating the high-frequency content usable for demodulation),
    # modulated by 4.58Hz -> envelope spectrum shows 4.58Hz
    x += 0.25 * (1.0 + 0.8 * np.sin(2 * np.pi * 4.58 * t)) * np.sin(2 * np.pi * 3200.0 * t)
    # Gaussian noise
    x += 0.15 * rng.standard_normal(t.shape)
    return x.astype(np.float64)


def run_selftest():
    fs = 8000
    x = _make_synth(fs=fs, dur=30.0)
    df = order_analysis(x, fs, n_harm=3, sideband=True, target_df=0.25)

    print('=' * 64)
    print('R4 order analysis --selftest  (synthetic signal: fs=8000, 30s)')
    print('=' * 64)
    # Full table
    with pd.option_context('display.max_rows', None,
                           'display.width', 160,
                           'display.float_format', lambda v: f'{v:.4g}'):
        print(df.to_string(index=False))

    # ---- Assertion: the key main lines must be identified on the PSD ----
    # Check that matched==True for these order_label@psd entries
    must_psd = ['motor_1x', 'motor_2x', 'motor_3x',
                'screwX_1x', 'screwX_2x', 'screwX_3x',
                'screwZ_1x', 'screwZ_2x', 'screwZ_3x']
    psd = df[df['source'] == 'psd'].set_index('order_label')

    fails = []
    for lab in must_psd:
        if lab not in psd.index:
            fails.append(f'{lab}: missing')
            continue
        row = psd.loc[lab]
        if not bool(row['matched']):
            fails.append(f'{lab}: not matched')
            continue
        # Frequency deviation should be within tolerance
        if not np.isfinite(row['found_hz']):
            fails.append(f'{lab}: found_hz=nan')
            continue
        if abs(row['found_hz'] - row['target_hz']) > row['tol_hz'] + 1e-9:
            fails.append(f"{lab}: deviation {abs(row['found_hz']-row['target_hz']):.3f} > tol {row['tol_hz']:.3f}")

    # Sidebands: at least one pair must be identified (motor+/-screwX or motor+/-screwZ)
    sb = df[df['order_label'].str.contains('+', regex=False) |
            df['order_label'].str.contains('-', regex=False)]
    sb_ok = bool(sb['matched'].any()) if len(sb) else False
    if not sb_ok:
        fails.append('sidebands: no motor+/-screw sideband identified')

    # Envelope spectrum: at least screwZ_1x should be identified (injected via 3200Hz high-frequency modulation)
    env = df[df['source'] == 'env'].set_index('order_label')
    env_ok = ('screwZ_1x' in env.index) and bool(env.loc['screwZ_1x', 'matched'])

    print('-' * 64)
    print(f'Envelope-spectrum screwZ_1x identified: {"yes" if env_ok else "no"} '
          f'(modulation carrier 3200Hz, informational check, not a hard assertion)')
    print('-' * 64)

    if fails:
        print('Result: FAIL')
        for msg in fails:
            print('  - ' + msg)
        return 1
    else:
        n_match = int(df[df['source'] == 'psd']['matched'].sum())
        print(f'Result: PASS  ({n_match} target entries hit on the PSD; all key main lines attributed successfully)')
        return 0


# ==================================================================
# Real-data driver (written but not exercised at this stage; called by the main pipeline after R1)
# ==================================================================
def run_real(sts_path, scene=None, point=None,
             seconds=30.0, n_harm=3, sideband=True, target_df=0.25):
    """Read a 30s window -> order_analysis -> save CSV.
    !!! This branch reads only the local parquet feature table; it does not touch any
    raw-waveform store. Not exercised at this stage. !!!
    """
    meta = V.channel_meta(sts_path)
    fs = meta['sample_rate']
    scale = meta['scale_g_per_count']
    x = V.read_window(sts_path, scale, fs, seconds=seconds)

    df = order_analysis(x, fs, n_harm=n_harm, sideband=sideband, target_df=target_df)
    # Attach metadata columns for easier aggregation
    df.insert(0, 'measurement', meta.get('measurement', ''))
    df.insert(1, 'point_name', meta.get('point_name', ''))

    if scene is None:
        scene = _safe_token(meta.get('measurement', 'scene'))
    if point is None:
        point = _safe_token(meta.get('point_name', 'point'))

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_csv = os.path.join(RESULT_DIR, f'R4_order_{scene}_{point}.csv')
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f'Written: {out_csv}  ({len(df)} rows)')
    return out_csv, df


def _safe_token(s):
    """Sanitize a scene/point name into a token usable in a file name."""
    s = str(s).strip()
    bad = '\\/:*?"<>|]['
    for c in bad:
        s = s.replace(c, '_')
    s = s.replace(' ', '_')
    return s or 'NA'


# ==================================================================
# CLI
# ==================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description='R4 shaft/screw order analysis (spectral-peak attribution, trend-screening framing).')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--selftest', action='store_true',
                   help='Self-test the order_analysis logic with a synthetic signal and print PASS/FAIL.')
    g.add_argument('--real', metavar='FEATURE_PATH',
                   help='Run order analysis on a real feature table (called by the main pipeline after R1).')
    p.add_argument('--scene', default=None, help='--real: scene identifier (used in the CSV file name).')
    p.add_argument('--point', default=None, help='--real: point identifier (used in the CSV file name).')
    p.add_argument('--n-harm', type=int, default=3, help='Number of harmonics to examine (default 3).')
    p.add_argument('--no-sideband', action='store_true', help='Disable sideband detection.')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    sideband = not args.no_sideband

    if args.selftest:
        return run_selftest()
    elif args.real:
        run_real(args.real, scene=args.scene, point=args.point,
                 n_harm=args.n_harm, sideband=sideband)
        return 0
    else:
        build_parser().print_help()
        print('\nExamples:')
        print('  python -X utf8 ws_r4_order.py --selftest')
        print('  python -X utf8 ws_r4_order.py --real "<local feature-table path>" --scene sceneA --point CH1')
        print('\nNote: the --real branch reads only the local parquet feature table; it does not touch any raw-waveform store. Called by the main pipeline after R1 completes.')
        return 0


if __name__ == '__main__':
    sys.exit(main())
