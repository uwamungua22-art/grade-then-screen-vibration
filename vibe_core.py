# -*- coding: utf-8 -*-
"""
Vibration acceleration data — core decoding and feature library
数据格式: little-endian int32, 无表头, 样本数 = 文件大小/4
标定(经确认: 不除增益): g = raw * (Voltage_to / Range_to) / Sensitivity
设计要点:
  - 时域统计用分块累加矩(内存恒定, 应对超长记录)
  - 频谱/包络用限长分析窗(中段, 默认20s)
"""
import os, re, glob
import numpy as np
from scipy import signal, stats

SR_DEFAULT = 8000
ANALYSIS_SECONDS = 30.0          # 频谱/包络分析窗长度
CHUNK_SAMPLES = 4_000_000        # 时域分块大小(int32, 约16MB/块)


def _nperseg(fs, n):
    """自适应谱段长: 目标约1Hz分辨率, 介于2048~32768, 不超过数据长度。"""
    target = int(fs)
    target = max(2048, min(target, 32768))
    return min(target, n)


# ---------------- .par 参数解析 ----------------
def parse_par(par_path):
    """解析 .par(INI风格), 容错中文(GBK)。返回 dict。"""
    d = {}
    with open(par_path, 'rb') as f:
        raw = f.read()
    text = raw.decode('gbk', errors='replace')
    for line in text.splitlines():
        if '=' in line and not line.strip().startswith('['):
            k, _, v = line.partition('=')
            d[k.strip()] = v.strip()
    return d


def get_scale(par):
    """物理量换算系数: g = raw_int * scale。不除增益。"""
    try:
        v_to = float(par.get('Voltage to', 10000.0))      # mV
        r_to = float(par.get('Range to', 8388607.0))      # ADC counts
        sens = float(par.get('Sensitivity', 98.6))        # mV/g
    except ValueError:
        v_to, r_to, sens = 10000.0, 8388607.0, 98.6
    if sens == 0:
        sens = 1.0
    mv_per_count = v_to / r_to
    return mv_per_count / sens   # g per count


def channel_meta(sts_path):
    """从 .sts 路径推导通道元信息(读取同名 .par)。"""
    base = sts_path[:-4]  # 去掉 .sts
    par_path = base + '.par'
    fname = os.path.basename(base)              # 例: 'CH 1-右门外导轨右-z轴]'
    meas = os.path.basename(os.path.dirname(sts_path))
    par = parse_par(par_path) if os.path.exists(par_path) else {}
    fsize = os.path.getsize(sts_path)
    nsamp = fsize // 4
    fs = int(float(par.get('Sample rate', SR_DEFAULT) or SR_DEFAULT))
    if fs <= 0:
        fs = SR_DEFAULT
    return {
        'measurement': meas,
        'file': fname,
        'point_name': fname.rstrip(']').strip(),
        'sts_path': sts_path,
        'sample_rate': fs,
        'ad_bit': par.get('AD Bit', ''),
        'unit': par.get('Unit', ''),
        'physical': par.get('Physical', ''),
        'sensitivity_mv_g': par.get('Sensitivity', ''),
        'voltage_to_mv': par.get('Voltage to', ''),
        'range_to': par.get('Range to', ''),
        'gain': par.get('Gain', ''),
        'transducer': par.get('Transducer', ''),
        'file_bytes': fsize,
        'n_samples': nsamp,
        'duration_s': nsamp / fs if fs else 0.0,
        'scale_g_per_count': get_scale(par),
    }


# ---------------- 数据读取 ----------------
def read_window(sts_path, scale, fs, seconds=ANALYSIS_SECONDS):
    """读取中段分析窗, 返回物理量(g, float64)。"""
    nsamp = os.path.getsize(sts_path) // 4
    win = min(int(seconds * fs), nsamp)
    start = max(0, (nsamp - win) // 2)
    x = np.fromfile(sts_path, dtype='<i4', count=win, offset=start * 4)
    return x.astype(np.float64) * scale


# ---------------- 时域特征(分块累加, 内存恒定) ----------------
def time_domain_features(sts_path, scale):
    """流式读取全量数据, 累加矩, 计算时域特征(物理量 g)。"""
    n = 0
    s1 = s2 = s3 = s4 = 0.0
    s_abs = 0.0
    s_sqrt_abs = 0.0
    vmax, vmin = -np.inf, np.inf
    offset = 0
    fsize = os.path.getsize(sts_path)
    total = fsize // 4
    while offset < total:
        cnt = min(CHUNK_SAMPLES, total - offset)
        chunk = np.fromfile(sts_path, dtype='<i4', count=cnt, offset=offset * 4).astype(np.float64) * scale
        offset += cnt
        n += chunk.size
        s1 += chunk.sum()
        s2 += np.square(chunk).sum()
        s3 += (chunk ** 3).sum()
        s4 += (chunk ** 4).sum()
        a = np.abs(chunk)
        s_abs += a.sum()
        s_sqrt_abs += np.sqrt(a).sum()
        vmax = max(vmax, chunk.max())
        vmin = min(vmin, chunk.min())
    if n == 0:
        return {}
    mean = s1 / n
    m2 = s2 / n - mean ** 2            # 方差
    rms = np.sqrt(s2 / n)
    std = np.sqrt(max(m2, 0.0))
    abs_mean = s_abs / n
    sqrt_abs_mean = s_sqrt_abs / n
    # 中心矩 -> 偏度/峭度
    m3 = s3 / n - 3 * mean * (s2 / n) + 2 * mean ** 3
    m4 = s4 / n - 4 * mean * (s3 / n) + 6 * mean ** 2 * (s2 / n) - 3 * mean ** 4
    skew = m3 / std ** 3 if std > 0 else 0.0
    kurt = m4 / std ** 4 if std > 0 else 0.0          # 标准峭度(高斯≈3)
    peak = max(abs(vmax), abs(vmin))
    p2p = vmax - vmin
    return {
        'td_n': n,
        'td_mean': mean,
        'td_std': std,
        'td_var': m2,
        'td_rms': rms,
        'td_max': vmax,
        'td_min': vmin,
        'td_peak': peak,
        'td_p2p': p2p,
        'td_abs_mean': abs_mean,
        'td_skewness': skew,
        'td_kurtosis': kurt,
        'td_crest_factor': peak / rms if rms > 0 else 0.0,
        'td_shape_factor': rms / abs_mean if abs_mean > 0 else 0.0,
        'td_impulse_factor': peak / abs_mean if abs_mean > 0 else 0.0,
        'td_clearance_factor': peak / (sqrt_abs_mean ** 2) if sqrt_abs_mean > 0 else 0.0,
    }


# ---------------- 频域特征(Welch PSD) ----------------
def frequency_features(x, fs, n_bands=8):
    nper = _nperseg(fs, len(x))
    if nper < 256:
        return {}
    f, Pxx = signal.welch(x, fs=fs, nperseg=nper, noverlap=nper // 2)
    df = f[1] - f[0]
    P = Pxx.copy()
    P[0] = 0.0                       # 去直流
    total = P.sum()
    if total <= 0:
        return {'fd_total_power': 0.0}
    dom_i = int(np.argmax(P))
    centroid = float((f * P).sum() / total)
    rms_freq = float(np.sqrt((f ** 2 * P).sum() / total))
    freq_std = float(np.sqrt(((f - centroid) ** 2 * P).sum() / total))
    feats = {
        'fd_total_power': float(total * df),
        'fd_dominant_freq': float(f[dom_i]),
        'fd_dominant_amp': float(P[dom_i]),
        'fd_spectral_centroid': centroid,
        'fd_rms_freq': rms_freq,
        'fd_freq_std': freq_std,
    }
    # 频带能量占比(等分到 Nyquist)
    edges = np.linspace(0, fs / 2, n_bands + 1)
    for b in range(n_bands):
        m = (f >= edges[b]) & (f < edges[b + 1])
        feats[f'fd_band{b+1}_ratio'] = float(P[m].sum() / total)
    return feats


# ---------------- 包络谱特征(Hilbert, 故障诊断) ----------------
def envelope_features(x, fs, band=None, top_k=5, mod_fmax=None):
    """带通(高频共振解调)->Hilbert包络->包络谱; 提取调制/故障频段主峰。
    band 缺省按 fs 自适应取上半频段(0.3~0.95*Nyquist); mod_fmax 缺省 0.05*fs。"""
    nyq = fs / 2.0
    if band is None:
        band = (0.30 * nyq, 0.95 * nyq)
    if mod_fmax is None:
        mod_fmax = max(200.0, 0.05 * fs)
    lo = max(band[0], 1.0) / nyq
    hi = min(band[1], nyq * 0.99) / nyq
    if hi <= lo:
        return {}
    try:
        sos = signal.butter(4, [lo, hi], btype='band', output='sos')
        xb = signal.sosfiltfilt(sos, x)
    except Exception:
        return {}
    env = np.abs(signal.hilbert(xb))
    env = env - env.mean()
    nper = _nperseg(fs, len(env))
    if nper < 256:
        return {}
    f, Pe = signal.welch(env, fs=fs, nperseg=nper, noverlap=nper // 2)
    band_m = (f > 0) & (f <= mod_fmax)
    fb, Pb = f[band_m], Pe[band_m]
    if Pb.size == 0 or Pb.sum() <= 0:
        return {}
    order = np.argsort(Pb)[::-1]
    feats = {'env_band_power': float(Pb.sum()),
             'env_demod_lo_hz': float(band[0]), 'env_demod_hi_hz': float(band[1])}
    for i in range(top_k):
        if i < len(order):
            feats[f'env_peak{i+1}_freq'] = float(fb[order[i]])
            feats[f'env_peak{i+1}_amp'] = float(Pb[order[i]])
        else:
            feats[f'env_peak{i+1}_freq'] = 0.0
            feats[f'env_peak{i+1}_amp'] = 0.0
    feats['env_centroid'] = float((fb * Pb).sum() / Pb.sum())
    return feats


# ---------------- 单通道完整处理 ----------------
def process_channel(sts_path):
    meta = channel_meta(sts_path)
    scale = meta['scale_g_per_count']
    fs = meta['sample_rate']
    out = dict(meta)
    try:
        out.update(time_domain_features(sts_path, scale))
        x = read_window(sts_path, scale, fs)
        out.update(frequency_features(x, fs))
        if str(meta.get('physical', '')).lower().startswith('acc'):
            out.update(envelope_features(x, fs))
        out['status'] = 'ok'
    except Exception as e:
        out['status'] = 'error: ' + repr(e)
    return out


def find_all_sts(root):
    return sorted(glob.glob(os.path.join(root, '**', '*.sts'), recursive=True))


if __name__ == '__main__':
    import sys, json
    r = process_channel(sys.argv[1])
    print(json.dumps({k: (v if not isinstance(v, float) else round(v, 6)) for k, v in r.items()},
                     ensure_ascii=False, indent=2))
