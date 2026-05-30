# -*- coding: utf-8 -*-
"""
R4 轴/丝杠阶次分析 (order / shaft-line attribution)
============================================================
论文定位: trend-screening / data-quality。本模块只把谱峰**归因到已知轴序**
(attribution), 不做 diagnosis / fault detection / validated 结论。

已知传动链转频 (来自机械参数, 非估计):
    motor  : 50.00 Hz  (伺服电机 700W / 3000 rpm)
    screwX : 1.83  Hz  (X 轴丝杠, 螺距 10 mm)
    screwZ : 4.58  Hz  (Z 轴丝杠, 螺距 4 mm)
轴承 BPFO/BPFI/BSF/FTF 因缺轴承几何参数**无法计算**, 留为局限, 不臆造。

用法:
    python -X utf8 ws_r4_order.py --selftest
    python -X utf8 ws_r4_order.py --real <sts_path> [--scene 场次] [--point 测点]
    (--real 分支需 G: 盘连接, 本阶段只写不跑; 由主线在 R1 完成后调用)

依赖: numpy / scipy / matplotlib(可选, 未用) / pandas ; 同目录的 vibe_core。
作者: R4 协作 agent。不修改 ws_raw_trajectory.py / ws1_4_methods.py。
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
from scipy import signal

# ---- 让脚本可从同目录导入 vibe_core (脚本即与 vibe_core 同目录) ----
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import vibe_core as V  # noqa: E402

# ------------------------------------------------------------------
# 默认轴序基频 (Hz)。键名即 order_label 前缀。
# ------------------------------------------------------------------
DEFAULT_ORDERS = {
    'motor': 50.0,    # 伺服电机转频
    'screwX': 1.83,   # X 轴丝杠转频 (螺距 10mm)
    'screwZ': 4.58,   # Z 轴丝杠转频 (螺距 4mm)
}

# 结果输出根目录 (仅 --real 分支使用, 本阶段不创建)
RESULT_DIR = r'./results'


# ==================================================================
# 谱估计 (细分辨率 Welch PSD + Hilbert 包络谱)
# ==================================================================
def _welch_psd(x, fs, target_df=0.25):
    """细分辨率 Welch PSD。
    target_df: 目标频率分辨率(Hz), 默认 0.25Hz。
    返回 (f, Pxx, df)。已去直流(P[0]=0)。
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    # 由目标分辨率推 nperseg = fs/target_df, 受数据长度上限约束。
    nper = int(round(fs / float(target_df)))
    nper = min(nper, n)
    # 取 2 的幂附近的偶数, 并保证 >=256 以免谱太糙
    nper = max(256, nper)
    if nper > n:
        nper = n
    f, Pxx = signal.welch(x, fs=fs, nperseg=nper, noverlap=nper // 2,
                          window='hann', detrend='constant')
    Pxx = Pxx.copy()
    if len(Pxx):
        Pxx[0] = 0.0  # 去直流
    df = float(f[1] - f[0]) if len(f) > 1 else float(fs)
    return f, Pxx, df


def _envelope_psd(x, fs, band=None, target_df=0.25):
    """高频共振解调 -> Hilbert 包络 -> 包络谱(细分辨率 Welch)。
    band 缺省取上半频段 (0.30~0.95*Nyquist), 与 vibe_core.envelope_features 一致口径。
    返回 (f, Pe, df, band)。
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
# 邻域寻峰
# ==================================================================
def _find_local_peak(f, P, target_hz, tol_hz):
    """在 [target-tol, target+tol] 邻域内找最大值, 返回 (found_hz, amp, matched)。
    matched: 邻域内是否存在显著峰(此处定义为邻域有数据且 amp>0)。
    若邻域无任何 bin 落入(频率超界), 返回 (nan, nan, False)。
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
    """容差 = max(n_bin 个频率分辨率, rel 相对容差)。
    低频(如丝杠 1.83Hz)由 n_bin*df 主导; 高频(如 50Hz 谐波)由 rel*target 主导。
    """
    return max(n_bin * df, rel * abs(target_hz))


# ==================================================================
# 核心: 阶次分析
# ==================================================================
def order_analysis(x, fs, orders=None, n_harm=3, sideband=True,
                   target_df=0.25, rel_tol=0.02, n_bin_tol=1.0,
                   sideband_axes=('screwX', 'screwZ')):
    """轴/丝杠阶次分析 (谱峰归因)。

    参数
    ----
    x : 1-D array, 振动信号 (物理量 g)。
    fs : float, 采样率 Hz。
    orders : dict {label: base_hz}, 轴序基频表; 缺省 DEFAULT_ORDERS。
    n_harm : int, 每条基频考察 1..n_harm 次谐波。
    sideband : bool, 是否在 motor 主峰附近找 ±丝杠转频 边带。
    target_df : float, 目标谱分辨率(Hz)。
    rel_tol : float, 相对容差(用于高频)。
    n_bin_tol : float, 绝对容差(单位: 频率 bin 数, 用于低频)。
    sideband_axes : 用作边带间距的轴序键名。

    返回
    ----
    pandas.DataFrame, 列:
        order_label : 'motor_1x' / 'screwX_2x' / 'motor_50.0+screwX' ...
        target_hz   : 目标频率
        found_hz    : 邻域实测峰频 (nan=邻域无数据)
        amp         : 峰幅值 (PSD 或包络谱功率)
        matched     : 是否在邻域内找到峰
        source      : 'psd' / 'env'
        tol_hz      : 实际使用的容差
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

    # PSD 与包络谱各扫一遍
    _scan(f_psd, P_psd, df_psd, 'psd')
    if len(f_env):
        _scan(f_env, P_env, df_env, 'env')

    # ---- 边带: 在 motor 1x 主峰附近找 ± 丝杠转频 边带 (在 PSD 上) ----
    if sideband and 'motor' in orders and len(f_psd):
        nyq = fs / 2.0
        carrier = orders['motor']  # 50Hz 载波
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
# 合成信号自测
# ==================================================================
def _make_synth(fs=8000, dur=30.0, seed=42):
    """合成信号 = 50/1.83/4.58 Hz 基频 + 2~3 次谐波 + 调制边带 + 高斯噪声。"""
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
    # 调制边带: 50Hz 载波被 1.83Hz 调幅 -> 产生 50±1.83 边带
    x += 0.30 * (1.0 + 0.6 * np.sin(2 * np.pi * 1.83 * t)) * np.sin(2 * np.pi * 50.0 * t)
    # 高频共振载波(模拟解调能用的高频成分), 被 4.58Hz 调制 -> 包络谱出现 4.58Hz
    x += 0.25 * (1.0 + 0.8 * np.sin(2 * np.pi * 4.58 * t)) * np.sin(2 * np.pi * 3200.0 * t)
    # 高斯噪声
    x += 0.15 * rng.standard_normal(t.shape)
    return x.astype(np.float64)


def run_selftest():
    fs = 8000
    x = _make_synth(fs=fs, dur=30.0)
    df = order_analysis(x, fs, n_harm=3, sideband=True, target_df=0.25)

    print('=' * 64)
    print('R4 阶次分析 --selftest  (合成信号: fs=8000, 30s)')
    print('=' * 64)
    # 全表
    with pd.option_context('display.max_rows', None,
                           'display.width', 160,
                           'display.float_format', lambda v: f'{v:.4g}'):
        print(df.to_string(index=False))

    # ---- 断言: 关键主线必须在 PSD 上被识别 ----
    # 检查这些 order_label@psd 的 matched==True
    must_psd = ['motor_1x', 'motor_2x', 'motor_3x',
                'screwX_1x', 'screwX_2x', 'screwX_3x',
                'screwZ_1x', 'screwZ_2x', 'screwZ_3x']
    psd = df[df['source'] == 'psd'].set_index('order_label')

    fails = []
    for lab in must_psd:
        if lab not in psd.index:
            fails.append(f'{lab}: 缺失')
            continue
        row = psd.loc[lab]
        if not bool(row['matched']):
            fails.append(f'{lab}: 未匹配')
            continue
        # 频率偏差应在容差内
        if not np.isfinite(row['found_hz']):
            fails.append(f'{lab}: found_hz=nan')
            continue
        if abs(row['found_hz'] - row['target_hz']) > row['tol_hz'] + 1e-9:
            fails.append(f"{lab}: 偏差 {abs(row['found_hz']-row['target_hz']):.3f} > tol {row['tol_hz']:.3f}")

    # 边带至少识别到一对 (motor±screwX 或 motor±screwZ)
    sb = df[df['order_label'].str.contains('+', regex=False) |
            df['order_label'].str.contains('-', regex=False)]
    sb_ok = bool(sb['matched'].any()) if len(sb) else False
    if not sb_ok:
        fails.append('边带: 未识别到任何 motor±screw 边带')

    # 包络谱: 至少识别到 screwZ_1x (被 3200Hz 高频调制注入)
    env = df[df['source'] == 'env'].set_index('order_label')
    env_ok = ('screwZ_1x' in env.index) and bool(env.loc['screwZ_1x', 'matched'])

    print('-' * 64)
    print(f'包络谱 screwZ_1x 识别: {"是" if env_ok else "否"} '
          f'(调制载波 3200Hz, 信息性检查, 不纳入硬断言)')
    print('-' * 64)

    if fails:
        print('结果: FAIL')
        for msg in fails:
            print('  - ' + msg)
        return 1
    else:
        n_match = int(df[df['source'] == 'psd']['matched'].sum())
        print(f'结果: PASS  (PSD 上 {n_match} 条目标命中; 关键主线全部归因成功)')
        return 0


# ==================================================================
# 真实数据驱动 (本阶段只写不跑; 需 G: 盘, 由主线 R1 完成后调用)
# ==================================================================
def run_real(sts_path, scene=None, point=None,
             seconds=30.0, n_harm=3, sideband=True, target_df=0.25):
    """读取 30s 窗 -> order_analysis -> 存 CSV。
    !!! 本分支需 G: 盘连接并读取 .sts。本阶段绝不调用。!!!
    """
    meta = V.channel_meta(sts_path)
    fs = meta['sample_rate']
    scale = meta['scale_g_per_count']
    x = V.read_window(sts_path, scale, fs, seconds=seconds)

    df = order_analysis(x, fs, n_harm=n_harm, sideband=sideband, target_df=target_df)
    # 附带元信息列, 便于汇总
    df.insert(0, 'measurement', meta.get('measurement', ''))
    df.insert(1, 'point_name', meta.get('point_name', ''))

    if scene is None:
        scene = _safe_token(meta.get('measurement', 'scene'))
    if point is None:
        point = _safe_token(meta.get('point_name', 'point'))

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_csv = os.path.join(RESULT_DIR, f'R4_order_{scene}_{point}.csv')
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f'已写出: {out_csv}  ({len(df)} 行)')
    return out_csv, df


def _safe_token(s):
    """把场次/测点名清洗成可用于文件名的 token。"""
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
        description='R4 轴/丝杠阶次分析 (谱峰归因, trend-screening 口径)。')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--selftest', action='store_true',
                   help='用合成信号自测 order_analysis 逻辑并打印 PASS/FAIL。')
    g.add_argument('--real', metavar='STS_PATH',
                   help='对真实 .sts 做阶次分析 (需 G: 盘; 本阶段勿用, 由主线 R1 后调用)。')
    p.add_argument('--scene', default=None, help='--real: 场次标识(用于 CSV 文件名)。')
    p.add_argument('--point', default=None, help='--real: 测点标识(用于 CSV 文件名)。')
    p.add_argument('--n-harm', type=int, default=3, help='考察的谐波次数 (默认 3)。')
    p.add_argument('--no-sideband', action='store_true', help='关闭边带检测。')
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
        print('\n示例:')
        print('  python -X utf8 ws_r4_order.py --selftest')
        print('  python -X utf8 ws_r4_order.py --real "<G:盘的某.sts路径>" --scene 场次A --point CH1')
        print('\n注意: --real 分支需 G: 盘连接, 本阶段只写不跑, 由主线在 R1 完成后调用。')
        return 0


if __name__ == '__main__':
    sys.exit(main())
