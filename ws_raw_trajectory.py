# -*- coding: utf-8 -*-
"""R 类分析(需重读 220GB 原始波形, 外置盘连接后跑)。
R1 已实现: 关键通道全程滑窗 RMS/峭度轨迹(记录内时变, 最大增量来源)。
R2/R3/R4 为桩(spec 见 docstring), 待补完。
依赖 vibe_core(同目录)。解释器: python -X utf8
用法:
  python ws_raw_trajectory.py --check         # 仅校验外置盘可读 + 枚举 .sts
  python ws_raw_trajectory.py --r1            # 跑 R1 滑窗轨迹
"""
import sys, io, os, re, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd

import vibe_core as V

# ---- 配置: 外置盘根目录(盘符可能变, 连盘后核对) ----
RAW_ROOT = r'./data/raw'
OUT = r'./results'

# R1 关键通道(按 point_name 子串匹配, 复用会议版结论通道)
TARGET_SUBSTR = ['换向器', '右丝杠下', '升降右丝杠', '丝杠中']  # CH6/CH19/CH16/CH10
WIN_S = 1.0      # 滑窗长度(秒)
HOP_S = 1.0      # 步长(秒); 等于 WIN_S 即不重叠


def ts(m):
    g = re.search(r'(\d\d)-(\d\d)-(\d\d) (\d\d)-(\d\d)-(\d\d)', m)
    return pd.Timestamp(2000 + int(g[1]), int(g[2]), int(g[3]),
                        int(g[4]), int(g[5]), int(g[6])) if g else pd.NaT


def check_drive():
    if not os.path.isdir(RAW_ROOT):
        print(f'[FAIL] 外置盘根目录不存在: {RAW_ROOT}\n  -> 改本文件顶部 RAW_ROOT 为正确盘符/路径')
        return []
    stss = V.find_all_sts(RAW_ROOT)
    print(f'[OK] 枚举到 {len(stss)} 个 .sts')
    for p in stss[:3]:
        print('   ', p)
    return stss


def sliding_trajectory(sts_path, scale, fs, win_s=WIN_S, hop_s=HOP_S, max_windows=None):
    """对单条记录做全程滑窗 RMS/峭度轨迹。顺序读取, 内存恒定。"""
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
    return np.array(rows)  # 列: [t_sec, rms, kurtosis]


def run_r1():
    os.makedirs(OUT, exist_ok=True)
    stss = check_drive()
    if not stss:
        return
    # 建索引: 每个 .sts 的 meta(用 8kHz, 跳过空/尖峰由后续可选过滤)
    recs = []
    for p in stss:
        meta = V.channel_meta(p)
        if meta['sample_rate'] != 8000:
            continue
        if not any(sub in meta['point_name'] for sub in TARGET_SUBSTR):
            continue
        recs.append(meta)
    print(f'匹配关键通道记录: {len(recs)}')
    summary = []
    for meta in recs:
        traj = sliding_trajectory(meta['sts_path'], meta['scale_g_per_count'], meta['sample_rate'])
        if traj.size == 0:
            continue
        sess = meta['measurement']
        pt = meta['point_name']
        # 存逐窗轨迹
        fn = f"{sess}__{pt}".replace(' ', '_').replace('/', '_').replace('\\', '_')[:120]
        np.savetxt(os.path.join(OUT, f'R1_traj__{fn}.csv'), traj,
                   delimiter=',', header='t_sec,rms,kurtosis', comments='', fmt='%.6f')
        # 场次级汇总: 窗级分布而非单点(论文卖点: 揭示记录内非平稳)
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
    print(f'[done] R1 轨迹与窗级汇总 -> {OUT}')
    print(df.to_string(index=False))


# ---- R2/R3/R4 桩(spec, 待补完) ----
def run_r2():
    """R2 跨传感器空间一致性(替代验证):
    取同一部件相邻测点(如换向器多测点/同侧丝杠上中下), 计算其跨场次 RMS 趋势的
    两两相关; 真实退化应多测点同向(高相关), 单通道伪迹则孤立。
    输入: 可直接用本地 features_all.parquet 的 td_rms(不必重读原始!), 因此 R2 其实
    也能在本地先做 -- 见 ws1_4_methods 的 load()。产出: 测点×测点相关热图 + 同步性指标。"""
    raise NotImplementedError('R2 待补; 提示: 可用本地 parquet 先做, 见 docstring')


def run_r3():
    """R3 退化轨迹建模 + 变点检测:
    对 CH6/CH19 的跨场次 RMS 序列(序数轴)拟合 线性/指数, 给 Bootstrap CI;
    用 ruptures/PELT 或简单 CUSUM 检测退化拐点。强调序数轴非标定时间。
    可用本地 parquet, 不必重读原始。"""
    raise NotImplementedError('R3 待补')


def run_r4():
    """R4 轴/丝杠阶次分析(需原始波形):
    对关键通道关键场次重算细分辨率 Welch + 包络谱, 比对 50Hz(电机)、1.83/4.58Hz(丝杠转频)
    及其谐波/边带; 能匹配的标注为轴序来源, 轴承 BPFO 因缺几何明确留为局限。
    用 vibe_core.read_window + frequency_features/envelope_features(可调 band)。"""
    raise NotImplementedError('R4 待补')


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
