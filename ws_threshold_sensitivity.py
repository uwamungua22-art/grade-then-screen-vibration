# -*- coding: utf-8 -*-
"""
F5 数据质量分级阈值敏感性分析 (JVET 期刊扩展版)
==================================================
被扫描的阈值 theta = "电气尖峰判定阈值" = 满标比例 pr 的临界值 (基线 0.98 = 98% 满标)。

数据质量四分级口径 (全 888 条单通道 .sts 记录):
  - empty(空/截断)      : n_samples < 8000 (<1s)。**固定 144 条, 不受 theta 影响。**
  - trustworthy(幅值可信): 非空 & td_peak <= 50 (g)。边界由 ENDEVCO 传感器 ±50g
                           物理量程决定, **不是 98% 阈值, 不随 theta 变。**
  - spike(电气尖峰)     : 非空 & pr >= theta。50g 传感器物理上不可能产生, 判为采集链伪迹。
  - over-range(超量程)  : 非空 & td_peak > 50 & pr < theta。

==> theta 只移动 over-range <-> spike 之间的边界;
    empty(144) 与 trustworthy(412) 不随 theta 变。

注意: td_peak / td_rms 等已是物理量 g (已乘 scale_g_per_count)。
      adc_fs = range_to * scale_g_per_count, 是以 g 为单位的采集链满标 (典型 ~101g)。
      pr = td_peak / adc_fs。

下游稳定性 (沿用 ws1_4_methods.py 的 F3/WS3a 口径):
  候选集 = (sample_rate==8000) & (~is_empty), 取 td_kurtosis 最大的 Top20。
  该候选集与 Top20 这 20 条记录本身不随 theta 变;
  随 theta 变化的只是 "Top20 中有几条被判定为 hard_spike(pr>=theta)"。
  基线结论是 Top20 中尖峰占 ~50% (10/20), F5 检验其是否稳健。

环境: 仅用本地 parquet。**绝不读 G: 盘或任何 .sts。** 不修改 ws_raw_trajectory.py / ws1_4_methods.py。
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = r'.'
FEAT = os.path.join(BASE, '00_inputs', 'features_all.parquet')
OUT = os.path.join(BASE, '04_jvet_results')
os.makedirs(OUT, exist_ok=True)

# 扫描阈值: 基线 0.98 加密 + 95~99% 边界
THETAS = [0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98, 0.985, 0.99]


def load():
    df = pd.read_parquet(FEAT)
    df['adc_fs'] = df['range_to'].astype(float) * df['scale_g_per_count']
    df['pr'] = df['td_peak'] / df['adc_fs']
    df['is_empty'] = df['n_samples'].fillna(0) < 8000
    return df


def classify_counts(df, theta):
    """在给定 theta 下重算四类计数。"""
    ne = ~df['is_empty']
    n_empty = int(df['is_empty'].sum())                       # 固定, 不随 theta
    n_trust = int((ne & (df['td_peak'] <= 50)).sum())         # 由 50g 物理量程定, 不随 theta
    n_spike = int((ne & (df['pr'] >= theta)).sum())
    n_over = int((ne & (df['td_peak'] > 50) & (df['pr'] < theta)).sum())
    return n_trust, n_over, n_spike, n_empty


def kurt_top20_spike_frac(df, theta):
    """F3/WS3a 下游: 8kHz 非空记录峭度 Top20 中, 被判为电气尖峰(pr>=theta)的占比。
    候选集与 Top20 这 20 条不随 theta 变, 只是判定标签随 theta 变。"""
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

    # 基线 0.98 校验
    base = res[np.isclose(res['theta'], 0.98)].iloc[0]
    print('=== theta=0.98 复算 ===')
    print(f"trustworthy={base.n_trustworthy} over-range={base.n_overrange} "
          f"spike={base.n_spike} empty={base.n_empty} (期望 412/143/189/144)")
    expected = (412, 143, 189, 144)
    got = (int(base.n_trustworthy), int(base.n_overrange), int(base.n_spike), int(base.n_empty))
    print('复现期望计数:' , got == expected)

    # over-range<->spike 迁移幅度 (95% vs 99%)
    lo = res.iloc[0]
    hi = res.iloc[-1]
    print(f"\n=== theta {lo.theta:.3f} -> {hi.theta:.3f} 迁移 ===")
    print(f"spike: {int(lo.n_spike)} -> {int(hi.n_spike)} (Delta={int(hi.n_spike-lo.n_spike)})")
    print(f"over : {int(lo.n_overrange)} -> {int(hi.n_overrange)} (Delta={int(hi.n_overrange-lo.n_overrange)})")
    print(f"trustworthy 全程恒定: {res['n_trustworthy'].nunique()==1} (={int(res.n_trustworthy.iloc[0])})")
    print(f"empty 全程恒定: {res['n_empty'].nunique()==1} (={int(res.n_empty.iloc[0])})")
    print(f"\n=== 峭度Top20尖峰占比 ===")
    print(res[['theta', 'spike_frac_in_kurt_top20', 'n_spike_in_kurt_top20']].to_string(index=False))

    csv_path = os.path.join(OUT, 'F5_threshold_sensitivity.csv')
    res.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print('\nCSV ->', csv_path)

    # ---- 出图 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) 四类计数 vs theta
    ax = axes[0]
    ax.plot(res['theta'], res['n_trustworthy'], 'o-', label='幅值可信 trustworthy (固定)', color='#2ca02c')
    ax.plot(res['theta'], res['n_overrange'], 's-', label='超量程 over-range', color='#ff7f0e')
    ax.plot(res['theta'], res['n_spike'], '^-', label='电气尖峰 spike', color='#d62728')
    ax.plot(res['theta'], res['n_empty'], 'd--', label='空/截断 empty (固定)', color='#7f7f7f')
    ax.axvline(0.98, color='k', ls=':', lw=1, alpha=0.6)
    ax.text(0.98, ax.get_ylim()[1]*0.5, ' 基线θ=0.98', rotation=90, va='center', fontsize=9)
    ax.set_xlabel('电气尖峰判定阈值 θ (满标比例)')
    ax.set_ylabel('记录数')
    ax.set_title('(a) 四类质量分级计数 vs 阈值 θ')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (b) 峭度Top20 尖峰占比 vs theta
    ax = axes[1]
    ax.plot(res['theta'], res['spike_frac_in_kurt_top20']*100, 'o-', color='#d62728')
    for _, r in res.iterrows():
        ax.annotate(f"{int(r.n_spike_in_kurt_top20)}/20",
                    (r.theta, r.spike_frac_in_kurt_top20*100),
                    textcoords='offset points', xytext=(0, 7), ha='center', fontsize=8)
    ax.axhline(50, color='gray', ls='--', lw=1, alpha=0.7)
    ax.axvline(0.98, color='k', ls=':', lw=1, alpha=0.6)
    ax.set_ylim(0, 100)
    ax.set_xlabel('电气尖峰判定阈值 θ (满标比例)')
    ax.set_ylabel('峭度Top20中电气尖峰占比 (%)')
    ax.set_title('(b) 8kHz非空峭度Top20 尖峰占比 vs 阈值 θ')
    ax.grid(alpha=0.3)

    fig.suptitle('F5 数据质量分级阈值敏感性分析', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = os.path.join(OUT, 'F5_threshold_sensitivity.png')
    fig.savefig(png_path, dpi=180, bbox_inches='tight')
    print('PNG ->', png_path)


if __name__ == '__main__':
    main()
