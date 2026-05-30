# -*- coding: utf-8 -*-
"""F6 期刊版组图 (JVET, 英文标签, 期刊风格, >=300dpi, PNG+PDF)。
复用 ws1_4_methods.load() 的 usable 过滤口径, 仅用本地 parquet, 不读 G: 盘。
口径: data-quality grading / trend screening; 横轴为场次序数(非标定时间); 不作诊断声明。
端点变化统一用稳健"中位前3->中位后3"(与 QR2MSE 会议版一致), 而非字面单点。
解释器: python -X utf8
"""
import os, re, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr, theilslopes, norm

import ws1_4_methods as W

OUT = r'./results'
os.makedirs(OUT, exist_ok=True)

# ---- 期刊风格 ----
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 120,
    'savefig.dpi': 400,
    'savefig.bbox': 'tight',
})
# Okabe-Ito 色盲友好
CB = {'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'red': '#D55E00',
      'purple': '#CC79A7', 'sky': '#56B4E9', 'yellow': '#F0E442', 'grey': '#999999'}
CM = 1 / 2.54
COL1, COL2 = 8.6 * CM, 17.6 * CM  # Springer 单/双栏宽度


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'))
    plt.close(fig)
    print('  saved', name + '.png/.pdf')


def fisher_ci(stat, n, spearman=True, alpha=0.05):
    if abs(stat) >= 1 or n < 5:
        return (np.nan, np.nan)
    z = np.arctanh(stat)
    se = (1.03 if spearman else 1.0) / np.sqrt(n - 3)
    zc = norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))


def short_label(name):
    ch = re.search(r'CH\s*(\d+)', name)
    tag = f'CH{ch.group(1)}' if ch else name[:6]
    if '换向器' in name:
        kind = 'commutator'
    elif '丝杠' in name:
        kind = 'lead-screw'
    elif '导轨' in name:
        kind = 'guide-rail'
    else:
        kind = ''
    return (tag + (' ' + kind if kind else '')).strip()


def series(named, point):
    s = named[named.point_name == point].dropna(subset=['td_rms']).sort_values('t')
    return s.td_rms.values.astype(float)


def main():
    df = W.load()
    u = df[df['usable']].copy()
    named = u[u.point_name.str.contains('门|导轨|丝杠|换向|水电|升降', na=False)]
    pts = [p for p, c in named.point_name.value_counts().items() if c >= 8]

    # changepoints from R3 (序数, 0-based)
    cp = {'CH 6-下门-换向器-y轴': 7, 'CH19-下门-右丝杠下': 9, 'CH 16-DZQ升降右丝杠下': 16}

    # =========================================================
    # Fig 1: 数据质量分级 + 消融 + 阈值稳健性
    # =========================================================
    fig, ax = plt.subplots(1, 3, figsize=(COL2, 5.2 * CM))

    # (a) 四类分级计数
    cls = ['Trustworthy', 'Over-range', 'Electrical\nspike', 'Empty']
    cnt = [412, 143, 189, 144]
    cols = [CB['green'], CB['orange'], CB['red'], CB['grey']]
    bars = ax[0].bar(range(4), cnt, color=cols, width=0.7, edgecolor='k', linewidth=0.4)
    for b, c in zip(bars, cnt):
        ax[0].text(b.get_x() + b.get_width() / 2, c + 6, str(c), ha='center', va='bottom', fontsize=6.5)
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(cls, fontsize=6.5)
    ax[0].set_ylabel('Number of records')
    ax[0].set_ylim(0, 470)
    ax[0].set_title('(a) Data-quality grading (n=888)')
    ax[0].text(0.97, 0.93, 'Non-trustworthy\n476 / 888 = 54%', transform=ax[0].transAxes,
               ha='right', va='top', fontsize=6.5,
               bbox=dict(boxstyle='round', fc='white', ec=CB['grey'], lw=0.6))

    # (b) 峭度: 尖峰 vs 可信
    nonempty = df[(df.sample_rate == 8000) & (~df.is_empty)]
    k_spike = nonempty[nonempty.hard_spike].td_kurtosis.dropna().values
    k_trust = df[df.usable].td_kurtosis.dropna().values
    bp = ax[1].boxplot([k_trust, k_spike], positions=[0, 1], widths=0.55,
                       showfliers=False, patch_artist=True)
    for patch, c in zip(bp['boxes'], [CB['green'], CB['red']]):
        patch.set_facecolor(c); patch.set_alpha(0.6); patch.set_edgecolor('k'); patch.set_linewidth(0.5)
    for med in bp['medians']:
        med.set_color('k'); med.set_linewidth(1.0)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(['Trustworthy', 'Electrical\nspike'], fontsize=6.5)
    ax[1].set_xlim(-0.6, 1.85)
    ax[1].set_ylabel('Kurtosis')
    ax[1].set_title('(b) Kurtosis inflation by spikes')
    ax[1].text(0.5, 1.0, '10 / 20 of kurtosis Top-20\nare spike artifacts',
               transform=ax[1].transAxes, ha='center', va='top', fontsize=6.5,
               bbox=dict(boxstyle='round', fc='white', ec=CB['grey'], lw=0.6))
    ax[1].text(0.33, np.median(k_trust), f'med {np.median(k_trust):.1f}', fontsize=6,
               color=CB['green'], va='center', ha='left')
    ax[1].text(1.33, np.median(k_spike), f'med {np.median(k_spike):.1f}', fontsize=6,
               color=CB['red'], va='center', ha='left')

    # (c) 阈值敏感性
    f5 = pd.read_csv(os.path.join(OUT, 'F5_threshold_sensitivity.csv'))
    ax2 = ax[2]
    ax2.plot(f5.theta * 100, f5.n_spike, 'o-', color=CB['red'], ms=3, lw=1.0, label='Spike')
    ax2.plot(f5.theta * 100, f5.n_overrange, 's-', color=CB['orange'], ms=3, lw=1.0, label='Over-range')
    ax2.set_xlabel('Spike threshold (% of full scale)')
    ax2.set_ylabel('Number of records')
    ax2.set_ylim(120, 210)
    ax2.set_title('(c) Threshold robustness')
    ax3 = ax2.twinx()
    ax3.plot(f5.theta * 100, f5.spike_frac_in_kurt_top20 * 100, '^--', color=CB['blue'], ms=3, lw=1.0,
             label='Spike % in kurtosis Top-20')
    ax3.set_ylabel('Spike share in Top-20 (%)', color=CB['blue'])
    ax3.set_ylim(0, 100); ax3.tick_params(axis='y', colors=CB['blue'])
    ax3.spines['right'].set_visible(True); ax3.spines['top'].set_visible(False)
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc='center right', frameon=False, fontsize=6)
    save(fig, 'F6_fig1_dataquality')

    # =========================================================
    # Fig 2: 趋势筛查 + 效应量CI (接受 CH6/CH19, 拒绝 CH16/CH10)
    # =========================================================
    panels = [
        ('CH 6-下门-换向器-y轴', 'ACCEPTED', CB['green']),
        ('CH19-下门-右丝杠下', 'ACCEPTED', CB['green']),
        ('CH 16-DZQ升降右丝杠下', 'WITHDRAWN (non-monotonic)', CB['orange']),
        ('CH 10-右门-丝杠中', 'EXCLUDED (negative artifact)', CB['red']),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(COL2, 12 * CM))
    for axp, (pt, verdict, vc) in zip(axs.ravel(), panels):
        y = series(named, pt)
        n = len(y); x = np.arange(n)
        rho, _ = spearmanr(x, y); rlo, rhi = fisher_ci(rho, n)
        sen, b0, slo, shi = theilslopes(y, x, 0.95)
        # 稳健端点(中位前3->后3)与字面
        med0, med1 = np.median(y[:3]), np.median(y[-3:])
        pct = 100 * (med1 / med0 - 1)
        axp.scatter(x, y, s=14, color=vc, edgecolor='k', linewidth=0.3, zorder=3, alpha=0.85)
        xs = np.array([0, n - 1])
        axp.plot(xs, b0 + sen * xs, '-', color='k', lw=1.2, zorder=4,
                 label=f'Theil–Sen {sen:+.3f} g/session')
        # CI 带
        axp.fill_between([0, n - 1], [b0 + slo * 0, b0 + slo * (n - 1)],
                         [b0 + shi * 0, b0 + shi * (n - 1)], color='k', alpha=0.12, zorder=1)
        # 物理量 RMS>=0: 钳下限到 0, 上限留题注空间; CI 带负值部分被裁
        ymax = float(y.max())
        axp.set_ylim(0, ymax * 1.18)
        if pt in cp:
            axp.axvline(cp[pt], color=CB['purple'], ls=':', lw=1.0, zorder=2)
            axp.text(cp[pt], 0.96, f' CP@{cp[pt]}', color=CB['purple'], fontsize=6,
                     va='top', ha='left', transform=axp.get_xaxis_transform())
        axp.set_xlabel('Session ordinal index (not calibrated time)')
        axp.set_ylabel('RMS (g)')
        axp.set_title(f'{short_label(pt)} — {verdict}', color=vc, fontsize=8)
        txt = (f'$\\rho$ = {rho:.2f} [{rlo:.2f}, {rhi:.2f}]\n'
               f'Sen = {sen:+.3f} [{slo:+.3f}, {shi:+.3f}] g/session\n'
               f'robust endpoints {med0:.2f} → {med1:.2f} g ({pct:+.0f}%)')
        axp.text(0.03, 0.97, txt, transform=axp.transAxes, ha='left', va='top', fontsize=6,
                 bbox=dict(boxstyle='round', fc='white', ec=vc, lw=0.7))
        axp.legend(loc='lower right', frameon=False, fontsize=6)
    fig.suptitle('Trend screening on the session-ordinal axis (single specimen; screening, not diagnosis)',
                 fontsize=8.5, y=1.0)
    save(fig, 'F6_fig2_trend_screening')

    # =========================================================
    # Fig 3: 跨传感器空间一致性 (替代验证)
    # =========================================================
    # 重算两两 Spearman 矩阵(共同场次>=8)
    labels = [short_label(p) for p in pts]
    order = np.argsort(labels)
    pts_o = [pts[i] for i in order]; labels_o = [labels[i] for i in order]
    m = len(pts_o)
    M = np.full((m, m), np.nan)
    for i in range(m):
        si = named[named.point_name == pts_o[i]].dropna(subset=['td_rms']).set_index('measurement')['td_rms']
        for j in range(m):
            sj = named[named.point_name == pts_o[j]].dropna(subset=['td_rms']).set_index('measurement')['td_rms']
            common = si.index.intersection(sj.index)
            if len(common) >= 8:
                r, _ = spearmanr(si.loc[common].values, sj.loc[common].values)
                M[i, j] = r
    fig, ax = plt.subplots(1, 2, figsize=(COL2, 8 * CM),
                           gridspec_kw={'width_ratios': [1.5, 1], 'wspace': 0.55})
    im = ax[0].imshow(M, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax[0].set_xticks(range(m)); ax[0].set_xticklabels(labels_o, rotation=90, fontsize=5.5)
    ax[0].set_yticks(range(m)); ax[0].set_yticklabels(labels_o, fontsize=5.5)
    ax[0].set_title('(a) Cross-sensor RMS-trend correlation (Spearman)')
    cb = fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    cb.set_label('Spearman $\\rho$', fontsize=6.5); cb.ax.tick_params(labelsize=6)
    ax[0].spines[:].set_visible(False)

    syn = pd.read_csv(os.path.join(OUT, 'R2_synchrony_metrics.csv'))
    wm = float(syn[syn.metric == 'within_group_mean_spearman'].value.iloc[0])
    am = float(syn[syn.metric == 'across_group_mean_spearman'].value.iloc[0])
    ax[1].bar([0, 1], [wm, am], color=[CB['blue'], CB['grey']], width=0.6,
              edgecolor='k', linewidth=0.4)
    for xi, v in zip([0, 1], [wm, am]):
        ax[1].text(xi, v + 0.005, f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(['Within\ncomponent', 'Across\ncomponent'], fontsize=6.5)
    ax[1].set_ylabel('Mean pairwise $\\rho$')
    ax[1].set_ylim(0, 0.33)
    ax[1].set_title('(b) Co-located coherence')
    ax[1].text(0.5, 0.9, 'co-located sensors\nmore coherent →\nsupports component-\nlevel degradation',
               transform=ax[1].transAxes, ha='center', va='top', fontsize=6,
               bbox=dict(boxstyle='round', fc='white', ec=CB['grey'], lw=0.6))
    save(fig, 'F6_fig3_spatial_consistency')

    print('\nF6 期刊版组图完成 ->', OUT)


if __name__ == '__main__':
    main()
