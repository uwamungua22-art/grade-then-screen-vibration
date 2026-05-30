# -*- coding: utf-8 -*-
"""
ws_r3_trajmodel.py  (R3)
退化轨迹建模 + 变点检测 —— 质控/趋势筛查(trend-screening)用途。

口径铁律(务必牢记，勿在产出中越界)：
- 数据=单试件、真实在役现场振动记录，含真实退化但**无拆检/标签 ground truth**。
- 定位 data-quality / trend-screening；非 diagnosis / fault detection / RUL prediction / validated。
- 轨迹拟合只是描述趋势形态并给不确定度，不是寿命预测。
- 横轴为**场次序数(ordinal session index)，非标定时间**(场次间有拆装、间隔不均)。
- 关键通道 n≈23~29，结论须带小样本 caveat。

环境约束：
- 解释器 python
- 只读本地 parquet，绝不碰 G: / .sts。
- ruptures 未装 -> 用自实现 binary segmentation (CUSUM-style) fallback。
- 不修改 ws_raw_trajectory.py / ws1_4_methods.py。
"""
import os, re, sys
import numpy as np
import pandas as pd
from scipy import optimize, stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

FEAT = r"./data/features_all.parquet"
OUTDIR = r"./results"
os.makedirs(OUTDIR, exist_ok=True)

RNG = np.random.default_rng(20260528)
N_BOOT = 3000

# 关键测点(逐字)。display=图/CSV 中的短标签
CHANNELS = [
    ("CH 6-下门-换向器-y轴",   "CH6",  "主退化通道"),
    ("CH19-下门-右丝杠下",      "CH19", "次退化通道"),
    ("CH 16-DZQ升降右丝杠下",  "CH16", "对照(已知非单调)"),
]


def ts(m):
    g = re.search(r'(\d\d)-(\d\d)-(\d\d) (\d\d)-(\d\d)-(\d\d)', str(m))
    return pd.Timestamp(2000+int(g[1]), int(g[2]), int(g[3]),
                        int(g[4]), int(g[5]), int(g[6])) if g else pd.NaT


def load_usable():
    df = pd.read_parquet(FEAT)
    df['adc_fs'] = df['range_to'].astype(float) * df['scale_g_per_count']
    df['pr'] = df['td_peak'] / df['adc_fs']
    df['is_empty'] = df['n_samples'].fillna(0) < 8000
    df['hard_spike'] = (~df['is_empty']) & (df['pr'] >= 0.98)
    df['usable'] = (~df['is_empty']) & (~df['hard_spike']) & (df['sample_rate'] == 8000)
    df['t'] = df['measurement'].map(ts)
    return df


# ---------------------------------------------------------------------------
# 拟合工具
# ---------------------------------------------------------------------------
def _aic(n, rss, k):
    """k = 自由参数个数(不含残差方差)。AIC = n*ln(rss/n) + 2*(k+1)。"""
    if rss <= 0:
        rss = 1e-12
    return n * np.log(rss / n) + 2 * (k + 1)


def fit_linear(x, y):
    """y = a + b*x，最小二乘。返回 dict。"""
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    yhat = a + b * x
    rss = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - rss / sst if sst > 0 else np.nan
    return dict(a=a, b=b, yhat=yhat, rss=rss, r2=r2, aic=_aic(len(x), rss, 2))


def _exp_model(x, a, b):
    return a * np.exp(b * x)


def fit_exp(x, y):
    """y = a*exp(b*x)。先用 log y 线性回归取初值，再 curve_fit。返回 dict 或 None。"""
    ypos = np.clip(y, 1e-6, None)
    # 初值：对 log y 做线性回归
    A = np.vstack([np.ones_like(x), x]).T
    c, *_ = np.linalg.lstsq(A, np.log(ypos), rcond=None)
    a0, b0 = np.exp(c[0]), c[1]
    try:
        popt, _ = optimize.curve_fit(_exp_model, x, y, p0=[a0, b0], maxfev=20000)
        a, b = popt
        yhat = _exp_model(x, a, b)
        if not np.all(np.isfinite(yhat)):
            raise RuntimeError("non-finite")
    except Exception:
        # 退回 log 线性结果
        a, b = a0, b0
        yhat = _exp_model(x, a, b)
    rss = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - rss / sst if sst > 0 else np.nan
    return dict(a=a, b=b, yhat=yhat, rss=rss, r2=r2, aic=_aic(len(x), rss, 2))


def bootstrap_ci(x, y, kind="linear", n_boot=N_BOOT, alpha=0.05):
    """对 (x,y) 点对做 bootstrap 重采样，返回参数 CI 与逐点拟合曲线 CI。"""
    n = len(x)
    bs_b = []
    bs_a = []
    curves = []
    xg = x.astype(float)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if len(np.unique(xb)) < 2:
            continue
        try:
            if kind == "linear":
                f = fit_linear(xb, yb)
                yhat_full = f['a'] + f['b'] * xg
            else:
                f = fit_exp(xb, yb)
                yhat_full = _exp_model(xg, f['a'], f['b'])
            if not np.all(np.isfinite(yhat_full)):
                continue
            bs_a.append(f['a']); bs_b.append(f['b']); curves.append(yhat_full)
        except Exception:
            continue
    bs_a = np.array(bs_a); bs_b = np.array(bs_b); curves = np.array(curves)
    lo, hi = 100*alpha/2, 100*(1-alpha/2)
    out = dict(
        a_ci=(np.percentile(bs_a, lo), np.percentile(bs_a, hi)) if len(bs_a) else (np.nan, np.nan),
        b_ci=(np.percentile(bs_b, lo), np.percentile(bs_b, hi)) if len(bs_b) else (np.nan, np.nan),
        n_eff=len(bs_b),
    )
    if len(curves):
        out['curve_lo'] = np.percentile(curves, lo, axis=0)
        out['curve_hi'] = np.percentile(curves, hi, axis=0)
    else:
        out['curve_lo'] = out['curve_hi'] = None
    return out


# ---------------------------------------------------------------------------
# 变点检测 (fallback: 二分分割，残差平方和增益准则 + 排列检验显著性)
# ---------------------------------------------------------------------------
def _seg_rss(y):
    return float(np.sum((y - y.mean()) ** 2))


def _best_split(y, min_size):
    """单段内找最优均值突变点：使 RSS 下降最大。返回 (位置, 增益)；无有效切点返回 (None, 0)。"""
    n = len(y)
    base = _seg_rss(y)
    best_pos, best_gain = None, 0.0
    for k in range(min_size, n - min_size + 1):
        gain = base - (_seg_rss(y[:k]) + _seg_rss(y[k:]))
        if gain > best_gain:
            best_gain, best_pos = gain, k
    return best_pos, best_gain


def _perm_pvalue(y, pos, n_perm=2000):
    """对'是否存在该幅度的均值突变'做排列检验：打乱顺序后能否出现>=观测增益的切分。"""
    n = len(y)
    base = _seg_rss(y)
    obs_gain = base - (_seg_rss(y[:pos]) + _seg_rss(y[pos:]))
    cnt = 0
    for _ in range(n_perm):
        yp = RNG.permutation(y)
        # 在打乱序列上找全局最优增益(更严格的零分布)
        _, g = _best_split(yp, min_size=2)
        if g >= obs_gain:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def detect_changepoints(y, max_cp=2, min_size=4, alpha=0.05):
    """二分分割：递归找显著均值突变点。
    返回 list[dict(pos, gain, pvalue)]，pos 为序数(0-based, 段右起点)。"""
    n = len(y)
    found = []
    segments = [(0, n)]
    while len(found) < max_cp and segments:
        # 在所有当前段中找全局最优候选
        cand = []
        for (s, e) in segments:
            seg = y[s:e]
            if len(seg) < 2 * min_size:
                continue
            pos, gain = _best_split(seg, min_size)
            if pos is not None:
                cand.append((gain, s + pos, s, e))
        if not cand:
            break
        cand.sort(reverse=True)
        gain, abspos, s, e = cand[0]
        pval = _perm_pvalue(y[s:e], abspos - s)
        if pval > alpha:
            break  # 最优候选都不显著则停止
        found.append(dict(pos=int(abspos), gain=float(gain), pvalue=float(pval),
                          seg=(s, e)))
        # 拆分该段
        segments = [seg for seg in segments if seg != (s, e)]
        segments += [(s, abspos), (abspos, e)]
    found.sort(key=lambda d: d['pos'])
    return found


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    df = load_usable()
    fit_rows = []
    cp_rows = []
    panels = []

    for full, disp, role in CHANNELS:
        sub = df[(df['point_name'] == full) & df['usable']].sort_values('t')
        n = len(sub)
        if n < 5:
            print(f"[WARN] {disp} 仅 {n} 条 usable，跳过")
            continue
        y = sub['td_rms'].to_numpy(dtype=float)
        x = np.arange(n, dtype=float)

        lin = fit_linear(x, y)
        exp = fit_exp(x, y)
        ci_lin = bootstrap_ci(x, y, "linear")
        ci_exp = bootstrap_ci(x, y, "exp")

        better = "linear" if lin['aic'] <= exp['aic'] else "exp"
        d_aic = exp['aic'] - lin['aic']  # >0 -> 线性更优

        fit_rows.append(dict(
            channel=disp, point_name=full, role=role, n=n,
            model="linear", a=lin['a'], b_slope=lin['b'],
            b_ci_lo=ci_lin['b_ci'][0], b_ci_hi=ci_lin['b_ci'][1],
            a_ci_lo=ci_lin['a_ci'][0], a_ci_hi=ci_lin['a_ci'][1],
            R2=lin['r2'], AIC=lin['aic'], boot_n_eff=ci_lin['n_eff'],
            rms_first=y[0], rms_last=y[-1],
            pct_change=100*(y[-1]-y[0])/y[0] if y[0] != 0 else np.nan,
            better_model=better, delta_AIC_exp_minus_lin=d_aic,
        ))
        fit_rows.append(dict(
            channel=disp, point_name=full, role=role, n=n,
            model="exp", a=exp['a'], b_slope=exp['b'],
            b_ci_lo=ci_exp['b_ci'][0], b_ci_hi=ci_exp['b_ci'][1],
            a_ci_lo=ci_exp['a_ci'][0], a_ci_hi=ci_exp['a_ci'][1],
            R2=exp['r2'], AIC=exp['aic'], boot_n_eff=ci_exp['n_eff'],
            rms_first=y[0], rms_last=y[-1],
            pct_change=100*(y[-1]-y[0])/y[0] if y[0] != 0 else np.nan,
            better_model=better, delta_AIC_exp_minus_lin=d_aic,
        ))

        cps = detect_changepoints(y, max_cp=2, min_size=4, alpha=0.05)
        if cps:
            for c in cps:
                conf = ("显著(p<0.05, 排列检验)" if c['pvalue'] < 0.05
                        else "弱(p>=0.05)")
                cp_rows.append(dict(
                    channel=disp, point_name=full, n=n,
                    method="binary_segmentation(fallback,均值突变+排列检验)",
                    changepoint_ordinal=c['pos'], rss_gain=round(c['gain'], 4),
                    p_value=round(c['pvalue'], 4), confidence=conf,
                    note="序数位置(0-based)，非标定时间；小样本谨慎解读",
                ))
        else:
            cp_rows.append(dict(
                channel=disp, point_name=full, n=n,
                method="binary_segmentation(fallback,均值突变+排列检验)",
                changepoint_ordinal=-1, rss_gain=0.0, p_value=np.nan,
                confidence="未检出显著变点(alpha=0.05)",
                note="序数位置，非标定时间；小样本谨慎解读",
            ))

        panels.append((disp, role, x, y, lin, exp, ci_lin, ci_exp, cps, n))

        print(f"[{disp}] n={n} 线性b={lin['b']:.4f} CI=[{ci_lin['b_ci'][0]:.4f},"
              f"{ci_lin['b_ci'][1]:.4f}] R2={lin['r2']:.3f} AIC={lin['aic']:.2f} | "
              f"指数b={exp['b']:.4f} R2={exp['r2']:.3f} AIC={exp['aic']:.2f} | "
              f"更优={better} | 变点={[c['pos'] for c in cps] if cps else '无'}")

    # 保存 CSV
    fdf = pd.DataFrame(fit_rows)
    cdf = pd.DataFrame(cp_rows)
    f_csv = os.path.join(OUTDIR, "R3_trajectory_fits.csv")
    c_csv = os.path.join(OUTDIR, "R3_changepoints.csv")
    fdf.to_csv(f_csv, index=False, encoding="utf-8-sig")
    cdf.to_csv(c_csv, index=False, encoding="utf-8-sig")
    print("saved:", f_csv)
    print("saved:", c_csv)

    # 出图：综合 + 每通道单图
    npan = len(panels)
    fig, axes = plt.subplots(1, npan, figsize=(6*npan, 5), dpi=160)
    if npan == 1:
        axes = [axes]
    for ax, (disp, role, x, y, lin, exp, ci_lin, ci_exp, cps, n) in zip(axes, panels):
        _plot_one(ax, disp, role, x, y, lin, exp, ci_lin, exp_ci=ci_exp, cps=cps, n=n)
    fig.suptitle("退化轨迹建模与变点检测（趋势筛查；横轴=场次序数，非标定时间；单试件，小样本）",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    big = os.path.join(OUTDIR, "R3_traj_fits.png")
    fig.savefig(big, dpi=160)
    plt.close(fig)
    print("saved:", big)

    for (disp, role, x, y, lin, exp, ci_lin, ci_exp, cps, n) in panels:
        f1, a1 = plt.subplots(figsize=(7, 5), dpi=160)
        _plot_one(a1, disp, role, x, y, lin, exp, ci_lin, exp_ci=ci_exp, cps=cps, n=n)
        p = os.path.join(OUTDIR, f"R3_traj_{disp}.png")
        f1.tight_layout()
        f1.savefig(p, dpi=160)
        plt.close(f1)
        print("saved:", p)

    print("\nDONE. 未触碰 G: / .sts / 共享脚本。")


def _plot_one(ax, disp, role, x, y, lin, exp, ci_lin, exp_ci, cps, n):
    ax.scatter(x, y, s=42, color="#1f77b4", zorder=3, label="usable RMS 数据点")
    ax.plot(x, lin['yhat'], color="#d62728", lw=2,
            label=f"线性拟合 b={lin['b']:.3f} (R2={lin['r2']:.2f})")
    if ci_lin.get('curve_lo') is not None:
        ax.fill_between(x, ci_lin['curve_lo'], ci_lin['curve_hi'],
                        color="#d62728", alpha=0.18, label="线性 Bootstrap 95% CI")
    ax.plot(x, exp['yhat'], color="#2ca02c", lw=2, ls="--",
            label=f"指数拟合 b={exp['b']:.3f} (R2={exp['r2']:.2f})")
    sig_cps = [c for c in cps if c['pvalue'] < 0.05]
    for i, c in enumerate(sig_cps):
        ax.axvline(c['pos'], color="#7f3fbf", lw=1.8, ls=":",
                   label=("变点(序数,p<0.05)" if i == 0 else None))
        ax.annotate(f"x={c['pos']}", xy=(c['pos'], ax.get_ylim()[1]),
                    xytext=(c['pos'], y.max()*0.98), color="#7f3fbf",
                    fontsize=9, ha="center")
    ax.set_xlabel("场次序数（非标定时间）")
    ax.set_ylabel("RMS (g)")
    ax.set_title(f"{disp}  {role}  (n={n})")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)


if __name__ == "__main__":
    main()
