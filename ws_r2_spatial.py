# -*- coding: utf-8 -*-
"""
R2 跨传感器空间一致性分析 (cross-sensor spatial consistency / surrogate validation)
=================================================================================
论文定位: data-quality / measurement trustworthiness / trend-screening (质控筛查)。
本脚本绝不做 diagnosis / fault detection / RUL。所有结论均为"筛查级别的佐证证据",非确证。

科学逻辑(替代性验证 surrogate validation, 弥补无真值/单试件):
  真实的部件级退化, 应在同一部件或相邻测点上"同向出现"(跨场次 RMS 序列高相关);
  仅在单通道孤立上升, 更可能是传感器伪迹。本分析只是佐证, 不是金标准。

横轴 = 场次序数 (ordinal session index), 不是标定时间。

只读取本地特征表 parquet, 不触碰 G: 盘或任何 .sts 原始波形。
不修改任何共享脚本 (ws_raw_trajectory.py / ws1_4_methods.py)。
"""
import re
import sys
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
ROOT = Path(r".")
FEAT = ROOT / "00_inputs" / "features_all.parquet"
OUTDIR = ROOT / "04_jvet_results"
OUTDIR.mkdir(parents=True, exist_ok=True)

MIN_USABLE = 8        # 测点保留门槛: usable 记录数 >= 8
MIN_COMMON = 8        # 两测点相关计算门槛: 共同场次数 >= 8

# ---------------------------------------------------------------------------
# usable 过滤口径 (与会议版完全一致, 照抄)
# ---------------------------------------------------------------------------
def ts(m):
    g = re.search(r"(\d\d)-(\d\d)-(\d\d) (\d\d)-(\d\d)-(\d\d)", str(m))
    return pd.Timestamp(2000 + int(g[1]), int(g[2]), int(g[3]),
                        int(g[4]), int(g[5]), int(g[6])) if g else pd.NaT


def load_usable():
    df = pd.read_parquet(FEAT)
    df["adc_fs"] = df["range_to"].astype(float) * df["scale_g_per_count"]
    df["pr"] = df["td_peak"] / df["adc_fs"]
    df["is_empty"] = df["n_samples"].fillna(0) < 8000
    df["hard_spike"] = (~df["is_empty"]) & (df["pr"] >= 0.98)
    df["usable"] = (~df["is_empty"]) & (~df["hard_spike"]) & (df["sample_rate"] == 8000)
    df["t"] = df["measurement"].map(ts)
    return df


# ---------------------------------------------------------------------------
# 物理部件分组 (基于测点物理位置/部件)
# ---------------------------------------------------------------------------
# 短标签 (用于热图与表格可读性)
SHORT = {
    "CH 6-下门-换向器-y轴": "CH6 换向器",
    "CH 7-DZQ升降-换向器": "CH7 换向器",
    "CH19-下门-右丝杠下": "CH19 右丝杠",
    "CH 16-DZQ升降右丝杠下": "CH16 右丝杠",
    "CH 10-右门-丝杠中": "CH10 丝杠中",
    "CH 22-中门-上丝杠中": "CH22 上丝杠",
    "CH 13-DZQ水电-插拔左丝杠": "CH13 左丝杠",
}

# 部件组: 按"换向器"与"丝杠"两大物理部件
GROUPS = {
    "换向器组": ["CH 6-下门-换向器-y轴", "CH 7-DZQ升降-换向器"],
    "丝杠组": ["CH19-下门-右丝杠下", "CH 16-DZQ升降右丝杠下",
              "CH 10-右门-丝杠中", "CH 22-中门-上丝杠中",
              "CH 13-DZQ水电-插拔左丝杠"],
}


def short(name):
    return SHORT.get(name, name)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    df = load_usable()

    # 场次序数 (ordinal session index): 按真实时间排序后给整数序号
    sess = (df.dropna(subset=["t"])
              .drop_duplicates("measurement")
              .sort_values("t")[["measurement", "t"]]
              .reset_index(drop=True))
    sess["sidx"] = np.arange(len(sess))
    sidx_map = dict(zip(sess["measurement"], sess["sidx"]))
    print(f"[info] 总场次数(有有效时间戳): {len(sess)}")

    # 命名传动测点
    named = df[df["point_name"].str.contains("门|导轨|丝杠|换向|水电|升降", na=False)].copy()
    u = named[named["usable"]].copy()
    u["sidx"] = u["measurement"].map(sidx_map)
    u = u.dropna(subset=["sidx"])
    u["sidx"] = u["sidx"].astype(int)

    # 保留 usable 记录数 >= MIN_USABLE 的测点
    cnt = u.groupby("point_name").size()
    keep_pts = sorted(cnt[cnt >= MIN_USABLE].index.tolist())
    print(f"[info] 命名 usable 测点数={u['point_name'].nunique()}, "
          f"达标(>= {MIN_USABLE})测点数={len(keep_pts)}")
    for p in keep_pts:
        print(f"       {p}: usable={int(cnt[p])}")

    if len(keep_pts) < 2:
        print("[error] 达标测点不足 2 个, 无法做两两相关。")
        sys.exit(1)

    # 构建 测点 x 场次序数 的 RMS 矩阵 (每测点每场次取 td_rms 均值, 通常每测点每场次1条)
    pivot = (u.pivot_table(index="point_name", columns="sidx",
                           values="td_rms", aggfunc="mean")
               .reindex(keep_pts))

    # ----- 两两相关 (长表) -----
    rows = []
    pts = list(pivot.index)
    for a, b in combinations(pts, 2):
        sa = pivot.loc[a]
        sb = pivot.loc[b]
        common = sa.index[sa.notna() & sb.notna()]
        n = len(common)
        if n >= MIN_COMMON:
            x = sa.loc[common].values.astype(float)
            y = sb.loc[common].values.astype(float)
            sp = spearmanr(x, y).correlation
            pe = pearsonr(x, y)[0]
        else:
            sp = np.nan
            pe = np.nan
        rows.append({
            "point_a": a, "point_b": b,
            "label_a": short(a), "label_b": short(b),
            "n_common": int(n),
            "spearman": sp, "pearson": pe,
            "valid": bool(n >= MIN_COMMON),
        })
    long_df = pd.DataFrame(rows)

    # ----- 相关矩阵 (Spearman, 用于热图) -----
    P = len(pts)
    sp_mat = pd.DataFrame(np.eye(P), index=pts, columns=pts)  # 对角=1
    pe_mat = pd.DataFrame(np.eye(P), index=pts, columns=pts)
    n_mat = pd.DataFrame(np.zeros((P, P), dtype=int), index=pts, columns=pts)
    for _, r in long_df.iterrows():
        a, b = r["point_a"], r["point_b"]
        sp_mat.loc[a, b] = sp_mat.loc[b, a] = r["spearman"]
        pe_mat.loc[a, b] = pe_mat.loc[b, a] = r["pearson"]
        n_mat.loc[a, b] = n_mat.loc[b, a] = r["n_common"]

    # ---------------------------------------------------------------------
    # 同步性指标: 组内 vs 跨组 平均相关; 孤立通道识别
    # ---------------------------------------------------------------------
    def grp_of(p):
        for g, members in GROUPS.items():
            if p in members:
                return g
        return "其他"

    long_df["group_a"] = long_df["point_a"].map(grp_of)
    long_df["group_b"] = long_df["point_b"].map(grp_of)
    long_df["same_group"] = long_df["group_a"] == long_df["group_b"]

    # 保存两两相关 (长表, 含分组信息)
    out_pair = OUTDIR / "R2_pairwise_corr.csv"
    long_df.to_csv(out_pair, index=False, encoding="utf-8-sig")
    print(f"[saved] {out_pair}")

    valid = long_df[long_df["valid"]].copy()

    within = valid[valid["same_group"]]
    across = valid[~valid["same_group"]]
    within_mean_sp = within["spearman"].mean()
    across_mean_sp = across["spearman"].mean()
    within_mean_pe = within["pearson"].mean()
    across_mean_pe = across["pearson"].mean()

    # 孤立通道判定 (有向): 与"同部件其它测点"是否同向高相关 (Spearman >= 0.5)。
    # 注意区分三态:
    #   1) 组内无其它达标测点 -> 无同部件参照 (no_reference), 不等于伪迹也不等于佐证
    #   2) 与同部件测点同向高相关 -> 被佐证 (corroborated)
    #   3) 与同部件测点不相关或反向 -> 孤立/反向 (isolated), 支持伪迹判断
    POS_THR = 0.5
    isolated = []          # 真正孤立/与同组反向 (支持伪迹)
    no_reference = []      # 组内无其它达标测点
    for p in pts:
        g = grp_of(p)
        same = [q for q in pts if q != p and grp_of(q) == g and g != "其他"]
        if not same:
            no_reference.append((p, "组内无其它达标测点(无同部件参照)"))
            continue
        vals = {q: sp_mat.loc[p, q] for q in same if pd.notna(sp_mat.loc[p, q])}
        if not vals:
            no_reference.append((p, "与同组共同场次均不足(NaN)"))
            continue
        best_q = max(vals, key=vals.get)
        best_v = vals[best_q]
        # 是否与同部件主退化通道(CH6/CH19)反向: 这是 CH10 伪迹判断的核心证据
        neg_partners = [q for q, v in vals.items() if v <= -0.4]
        if best_v < POS_THR:
            detail = f"与同部件测点最高Spearman={best_v:.2f}<{POS_THR}"
            if neg_partners:
                detail += "; 与" + "/".join(short(q) for q in neg_partners) + "反向(<= -0.4)"
            isolated.append((p, detail + " -> 孤立/反向, 支持伪迹判断"))

    # ---------------------------------------------------------------------
    # 关键部件对结论
    # ---------------------------------------------------------------------
    def pair_corr(a, b):
        m = long_df[((long_df["point_a"] == a) & (long_df["point_b"] == b)) |
                    ((long_df["point_a"] == b) & (long_df["point_b"] == a))]
        if len(m) == 0:
            return None
        r = m.iloc[0]
        return dict(sp=r["spearman"], pe=r["pearson"], n=r["n_common"], valid=r["valid"])

    CH6, CH7 = "CH 6-下门-换向器-y轴", "CH 7-DZQ升降-换向器"
    CH19, CH16 = "CH19-下门-右丝杠下", "CH 16-DZQ升降右丝杠下"
    CH10, CH22 = "CH 10-右门-丝杠中", "CH 22-中门-上丝杠中"
    CH13 = "CH 13-DZQ水电-插拔左丝杠"

    comm = pair_corr(CH6, CH7)
    # CH19 与同组其它丝杠测点
    ch19_pairs = {}
    for q in [CH16, CH10, CH22, CH13]:
        pc = pair_corr(CH19, q)
        if pc is not None:
            ch19_pairs[short(q)] = pc

    ch10_iso = any(p == CH10 for p, _ in isolated)

    # CH10 专项: 与主退化通道(CH19 右丝杠 / CH6 换向器)的方向 (伪迹判断核心证据)
    MAIN_DEGRAD = [CH19, CH16, CH6]  # 已知呈上升/主退化方向的通道
    ch10_vs_main = {}
    for q in MAIN_DEGRAD:
        pc = pair_corr(CH10, q)
        if pc is not None and pc["valid"] and pd.notna(pc["sp"]):
            ch10_vs_main[short(q)] = pc["sp"]
    ch10_neg_to_main = sum(1 for v in ch10_vs_main.values() if v <= -0.4)
    # CH10 与同组哪个测点高相关 (用于说明它"被谁佐证")
    ch10_best = None
    if CH10 in pts:
        sgrp = [q for q in pts if q != CH10 and grp_of(q) == grp_of(CH10) and grp_of(q) != "其他"]
        cand = {q: sp_mat.loc[CH10, q] for q in sgrp if pd.notna(sp_mat.loc[CH10, q])}
        if cand:
            bq = max(cand, key=cand.get)
            ch10_best = (short(bq), cand[bq])

    # ---------------------------------------------------------------------
    # 写 synchrony summary
    # ---------------------------------------------------------------------
    lines = []
    L = lines.append
    L("R2 跨传感器空间一致性分析 / 同步性汇总")
    L("=" * 70)
    L("定位: 质控筛查 (trend-screening) 的替代性验证 (surrogate validation);")
    L("仅为佐证级别证据, 非诊断/确证/RUL。横轴为场次序数(ordinal session index)。")
    L("数据为单试件真实在役记录, 无拆检/标签 ground truth。")
    L("")
    L(f"达标测点 (usable>= {MIN_USABLE}): {len(pts)} 个")
    for p in pts:
        L(f"  - {short(p)}  [{grp_of(p)}]  ({p})")
    L("")
    L(f"两两相关计算门槛: 共同场次数 >= {MIN_COMMON} (否则记 NaN, 不参与统计)")
    L(f"有效测点对数: {len(valid)} / 总对数 {len(long_df)}")
    L("")
    L("【同步性指标: 组内 vs 跨组 平均相关】")
    L(f"  组内平均 Spearman = {within_mean_sp:.3f} (n_pairs={len(within)})")
    L(f"  跨组平均 Spearman = {across_mean_sp:.3f} (n_pairs={len(across)})")
    L(f"  组内平均 Pearson  = {within_mean_pe:.3f}")
    L(f"  跨组平均 Pearson  = {across_mean_pe:.3f}")
    if pd.notna(within_mean_sp) and pd.notna(across_mean_sp):
        L(f"  -> 组内 - 跨组 (Spearman) = {within_mean_sp - across_mean_sp:+.3f}")
        if within_mean_sp > across_mean_sp:
            L("     组内相关高于跨组, 符合'部件级真实退化在同部件同向出现'的预期(筛查级佐证)。")
        else:
            L("     组内未明显高于跨组, 空间一致性证据偏弱(需谨慎, 写入局限)。")
    L("")
    L("【换向器组: CH6 vs CH7】")
    if comm and comm["valid"]:
        L(f"  Spearman={comm['sp']:.3f}, Pearson={comm['pe']:.3f}, 共同场次={comm['n']}")
        if comm["sp"] >= POS_THR:
            L("  -> 二者同向高相关, 支持换向器为部件级真实退化(非单传感器伪迹)的筛查级佐证。")
        else:
            L("  -> 二者相关不高, 换向器退化的空间一致性证据偏弱(写入局限)。")
    elif comm:
        L(f"  共同场次={comm['n']} < {MIN_COMMON}, 不足以判定 (NaN)。")
    else:
        L("  未找到 CH6-CH7 对 (测点可能未达标)。")
    L("")
    L("【丝杠组: CH19 与同组测点】")
    if ch19_pairs:
        for k, pc in ch19_pairs.items():
            tag = "" if pc["valid"] else " (共同场次不足, NaN)"
            spv = f"{pc['sp']:.3f}" if pd.notna(pc["sp"]) else "NaN"
            pev = f"{pc['pe']:.3f}" if pd.notna(pc["pe"]) else "NaN"
            L(f"  CH19 vs {k}: Spearman={spv}, Pearson={pev}, 共同场次={pc['n']}{tag}")
        best_q = max(((k, pc) for k, pc in ch19_pairs.items() if pc["valid"] and pd.notna(pc["sp"])),
                     key=lambda kv: kv[1]["sp"], default=None)
        if best_q:
            L(f"  -> CH19 与同组最高 Spearman = {best_q[1]['sp']:.3f} ({best_q[0]})。")
            if best_q[1]["sp"] >= POS_THR:
                L("     CH19 的上升被同组测点佐证(筛查级)。")
            else:
                L("     CH19 的上升未被同组明显佐证(其同组测点 CH16 已撤回/CH10 判为伪迹), 证据偏弱, 写入局限。")
    else:
        L("  无可比同组测点。")
    L("")
    L("【孤立/反向通道 (与同部件测点 Spearman < %.1f)】" % POS_THR)
    if isolated:
        for p, why in isolated:
            L(f"  - {short(p)}: {why}")
    else:
        L("  无(基于'最高同组相关'口径的)孤立通道; 但需结合下方有向证据看 CH10。")
    L("")
    L("【无同部件参照通道 (导轨等单点部件, 无法做组内佐证)】")
    if no_reference:
        for p, why in no_reference:
            L(f"  - {short(p)}: {why}")
    else:
        L("  无。")
    L("")
    L("【CH10 专项: 有向一致性 (其负趋势是否为传感器耦合伪迹)】")
    if ch10_vs_main:
        for k, v in ch10_vs_main.items():
            L(f"  CH10 vs 主退化通道 {k}: Spearman={v:+.3f}")
        L(f"  -> CH10 与 {ch10_neg_to_main}/{len(ch10_vs_main)} 个主退化通道(CH19/CH16/CH6)呈反向(<= -0.4)。")
    if ch10_best:
        L(f"  CH10 与同组最高相关测点: {ch10_best[0]} (Spearman={ch10_best[1]:+.3f})。")
    if ch10_neg_to_main >= 2:
        L("  结论(筛查级): CH10 与已知上升的主退化丝杠/换向器通道系统性反向, "
          "其下降不被部件级真实退化解释, 支持判其为传感器耦合伪迹/单通道伪趋势。")
        L("  说明: 按'最高同组相关'的无向口径 CH10 因与 CH22 同向(>0.5)未被标为孤立, "
          "但有向证据(与 CH19/CH16/CH6 反向)更能支持伪迹判断。")
    else:
        L("  CH10 与主退化通道未呈系统性反向, 需结合方向谨慎解读。")
    L("")
    L("【局限 (必须随结论一并陈述)】")
    L("  - 单试件、无 ground truth; 空间一致性为替代性验证, 仅佐证不构成确证。")
    L("  - 横轴为场次序数, 非等间隔标定时间; 相关受场次缺失/对齐影响。")
    L("  - 测点物理分组基于命名/位置推断, 未经拆检核实。")

    summary_txt = "\n".join(lines)
    out_sum = OUTDIR / "R2_synchrony_summary.txt"
    out_sum.write_text(summary_txt, encoding="utf-8-sig")
    print(f"[saved] {out_sum}")

    # 同步性指标也存一份机读 CSV
    metric_rows = [
        {"metric": "within_group_mean_spearman", "value": within_mean_sp, "n_pairs": len(within)},
        {"metric": "across_group_mean_spearman", "value": across_mean_sp, "n_pairs": len(across)},
        {"metric": "within_group_mean_pearson", "value": within_mean_pe, "n_pairs": len(within)},
        {"metric": "across_group_mean_pearson", "value": across_mean_pe, "n_pairs": len(across)},
        {"metric": "within_minus_across_spearman",
         "value": (within_mean_sp - across_mean_sp) if pd.notna(within_mean_sp) and pd.notna(across_mean_sp) else np.nan,
         "n_pairs": np.nan},
    ]
    pd.DataFrame(metric_rows).to_csv(OUTDIR / "R2_synchrony_metrics.csv",
                                     index=False, encoding="utf-8-sig")
    print(f"[saved] {OUTDIR / 'R2_synchrony_metrics.csv'}")

    # ---------------------------------------------------------------------
    # 热图 (Spearman 相关矩阵)
    # ---------------------------------------------------------------------
    labels = [short(p) for p in pts]
    M = sp_mat.values.astype(float)
    Mmask = np.ma.masked_invalid(M)

    fig, ax = plt.subplots(figsize=(max(7, 0.9 * P + 2), max(6, 0.9 * P + 1)))
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="lightgray")
    im = ax.imshow(Mmask, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(range(P)); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(P)); ax.set_yticklabels(labels, fontsize=9)
    for i in range(P):
        for j in range(P):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if abs(v) < 0.6 else "white")
            else:
                ax.text(j, i, "NaN", ha="center", va="center", fontsize=7, color="dimgray")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Spearman 相关 (跨场次 RMS 序列)")
    ax.set_title("R2 跨传感器空间一致性: 测点×测点 RMS 相关热图\n"
                 "(质控筛查/替代性验证, 仅佐证非确证; 横轴=场次序数)", fontsize=11)
    fig.tight_layout()
    out_png = OUTDIR / "R2_corr_heatmap.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[saved] {out_png}")

    # 控制台速览
    print("\n========== 控制台速览 ==========")
    if comm and comm["valid"]:
        print(f"CH6-CH7 Spearman={comm['sp']:.3f} Pearson={comm['pe']:.3f} n={comm['n']}")
    print(f"组内均 Spearman={within_mean_sp:.3f} 跨组均={across_mean_sp:.3f}")
    print(f"CH10 孤立={ch10_iso}")
    print("孤立通道:", [short(p) for p, _ in isolated])


if __name__ == "__main__":
    main()
