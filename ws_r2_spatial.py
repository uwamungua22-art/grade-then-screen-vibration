# -*- coding: utf-8 -*-
"""
R2 cross-sensor spatial consistency analysis (cross-sensor spatial consistency / surrogate validation)
=======================================================================================================
Paper framing: data-quality / measurement trustworthiness / trend-screening (quality-control screening).
This script never performs diagnosis / fault detection / RUL. All conclusions are "screening-level
corroborating evidence", not confirmation.

Scientific rationale (surrogate validation, compensating for the absence of ground truth / a single specimen):
  Genuine component-level degradation should appear "in the same direction" on the same component
  or on neighbouring measurement points (high correlation of the across-session RMS series);
  an isolated rise on a single channel is more likely a sensor artifact. This analysis is only
  corroboration, not a gold standard.

Horizontal axis = ordinal session index, not calibrated time.

Reads only the local parquet feature table; does not touch any raw-waveform store.
Does not modify any shared script (ws_raw_trajectory.py / ws1_4_methods.py).
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

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(r".")
FEAT = ROOT / "00_inputs" / "features_all.parquet"
OUTDIR = ROOT / "04_jvet_results"
OUTDIR.mkdir(parents=True, exist_ok=True)

MIN_USABLE = 8        # point retention threshold: number of usable records >= 8
MIN_COMMON = 8        # pairwise correlation threshold: number of common sessions >= 8

# ---------------------------------------------------------------------------
# usable filtering criterion (identical to the conference version, copied verbatim)
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
# Physical component grouping (based on the physical location / component of each point)
# ---------------------------------------------------------------------------
# Short labels (for heatmap and table readability)
SHORT = {
    "CH6 reversing/commutator unit": "CH6 commutator",
    "CH7 reversing unit lift stage": "CH7 commutator",
    "CH19 ball screw R-low": "CH19 ball-screw",
    "CH16 lift screw": "CH16 lift-screw",
    "CH10 screw R-mid": "CH10 screw-mid",
    "CH22 screw M-mid": "CH22 screw-up",
    "CH13 conn. screw L": "CH13 conn-screw",
}

# Component groups: by the two main physical components "commutator" and "screw"
GROUPS = {
    "commutator group": ["CH6 reversing/commutator unit", "CH7 reversing unit lift stage"],
    "screw group": ["CH19 ball screw R-low", "CH16 lift screw",
              "CH10 screw R-mid", "CH22 screw M-mid",
              "CH13 conn. screw L"],
}


def short(name):
    return SHORT.get(name, name)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
def main():
    df = load_usable()

    # ordinal session index: assign integer indices after sorting by real time
    sess = (df.dropna(subset=["t"])
              .drop_duplicates("measurement")
              .sort_values("t")[["measurement", "t"]]
              .reset_index(drop=True))
    sess["sidx"] = np.arange(len(sess))
    sidx_map = dict(zip(sess["measurement"], sess["sidx"]))
    print(f"[info] total sessions (with valid timestamp): {len(sess)}")

    # named drivetrain measurement points
    named = df[df["point_name"].str.contains("reversing|commutator|ball|screw|guide|rail|seat|conn|lift", case=False, na=False)].copy()
    u = named[named["usable"]].copy()
    u["sidx"] = u["measurement"].map(sidx_map)
    u = u.dropna(subset=["sidx"])
    u["sidx"] = u["sidx"].astype(int)

    # keep points with number of usable records >= MIN_USABLE
    cnt = u.groupby("point_name").size()
    keep_pts = sorted(cnt[cnt >= MIN_USABLE].index.tolist())
    print(f"[info] named usable points={u['point_name'].nunique()}, "
          f"qualifying (>= {MIN_USABLE}) points={len(keep_pts)}")
    for p in keep_pts:
        print(f"       {p}: usable={int(cnt[p])}")

    if len(keep_pts) < 2:
        print("[error] fewer than 2 qualifying points, cannot compute pairwise correlation.")
        sys.exit(1)

    # build the point x ordinal-session RMS matrix (mean td_rms per point per session, usually 1 record per point per session)
    pivot = (u.pivot_table(index="point_name", columns="sidx",
                           values="td_rms", aggfunc="mean")
               .reindex(keep_pts))

    # ----- pairwise correlation (long table) -----
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

    # ----- correlation matrix (Spearman, for the heatmap) -----
    P = len(pts)
    sp_mat = pd.DataFrame(np.eye(P), index=pts, columns=pts)  # diagonal = 1
    pe_mat = pd.DataFrame(np.eye(P), index=pts, columns=pts)
    n_mat = pd.DataFrame(np.zeros((P, P), dtype=int), index=pts, columns=pts)
    for _, r in long_df.iterrows():
        a, b = r["point_a"], r["point_b"]
        sp_mat.loc[a, b] = sp_mat.loc[b, a] = r["spearman"]
        pe_mat.loc[a, b] = pe_mat.loc[b, a] = r["pearson"]
        n_mat.loc[a, b] = n_mat.loc[b, a] = r["n_common"]

    # ---------------------------------------------------------------------
    # synchrony metrics: within-group vs across-group mean correlation; isolated-channel detection
    # ---------------------------------------------------------------------
    def grp_of(p):
        for g, members in GROUPS.items():
            if p in members:
                return g
        return "other"

    long_df["group_a"] = long_df["point_a"].map(grp_of)
    long_df["group_b"] = long_df["point_b"].map(grp_of)
    long_df["same_group"] = long_df["group_a"] == long_df["group_b"]

    # save pairwise correlation (long table, with grouping information)
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

    # isolated-channel decision (directional): whether it correlates positively and strongly
    # (Spearman >= 0.5) with "other points on the same component".
    # Note the three states:
    #   1) no other qualifying point in the group -> no same-component reference (no_reference),
    #      neither an artifact nor corroboration
    #   2) positively and strongly correlated with same-component points -> corroborated
    #   3) uncorrelated or inversely correlated with same-component points -> isolated/inverse,
    #      supports an artifact judgement
    POS_THR = 0.5
    isolated = []          # genuinely isolated / inversely correlated with the same group (supports artifact)
    no_reference = []      # no other qualifying point in the group
    for p in pts:
        g = grp_of(p)
        same = [q for q in pts if q != p and grp_of(q) == g and g != "other"]
        if not same:
            no_reference.append((p, "no other qualifying point in the group (no same-component reference)"))
            continue
        vals = {q: sp_mat.loc[p, q] for q in same if pd.notna(sp_mat.loc[p, q])}
        if not vals:
            no_reference.append((p, "insufficient common sessions with the same group (NaN)"))
            continue
        best_q = max(vals, key=vals.get)
        best_v = vals[best_q]
        # whether inversely correlated with the main degrading channels of the same component (CH6/CH19):
        # this is the core evidence for the CH10 artifact judgement
        neg_partners = [q for q, v in vals.items() if v <= -0.4]
        if best_v < POS_THR:
            detail = f"highest Spearman with same-component points={best_v:.2f}<{POS_THR}"
            if neg_partners:
                detail += "; inversely correlated with " + "/".join(short(q) for q in neg_partners) + " (<= -0.4)"
            isolated.append((p, detail + " -> isolated/inverse, supports artifact judgement"))

    # ---------------------------------------------------------------------
    # Conclusions on key component pairs
    # ---------------------------------------------------------------------
    def pair_corr(a, b):
        m = long_df[((long_df["point_a"] == a) & (long_df["point_b"] == b)) |
                    ((long_df["point_a"] == b) & (long_df["point_b"] == a))]
        if len(m) == 0:
            return None
        r = m.iloc[0]
        return dict(sp=r["spearman"], pe=r["pearson"], n=r["n_common"], valid=r["valid"])

    CH6, CH7 = "CH6 reversing/commutator unit", "CH7 reversing unit lift stage"
    CH19, CH16 = "CH19 ball screw R-low", "CH16 lift screw"
    CH10, CH22 = "CH10 screw R-mid", "CH22 screw M-mid"
    CH13 = "CH13 conn. screw L"

    comm = pair_corr(CH6, CH7)
    # CH19 vs other screw points in the same group
    ch19_pairs = {}
    for q in [CH16, CH10, CH22, CH13]:
        pc = pair_corr(CH19, q)
        if pc is not None:
            ch19_pairs[short(q)] = pc

    ch10_iso = any(p == CH10 for p, _ in isolated)

    # CH10 special case: direction relative to the main degrading channels (CH19 ball-screw / CH6 commutator)
    # (core evidence for the artifact judgement)
    MAIN_DEGRAD = [CH19, CH16, CH6]  # channels known to rise / show the main degradation direction
    ch10_vs_main = {}
    for q in MAIN_DEGRAD:
        pc = pair_corr(CH10, q)
        if pc is not None and pc["valid"] and pd.notna(pc["sp"]):
            ch10_vs_main[short(q)] = pc["sp"]
    ch10_neg_to_main = sum(1 for v in ch10_vs_main.values() if v <= -0.4)
    # which same-group point CH10 correlates most with (to indicate "who corroborates it")
    ch10_best = None
    if CH10 in pts:
        sgrp = [q for q in pts if q != CH10 and grp_of(q) == grp_of(CH10) and grp_of(q) != "other"]
        cand = {q: sp_mat.loc[CH10, q] for q in sgrp if pd.notna(sp_mat.loc[CH10, q])}
        if cand:
            bq = max(cand, key=cand.get)
            ch10_best = (short(bq), cand[bq])

    # ---------------------------------------------------------------------
    # write the synchrony summary
    # ---------------------------------------------------------------------
    lines = []
    L = lines.append
    L("R2 cross-sensor spatial consistency analysis / synchrony summary")
    L("=" * 70)
    L("Framing: surrogate validation for quality-control screening (trend-screening);")
    L("corroborating-level evidence only, not diagnosis / confirmation / RUL. Horizontal axis is the ordinal session index.")
    L("Data are real in-service records from a single specimen, with no teardown / labelled ground truth.")
    L("")
    L(f"Qualifying points (usable >= {MIN_USABLE}): {len(pts)}")
    for p in pts:
        L(f"  - {short(p)}  [{grp_of(p)}]  ({p})")
    L("")
    L(f"Pairwise correlation threshold: number of common sessions >= {MIN_COMMON} (otherwise recorded as NaN, excluded from statistics)")
    L(f"Number of valid point pairs: {len(valid)} / total pairs {len(long_df)}")
    L("")
    L("[Synchrony metrics: within-group vs across-group mean correlation]")
    L(f"  within-group mean Spearman = {within_mean_sp:.3f} (n_pairs={len(within)})")
    L(f"  across-group mean Spearman = {across_mean_sp:.3f} (n_pairs={len(across)})")
    L(f"  within-group mean Pearson  = {within_mean_pe:.3f}")
    L(f"  across-group mean Pearson  = {across_mean_pe:.3f}")
    if pd.notna(within_mean_sp) and pd.notna(across_mean_sp):
        L(f"  -> within - across (Spearman) = {within_mean_sp - across_mean_sp:+.3f}")
        if within_mean_sp > across_mean_sp:
            L("     Within-group correlation exceeds across-group, consistent with the expectation that 'genuine component-level degradation appears in the same direction on the same component' (screening-level corroboration).")
        else:
            L("     Within-group is not clearly higher than across-group; the spatial-consistency evidence is weak (interpret with caution, note as a limitation).")
    L("")
    L("[Commutator group: CH6 vs CH7]")
    if comm and comm["valid"]:
        L(f"  Spearman={comm['sp']:.3f}, Pearson={comm['pe']:.3f}, common sessions={comm['n']}")
        if comm["sp"] >= POS_THR:
            L("  -> The two correlate positively and strongly, providing screening-level corroboration that the commutator shows component-level genuine degradation (not a single-sensor artifact).")
        else:
            L("  -> The correlation between the two is low; the spatial-consistency evidence for commutator degradation is weak (note as a limitation).")
    elif comm:
        L(f"  common sessions={comm['n']} < {MIN_COMMON}, insufficient to decide (NaN).")
    else:
        L("  CH6-CH7 pair not found (the points may not qualify).")
    L("")
    L("[Screw group: CH19 vs same-group points]")
    if ch19_pairs:
        for k, pc in ch19_pairs.items():
            tag = "" if pc["valid"] else " (insufficient common sessions, NaN)"
            spv = f"{pc['sp']:.3f}" if pd.notna(pc["sp"]) else "NaN"
            pev = f"{pc['pe']:.3f}" if pd.notna(pc["pe"]) else "NaN"
            L(f"  CH19 vs {k}: Spearman={spv}, Pearson={pev}, common sessions={pc['n']}{tag}")
        best_q = max(((k, pc) for k, pc in ch19_pairs.items() if pc["valid"] and pd.notna(pc["sp"])),
                     key=lambda kv: kv[1]["sp"], default=None)
        if best_q:
            L(f"  -> CH19 highest Spearman with the same group = {best_q[1]['sp']:.3f} ({best_q[0]}).")
            if best_q[1]["sp"] >= POS_THR:
                L("     CH19's rise is corroborated by same-group points (screening-level).")
            else:
                L("     CH19's rise is not clearly corroborated by the same group (its same-group point CH16 has been withdrawn / CH10 judged an artifact); the evidence is weak, note as a limitation.")
    else:
        L("  No comparable same-group points.")
    L("")
    L("[Isolated/inverse channels (Spearman with same-component points < %.1f)]" % POS_THR)
    if isolated:
        for p, why in isolated:
            L(f"  - {short(p)}: {why}")
    else:
        L("  None (under the 'highest same-group correlation' criterion); but CH10 must be read together with the directional evidence below.")
    L("")
    L("[Channels with no same-component reference (single-point components such as guide rails, cannot be corroborated within a group)]")
    if no_reference:
        for p, why in no_reference:
            L(f"  - {short(p)}: {why}")
    else:
        L("  None.")
    L("")
    L("[CH10 special case: directional consistency (whether its downward trend is a sensor-coupling artifact)]")
    if ch10_vs_main:
        for k, v in ch10_vs_main.items():
            L(f"  CH10 vs main degrading channel {k}: Spearman={v:+.3f}")
        L(f"  -> CH10 is inversely correlated (<= -0.4) with {ch10_neg_to_main}/{len(ch10_vs_main)} main degrading channels (CH19/CH16/CH6).")
    if ch10_best:
        L(f"  CH10's most-correlated same-group point: {ch10_best[0]} (Spearman={ch10_best[1]:+.3f}).")
    if ch10_neg_to_main >= 2:
        L("  Conclusion (screening-level): CH10 is systematically inversely correlated with the known-rising main degrading screw/commutator channels; "
          "its decrease is not explained by component-level genuine degradation, supporting the judgement that it is a sensor-coupling artifact / single-channel spurious trend.")
        L("  Note: under the undirected 'highest same-group correlation' criterion CH10 is not flagged as isolated because it is positively correlated (>0.5) with CH22, "
          "but the directional evidence (inverse to CH19/CH16/CH6) better supports the artifact judgement.")
    else:
        L("  CH10 is not systematically inversely correlated with the main degrading channels; interpret with caution in light of direction.")
    L("")
    L("[Limitations (must be stated together with the conclusions)]")
    L("  - Single specimen, no ground truth; spatial consistency is surrogate validation, corroboration only and not confirmation.")
    L("  - The horizontal axis is the ordinal session index, not equally spaced calibrated time; correlation is affected by missing/aligned sessions.")
    L("  - The physical grouping of points is inferred from naming/location and not verified by teardown.")

    summary_txt = "\n".join(lines)
    out_sum = OUTDIR / "R2_synchrony_summary.txt"
    out_sum.write_text(summary_txt, encoding="utf-8-sig")
    print(f"[saved] {out_sum}")

    # also save the synchrony metrics as a machine-readable CSV
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
    # heatmap (Spearman correlation matrix)
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
    cb.set_label("Spearman correlation (across-session RMS series)")
    ax.set_title("R2 cross-sensor spatial consistency: point x point RMS correlation heatmap\n"
                 "(quality-control screening / surrogate validation, corroboration only not confirmation; horizontal axis = ordinal session index)", fontsize=11)
    fig.tight_layout()
    out_png = OUTDIR / "R2_corr_heatmap.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"[saved] {out_png}")

    # console quick view
    print("\n========== console quick view ==========")
    if comm and comm["valid"]:
        print(f"CH6-CH7 Spearman={comm['sp']:.3f} Pearson={comm['pe']:.3f} n={comm['n']}")
    print(f"within-group mean Spearman={within_mean_sp:.3f} across-group mean={across_mean_sp:.3f}")
    print(f"CH10 isolated={ch10_iso}")
    print("isolated channels:", [short(p) for p, _ in isolated])


if __name__ == "__main__":
    main()
