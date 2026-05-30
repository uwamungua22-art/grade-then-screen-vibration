# -*- coding: utf-8 -*-
"""F6 Fig4: R1 in-record window-level RMS distribution across sessions (English, journal style).
Reads R1_window_level_summary.csv and plots, for CH6/CH19, the per-session window-level RMS
median plus (p05, p95) band, revealing that degradation is accompanied by growing in-record
non-stationarity. Interpreter: python -X utf8"""
import os, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = r'./results'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                     'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 400,
                     'savefig.bbox': 'tight'})
CB = {'blue': '#0072B2', 'green': '#009E73', 'red': '#D55E00', 'grey': '#999999'}
CM = 1 / 2.54

df = pd.read_csv(os.path.join(OUT, 'R1_window_level_summary.csv'))
df['t'] = pd.to_datetime(df['t'])

panels = [('CH6 reversing/commutator unit', 'CH6 reversing/commutator', CB['green'], 7),
          ('CH19 ball screw R-low', 'CH19 right ball-screw', CB['blue'], 9)]
fig, axs = plt.subplots(1, 2, figsize=(17.6 * CM, 6.6 * CM))
for ax, (pt, title, c, cp) in zip(axs, panels):
    s = df[df.point_name == pt].sort_values('t').reset_index(drop=True)
    x = np.arange(len(s))
    ax.fill_between(x, s.rms_p05, s.rms_p95, color=c, alpha=0.25, lw=0,
                    label='window 5–95th pct')
    ax.plot(x, s.rms_p50, '-o', color=c, ms=3, lw=1.2, label='window median')
    ax.axvline(cp, color=CB['red'], ls=':', lw=1.0)
    ax.text(cp, 0.96, f' CP@{cp}', color=CB['red'], fontsize=6,
            va='top', ha='left', transform=ax.get_xaxis_transform())
    ax.set_xlabel('Session ordinal index (not calibrated time)')
    ax.set_ylabel('Within-record RMS (g)')
    ax.set_title(title, color=c)
    ax.legend(loc='upper left', frameon=False, fontsize=6)
fig.suptitle('In-record one-second sliding-window RMS distribution across sessions',
             fontsize=8.5, y=1.02)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT, f'F6_fig4_inrecord.{ext}'))
print('saved F6_fig4_inrecord.png/.pdf ->', OUT)
