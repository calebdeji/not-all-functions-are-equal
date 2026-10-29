"""
plot_sdl_weights.py

Two publication-ready figures for the interpretability case study:

  Figure 1 — Scatter (per APK, 3 panels):
    x = sim_max   (intra-class semantic alignment)
    y = sim_others (inter-class semantic alignment)
    colour = weight  (viridis)
    size = norm_complexity
    Top-10 functions labelled.

  Figure 2 — Ranked horizontal bars (per APK, 3 panels):
    Top-10 functions ranked by weight.
    Four grouped bars per function:
      weight | sim_max | sim_others | norm_complexity

Reads from:  results/interpretability/weights_*.csv
Writes to:   results/interpretability/fig_weights_scatter.pdf
             results/interpretability/fig_weights_bars.pdf
             (also .png at 300 dpi)

Usage:
  python -m src.analysis.plot_sdl_weights
  python -m src.analysis.plot_sdl_weights --top_n 10 --families drosel fakeadblocker fakeapp
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.core.utils import Utils

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Utils.get_project_root()
OUT_DIR      = os.path.join(PROJECT_ROOT, 'results', 'interpretability')

PALETTE = {
    'weight':           '#2171B5',
    'sim_max':          '#41AB5D',
    'sim_others':       '#CB181D',
    'norm_complexity':  '#FD8D3C',
}

FAMILY_ORDER = ['drosel', 'fakeadblocker', 'triada']
FAMILY_LABELS = {
    'drosel':        'Drosel (best)',
    'fakeadblocker': 'FakeAdblocker (best)',
    'triada':        'Triada (worst)',
}


def parse_dalvik(sig, max_len=32):
    """
    Lcom/rice/a/b;->LoadKit(...)V  →  b.LoadKit
    Falls back gracefully for unusual signatures.
    """
    try:
        cls_part, rest = sig.split(';->', 1)
        cls_name  = cls_part.lstrip('L')
        segments  = cls_name.split('/')
        method    = rest.split('(')[0]

        # Use last two class segments (skip single-letter obfuscated ones if possible)
        meaningful = [s for s in segments if len(s) > 1]
        if meaningful:
            cls_label = '.'.join(meaningful[-2:]) if len(meaningful) >= 2 else meaningful[-1]
        else:
            cls_label = '.'.join(segments[-2:]) if len(segments) >= 2 else segments[-1]

        label = f'{cls_label}.{method}'
        if len(label) > max_len:
            label = f'{segments[-1]}.{method}'
        if len(label) > max_len:
            label = label[:max_len - 2] + '..'
        return label
    except Exception:
        return sig[:max_len]


def load_csvs(families):
    """Return {family: DataFrame} sorted by rank ascending."""
    dfs = {}
    for family in families:
        pattern = os.path.join(OUT_DIR, f'weights_{family}_*.csv')
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f'[WARN] No CSV found for family "{family}" — skipping.')
            continue
        df = pd.read_csv(matches[-1])
        df['short_name'] = df['function_name'].apply(parse_dalvik)
        dfs[family] = df.sort_values('rank')
    return dfs


# ---------------------------------------------------------------------------
# Figure 1 — Scatter
# ---------------------------------------------------------------------------

def plot_scatter(dfs, top_n, out_prefix):
    families = [f for f in FAMILY_ORDER if f in dfs]
    ncols = len(families)
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5.2),
                             constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    vmin, vmax = 0.15, 0.55   # weight range across all three APKs

    for ax, family in zip(axes, families):
        df    = dfs[family]
        top   = df[df['rank'] <= top_n]
        rest  = df[df['rank'] > top_n]

        # All non-top functions — light grey background
        ax.scatter(rest['sim_max'], rest['sim_others'],
                   c='#CCCCCC', s=14, alpha=0.45, linewidths=0, zorder=1)

        # Top-N functions — coloured by weight
        sizes = 60 + rest['norm_complexity'].clip(0, 1) * 80   # irrelevant here, just background
        sc = ax.scatter(top['sim_max'], top['sim_others'],
                        c=top['weight'], cmap='RdYlGn',
                        vmin=vmin, vmax=vmax,
                        s=80 + top['norm_complexity'].clip(0, 1) * 120,
                        edgecolors='#333333', linewidths=0.6,
                        zorder=3)

        # Labels for top-N
        for _, row in top.iterrows():
            ax.annotate(
                row['short_name'],
                xy=(row['sim_max'], row['sim_others']),
                xytext=(4, 3), textcoords='offset points',
                fontsize=6.5, color='#1A1A1A',
                zorder=4,
            )

        ax.set_xlabel('sim_max  (intra-class alignment)', fontsize=10)
        ax.set_ylabel('sim_others  (inter-class alignment)', fontsize=10)
        ax.set_title(FAMILY_LABELS.get(family, family), fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=8)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.02, 0.45)

        # Diagonal guide: functions below this line are distinctive (sim_max > sim_others)
        lim = ax.get_xlim()
        ax.plot(lim, lim, ls='--', lw=0.8, color='#AAAAAA', zorder=0)
        ax.text(0.72, 0.74, 'sim_max = sim_others', fontsize=6,
                color='#AAAAAA', rotation=45, transform=ax.transData)

    # Shared colourbar
    cbar = fig.colorbar(sc, ax=axes, orientation='vertical',
                        fraction=0.025, pad=0.01, shrink=0.85)
    cbar.set_label('SDL weight  w(v)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Legend for point size = norm_complexity
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888',
               markersize=6,  label='c_norm ≈ 0'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888',
               markersize=10, label='c_norm ≈ 1'),
        Line2D([0], [0], marker='o', color='#CCCCCC', markersize=6,
               label='rank > top-N', linestyle='None'),
    ]
    axes[0].legend(handles=legend_elements, fontsize=7, loc='upper left',
                   framealpha=0.8)

    fig.suptitle(
        f'SDL-Complexity: function weight decomposition  '
        f'(top-{top_n} labelled, size ∝ norm_complexity)',
        fontsize=11, y=1.01,
    )

    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'{out_prefix}_scatter.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Grouped horizontal bars
# ---------------------------------------------------------------------------

def plot_bars(dfs, top_n, out_prefix, bottom_n=5):
    families = [f for f in FAMILY_ORDER if f in dfs]
    nrows = len(families)
    n_rows_per_apk = top_n + bottom_n + 1   # +1 gap row for separator

    fig, axes = plt.subplots(nrows, 1,
                             figsize=(10, (top_n + bottom_n) * 0.42 * nrows + 1.2 * nrows),
                             constrained_layout=True)
    if nrows == 1:
        axes = [axes]

    metrics = ['weight', 'sim_max', 'sim_others', 'norm_complexity']
    labels  = ['weight  w(v)', 'sim_max', 'sim_others', 'norm_complexity']
    colors  = [PALETTE[m] for m in metrics]
    n_met   = len(metrics)
    bar_h   = 0.18
    offsets = np.array([-(n_met / 2 - i - 0.5) * bar_h for i in range(n_met)])
    gap     = 1.2   # y-gap between top block and bottom block

    for ax, family in zip(axes, families):
        df  = dfs[family]

        # Top-N: rank 1..top_n, drawn at y = top_n-1 down to 0
        top = df[df['rank'] <= top_n].sort_values('rank', ascending=False)
        y_top = np.arange(len(top))                       # top_n-1 … 0

        # Bottom-N: last N by rank, drawn below a gap
        bot = df.sort_values('rank', ascending=False).head(bottom_n)   # lowest weights first
        y_bot = np.arange(len(bot)) - gap - len(bot)     # negative y, below gap

        for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
            ax.barh(y_top + offsets[i], top[metric], height=bar_h,
                    color=color, label=label, alpha=0.88,
                    edgecolor='white', linewidth=0.3)
            ax.barh(y_bot + offsets[i], bot[metric], height=bar_h,
                    color=color, alpha=0.55,              # slightly muted for bottom block
                    edgecolor='white', linewidth=0.3)

        # Y-tick labels
        all_y      = np.concatenate([y_top, y_bot])
        all_labels = list(top['short_name']) + list(bot['short_name'])
        ax.set_yticks(all_y)
        ax.set_yticklabels(all_labels, fontsize=8)

        # Rank labels on the right
        for j, (_, row) in enumerate(top.iterrows()):
            ax.text(1.01, y_top[j], f'#{int(row["rank"])}',
                    va='center', fontsize=7, color='#333333')
        for j, (_, row) in enumerate(bot.iterrows()):
            ax.text(1.01, y_bot[j], f'#{int(row["rank"])}',
                    va='center', fontsize=7, color='#888888')

        # Section labels
        ax.text(-0.01, y_top[0] + 0.6, 'highest weighted',
                ha='right', va='bottom', fontsize=7.5,
                color='#333333', fontstyle='italic',
                transform=ax.get_yaxis_transform())
        ax.text(-0.01, y_bot[-1] - 0.4, 'lowest weighted',
                ha='right', va='top', fontsize=7.5,
                color='#888888', fontstyle='italic',
                transform=ax.get_yaxis_transform())

        # Dashed separator
        sep_y = (y_top[-1] + y_bot[0]) / 2
        ax.axhline(sep_y, color='#AAAAAA', linestyle='--', linewidth=0.8)

        ax.set_xlabel('Value', fontsize=10)
        ax.set_xlim(0, 1.05)
        ax.set_title(FAMILY_LABELS.get(family, family), fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', labelsize=8)
        ax.xaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

        if family == families[0]:
            ax.legend(loc='lower right', fontsize=8, framealpha=0.85)

    fig.suptitle(
        f'SDL-Complexity: top-{top_n} and bottom-{bottom_n} weighted functions per APK  '
        f'(weight | sim_max | sim_others | norm_complexity)',
        fontsize=11,
    )

    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'{out_prefix}_bars.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_n',    type=int, default=10,
                        help='Number of top-weighted functions to highlight/plot.')
    parser.add_argument('--bottom_n', type=int, default=5,
                        help='Number of bottom-weighted functions to include in bar chart.')
    parser.add_argument('--families', nargs='+',
                        default=['drosel', 'fakeadblocker', 'triada'],
                        help='Families to include (must match CSV filename prefixes).')
    parser.add_argument('--prefix',   default='fig_weights',
                        help='Output filename prefix.')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    dfs = load_csvs(args.families)
    if not dfs:
        raise RuntimeError('No CSV files loaded. Run extract_sdl_weights.py first.')

    print(f'Loaded {len(dfs)} APK(s): {list(dfs.keys())}')
    for fam, df in dfs.items():
        print(f'  {fam}: {len(df)} functions, top-{args.top_n} plotted')

    plot_scatter(dfs, args.top_n, args.prefix)
    plot_bars(dfs, args.top_n, args.prefix, bottom_n=args.bottom_n)
    print('\nDone.')


if __name__ == '__main__':
    main()
