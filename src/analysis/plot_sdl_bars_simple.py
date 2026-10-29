"""
plot_sdl_bars_simple.py

Horizontal bar chart for the interpretability case study.
Shows top-N and bottom-N weighted functions per APK with three metrics:
  weight | sim_max | sim_others
(norm_complexity excluded)

Reads from:  results/interpretability/weights_*.csv
Writes to:   results/interpretability/fig_bars_simple.pdf
             results/interpretability/fig_bars_simple.png

Usage:
  python -m src.analysis.plot_sdl_bars_simple
  python -m src.analysis.plot_sdl_bars_simple --top_n 10 --bottom_n 5
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core.utils import Utils

PROJECT_ROOT = Utils.get_project_root()
OUT_DIR      = os.path.join(PROJECT_ROOT, 'results', 'interpretability')

FAMILY_ORDER = ['drosel', 'fakeadblocker', 'triada']
FAMILY_LABELS = {
    'drosel':        'Drosel (high-accuracy)',
    'fakeadblocker': 'FakeAdblocker (high-accuracy)',
    'triada':        'Triada (low-accuracy)',
}

METRICS = ['weight', 'sim_max', 'sim_others']
LABELS  = ['weight  w(v)', 'sim_max', 'sim_others']
COLORS  = ['#2171B5', '#41AB5D', '#CB181D']


def parse_dalvik(sig, max_len=36):
    try:
        cls_part, rest = sig.split(';->', 1)
        cls_name  = cls_part.lstrip('L')
        segments  = cls_name.split('/')
        method    = rest.split('(')[0]
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
    dfs = {}
    for family in families:
        pattern = os.path.join(OUT_DIR, f'weights_{family}_*.csv')
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f'[WARN] No CSV for "{family}" — skipping.')
            continue
        df = pd.read_csv(matches[-1])
        df['short_name'] = df['function_name'].apply(parse_dalvik)
        dfs[family] = df.sort_values('rank')
    return dfs


def plot_bars(dfs, top_n, bottom_n, out_prefix):
    families = [f for f in FAMILY_ORDER if f in dfs]
    nrows = len(families)

    bar_h   = 0.22
    n_met   = len(METRICS)
    offsets = np.array([-(n_met / 2 - i - 0.5) * bar_h for i in range(n_met)])
    gap     = 1.4

    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(11, (top_n + bottom_n) * 0.46 * nrows + 1.4 * nrows),
        constrained_layout=True,
    )
    if nrows == 1:
        axes = [axes]

    for ax, family in zip(axes, families):
        df  = dfs[family]

        top = df[df['rank'] <= top_n].sort_values('rank', ascending=False)
        bot = df.sort_values('rank', ascending=False).head(bottom_n)

        y_top = np.arange(len(top), dtype=float)
        y_bot = np.arange(len(bot), dtype=float) - gap - len(bot)

        for i, (metric, label, color) in enumerate(zip(METRICS, LABELS, COLORS)):
            ax.barh(y_top + offsets[i], top[metric], height=bar_h,
                    color=color, label=label, alpha=0.90,
                    edgecolor='white', linewidth=0.3)
            ax.barh(y_bot + offsets[i], bot[metric], height=bar_h,
                    color=color, alpha=0.55,
                    edgecolor='white', linewidth=0.3)

        # Y-tick labels
        ax.set_yticks(np.concatenate([y_top, y_bot]))
        ax.set_yticklabels(list(top['short_name']) + list(bot['short_name']), fontsize=8.5)

        # Rank annotations on the right
        for j, (_, row) in enumerate(top.iterrows()):
            ax.text(1.01, y_top[j], f'#{int(row["rank"])}',
                    va='center', fontsize=7, color='#333333')
        for j, (_, row) in enumerate(bot.iterrows()):
            ax.text(1.01, y_bot[j], f'#{int(row["rank"])}',
                    va='center', fontsize=7, color='#999999')

        # Section labels
        ax.text(-0.01, y_top[0] + 0.55, 'highest weighted',
                ha='right', va='bottom', fontsize=7.5,
                color='#333333', fontstyle='italic',
                transform=ax.get_yaxis_transform())
        ax.text(-0.01, y_bot[-1] - 0.45, 'lowest weighted',
                ha='right', va='top', fontsize=7.5,
                color='#999999', fontstyle='italic',
                transform=ax.get_yaxis_transform())

        # Dashed separator
        sep_y = (y_top[-1] + y_bot[0]) / 2
        ax.axhline(sep_y, color='#BBBBBB', linestyle='--', linewidth=0.9)

        ax.set_xlim(0, 1.08)
        ax.set_xlabel('Value', fontsize=10)
        ax.set_title(FAMILY_LABELS.get(family, family), fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', labelsize=8)
        ax.xaxis.grid(True, linestyle='--', alpha=0.35)
        ax.set_axisbelow(True)

        if family == families[0]:
            ax.legend(loc='lower right', fontsize=9, framealpha=0.85)

    fig.suptitle(
        f'SDL-Complexity: top-{top_n} and bottom-{bottom_n} weighted functions per APK',
        fontsize=11,
    )

    for ext in ('pdf', 'png'):
        path = os.path.join(OUT_DIR, f'{out_prefix}.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Saved: {path}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_n',    type=int,   default=10)
    parser.add_argument('--bottom_n', type=int,   default=5)
    parser.add_argument('--families', nargs='+',  default=FAMILY_ORDER)
    parser.add_argument('--prefix',   default='fig_bars_simple')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    dfs = load_csvs(args.families)
    if not dfs:
        raise RuntimeError('No CSV files loaded. Run extract_sdl_weights.py first.')

    print(f'Loaded: {list(dfs.keys())}')
    plot_bars(dfs, args.top_n, args.bottom_n, args.prefix)
    print('Done.')


if __name__ == '__main__':
    main()
