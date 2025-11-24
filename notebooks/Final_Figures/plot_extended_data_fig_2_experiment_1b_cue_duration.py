#!/usr/bin/env python
# coding: utf-8
try:
    get_ipython  # type: ignore
except NameError:
    def get_ipython():
        class _DummyShell:
            def run_line_magic(self, *args, **kwargs):
                pass

            def run_cell_magic(self, *args, **kwargs):
                pass

            def magic(self, *args, **kwargs):
                pass

        return _DummyShell()

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from final_figures_paths import DATA_ROOT


parser = argparse.ArgumentParser(
    description="Extended Data Figure 2: effect of cue duration (human vs model)"
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Skip writing figures to disk",
)
parser.add_argument(
    "--fig-dir",
    default="rebuttal_figs",
    help="Output directory for figures",
)
args = parser.parse_args()
DRY_RUN = args.dry_run

fig_out_dir = Path(args.fig_dir)
fig_out_dir.mkdir(parents=True, exist_ok=True)


# Load data
fname = DATA_ROOT / "experiment_1b_cue_duration_data_n_85_with_feature_gain_model.csv"
to_plot = pd.read_csv(fname)


# Main cue duration effect plot (human + model)
fig, ax = plt.subplots(1, figsize=(2.5, 2.5))
sns.lineplot(
    data=to_plot,
    x="cue_dur",
    y="adjusted_accuracy",
    style="group",
    hue="group",
    marker="",
    color="k",
    err_style="bars",
    errorbar=("se", 1),
    ax=ax,
)

ax.set_xlabel("Cue duration (s)")
ax.set_ylabel("Prop. target word")
sns.despine(ax=ax)
ax.set_xticks(ticks=sorted(to_plot.cue_dur.unique()))
ax.set_xticklabels(sorted(to_plot.cue_dur.unique()))
ax.legend(title="", frameon=False)
ax.set_ylim(0, 1)
ax.set_title("Effect of cue duration", y=1.05)

if not DRY_RUN:
    fig.savefig(
        fig_out_dir / "cue_duration_effect_human_model.pdf",
        bbox_inches="tight",
        transparent=True,
    )



