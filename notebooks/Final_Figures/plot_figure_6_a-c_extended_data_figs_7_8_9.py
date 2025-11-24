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
from copy import deepcopy
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src import util_analysis


parser = argparse.ArgumentParser(
    description="Figure 6a–c and Extended Data Figures 7–9: human–model similarity summary"
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Skip writing figures to disk",
)
parser.add_argument(
    "--fig-dir",
    default="rebuttal_figs/figure_6",
    help="Output directory for figures",
)
args = parser.parse_args()
DRY_RUN = args.dry_run

fig_out_dir = Path(args.fig_dir)
fig_out_dir.mkdir(parents=True, exist_ok=True)


matplotlib.rcParams.update({"font.size": 10})
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"


# Import pre-formatted data
results_dir = Path("data")

# Diotic SWC
diotic_results = pd.read_csv(results_dir / "experiment_1_df.csv")
diotic_results["experiment"] = "Diotic"

# Popham SWC
popham_results = pd.read_pickle(
    results_dir
    / "df_for_stats_2024_SWC_popham_conditions_humans_N-90_models_v10_w_control_archs.pdpkl"
)
popham_results.loc[popham_results.group.str.contains("Human"), "model"] = (
    popham_results.loc[popham_results.group.str.contains("Human"), "group"]
)
popham_results["snr"] = 0
popham_conds_to_keep = [
    "Harmonic_target_Harmonic_distractor",
    "Harmonic_target_No Distractor_distractor",
    "Inharmonic_target_Inharmonic_distractor",
    "Inharmonic_target_No Distractor_distractor",
    "Whispered_target_No Distractor_distractor",
    "Whispered_target_Whispered_distractor",
]
popham_results = popham_results[
    popham_results["background_condition"].isin(popham_conds_to_keep)
].reset_index(drop=True)
popham_results["experiment"] = "Harmonicity"

# Threshold results
threshold_results = pd.read_pickle(
    results_dir
    / "df_for_stats_and_summary_2024_thresholds_humans_N-33_models_v10_w_control_archs.pdpkl"
)
threshold_results["background_condition"] = (
    threshold_results["azim_delta"].astype("str")
    + " azim delta "
    + threshold_results["elev_delta"].astype("str")
    + " elev delta"
)
threshold_results["experiment"] = "Threshold"
threshold_results.rename(
    columns={"accuracy_sem": "acc_sem", "confusions_sem": "conf_sem"}, inplace=True
)

# Spotlight results
spotlight_results = pd.read_pickle(
    results_dir
    / "df_for_stats_and_summary_2024_spotlight_humans_N-28_models_v10_w_control_archs.pdpkl"
)
spotlight_results["snr"] = 0
spotlight_results["background_condition"] = (
    spotlight_results["target_azim"].astype("str")
    + " target azim "
    + spotlight_results["azim_delta"].astype("str")
    + " azim delta"
)
spotlight_results["experiment"] = "Spotlight"
spotlight_results.rename(
    columns={"accuracy_sem": "acc_sem", "confusions_sem": "conf_sem"}, inplace=True
)


combined_results = pd.concat(
    [diotic_results, popham_results, threshold_results, spotlight_results],
    axis=0,
)

combined_results.loc[combined_results["model"].str.contains("early"), "model"] = (
    "Early-only"
)
combined_results.loc[combined_results["model"].str.contains("late"), "model"] = (
    "Late-only"
)
combined_results.loc[combined_results["model"].str.contains("control"), "model"] = (
    "Baseline CNN"
)

combined_results["model"] = combined_results["model"].apply(
    util_analysis.get_model_name
)


# Add norm-before-gain model
outfile = results_dir / "norm_then_gain_model_summary_all_experiments.csv"
norm_then_gain_df = pd.read_csv(outfile)
combined_results = pd.concat(
    [combined_results, norm_then_gain_df],
    ignore_index=True,
    axis=0,
)


combined_results["snr_condition_str"] = (
    combined_results["snr"].astype(str)
    + " dB "
    + combined_results["background_condition"]
)
human_results = combined_results[combined_results.group.str.contains("Human")]
human_results = human_results.sort_values(["snr", "background_condition"])

model_list = [
    model for model in combined_results.model.unique() if "Human" not in model
]


# Human–model similarity (r^2 and RMSE) per model
model_sim_records = []
for model in model_list:
    model_results = combined_results[combined_results.model == model]
    model_results = model_results.sort_values(["snr", "background_condition"])

    r, _ = stats.pearsonr(human_results.accuracy, model_results.accuracy)
    acc_r = r**2
    acc_rmse = np.sqrt(
        np.mean(
            (human_results.accuracy.values - model_results.accuracy.values) ** 2
        )
    )

    r, _ = stats.pearsonr(human_results.confusions, model_results.confusions)
    conf_r = r**2
    conf_rmse = np.sqrt(
        np.mean(
            (human_results.confusions.values - model_results.confusions.values) ** 2
        )
    )

    record = {
        "model": model,
        "acc_r": acc_r,
        "acc_rmse": acc_rmse,
        "conf_r": conf_r,
        "conf_rmse": conf_rmse,
    }
    model_sim_records.append(record)

model_sim_df = pd.DataFrame.from_records(model_sim_records)


# Sign tests comparing Feature-gain vs alternative models
from src.util_analysis import bootstrap_sign_test_summary, sign_test

GLOBAL_MODEL_ORDER_LIST = [
    "Early-only",
    "Late-only",
    "Baseline CNN",
    "Norm before gain model",
]

fba_model_r_acc_dist = model_sim_df.loc[
    model_sim_df.model.str.contains("main|gain"), "acc_r"
].values
fba_model_r_conf_dist = model_sim_df.loc[
    model_sim_df.model.str.contains("main|gain"), "conf_r"
].values

for model in GLOBAL_MODEL_ORDER_LIST:
    y_acc = model_sim_df.loc[model_sim_df.model == model, "acc_r"].values
    stats_result = sign_test(fba_model_r_acc_dist, mu0=y_acc)
    print(
        f"Feature-gain v {model}, Pearson's r^2 accuracy sign test stat={stats_result[0]} p={stats_result[1]:.5f}"
    )
    y_conf = model_sim_df.loc[model_sim_df.model == model, "conf_r"].values
    stats_result = sign_test(fba_model_r_conf_dist, mu0=y_conf)
    print(
        f"Feature-gain v {model}, Pearson's r^2 confusions sign test stat={stats_result[0]} p={stats_result[1]:.5f}"
    )
    print("")

fba_model_r_acc_dist = model_sim_df.loc[
    model_sim_df.model.str.contains("main|gain"), "acc_rmse"
].values
fba_model_r_conf_dist = model_sim_df.loc[
    model_sim_df.model.str.contains("main|gain"), "conf_rmse"
].values

for model in GLOBAL_MODEL_ORDER_LIST:
    y_acc = model_sim_df.loc[model_sim_df.model == model, "acc_rmse"].values
    stats_result = sign_test(fba_model_r_acc_dist, mu0=y_acc)
    print(
        f"Feature-gain v {model}, RMSE accuracy sign test stat={stats_result[0]} p={stats_result[1]:.5f}"
    )
    y_conf = model_sim_df.loc[model_sim_df.model == model, "conf_rmse"].values
    stats_result = sign_test(fba_model_r_conf_dist, mu0=y_conf)
    print(
        f"Feature-gain v {model}, RMSE confusions sign test stat={stats_result[0]} p={stats_result[1]:.5f}"
    )
    print("")


# Pooled similarity across accuracy + confusions (Extended Data)
pooled_results = combined_results.melt(
    id_vars=["snr", "background_condition", "model"],
    value_vars=["accuracy", "confusions"],
    var_name="metric",
    value_name="measure",
)
pooled_results["condition_str"] = (
    pooled_results["snr"].astype(str)
    + " dB "
    + pooled_results["background_condition"]
    + " "
    + pooled_results["metric"]
)

human_results_pooled = pooled_results[pooled_results.model.str.contains("Human")]
human_results_pooled = human_results_pooled.sort_values("condition_str")
model_list_pooled = [
    model for model in pooled_results.model.unique() if "Human" not in model
]

model_sim_records_pooled = []
for model in model_list_pooled:
    model_results = pooled_results[pooled_results.model == model]
    model_results = model_results.sort_values("condition_str")

    r, _ = stats.pearsonr(human_results_pooled.measure, model_results.measure)
    r2 = r**2
    rmse = np.sqrt(
        np.mean(
            (human_results_pooled.measure.values - model_results.measure.values) ** 2
        )
    )
    record = {"model": model, "r2": r2, "rmse": rmse}
    model_sim_records_pooled.append(record)

model_sim_df_pooled = pd.DataFrame.from_records(model_sim_records_pooled)


# Bootstrap sign tests with CIs (Extended Data)
sign_test_dict = []

fba_model_r_dist = model_sim_df_pooled.loc[
    model_sim_df_pooled.model.str.contains("main|alt"), "r2"
].values
fba_model_rmse_dist = model_sim_df_pooled.loc[
    model_sim_df_pooled.model.str.contains("main|alt"), "rmse"
].values

comparison_r2 = {
    model: model_sim_df_pooled.loc[model_sim_df_pooled.model == model, "r2"].values
    for model in GLOBAL_MODEL_ORDER_LIST
}
comparison_rmse = {
    model: model_sim_df_pooled.loc[model_sim_df_pooled.model == model, "rmse"].values
    for model in GLOBAL_MODEL_ORDER_LIST
}

bootstrap_kwargs = dict(n_bootstrap=1000, random_state=42)

r2_bootstrap_results = bootstrap_sign_test_summary(
    baseline_values=fba_model_r_dist,
    comparison_values=comparison_r2,
    **bootstrap_kwargs,
)
rmse_bootstrap_results = bootstrap_sign_test_summary(
    baseline_values=fba_model_rmse_dist,
    comparison_values=comparison_rmse,
    **bootstrap_kwargs,
)

for model in GLOBAL_MODEL_ORDER_LIST:
    r2_summary = r2_bootstrap_results[model]
    rmse_summary = rmse_bootstrap_results[model]

    record = {
        "model": model,
        "r2_diff": r2_summary["diff_mean"],
        "r2_sign_test_stat": r2_summary["sign_test_stat"],
        "r2_n_pos": r2_summary["n_pos"],
        "r2_n_neg": r2_summary["n_neg"],
        "r2_n_total": r2_summary["n_total"],
        "r2_sign_test_p": r2_summary["sign_test_p"],
        "r2_diff_of_mean": r2_summary["diff_mean"],
        "r2_diff_ci_low": r2_summary["boot_ci_low"],
        "r2_diff_ci_high": r2_summary["boot_ci_high"],
        "rmse_diff": rmse_summary["diff_mean"],
        "rmse_sign_test_stat": rmse_summary["sign_test_stat"],
        "rmse_n_pos": rmse_summary["n_pos"],
        "rmse_n_neg": rmse_summary["n_neg"],
        "rmse_n_total": rmse_summary["n_total"],
        "rmse_sign_test_p": rmse_summary["sign_test_p"],
        "rmse_diff_of_mean": rmse_summary["diff_mean"],
        "rmse_diff_ci_low": rmse_summary["boot_ci_low"],
        "rmse_diff_ci_high": rmse_summary["boot_ci_high"],
    }
    sign_test_dict.append(record)

sign_test_df = pd.DataFrame.from_records(sign_test_dict)


# Bootstrap CIs for bar plot error bars
np.random.seed(0)
data_for_barplot = combined_results[
    ~combined_results.model.str.contains("alt")
].copy()
data_for_barplot.loc[data_for_barplot["group"].str.contains("Human"), "group"] = (
    "Human"
)

melted_for_errbars = pd.melt(
    data_for_barplot,
    id_vars=["snr", "background_condition", "group"],
    value_vars=["accuracy", "confusions"],
    var_name="metric",
    value_name="measure",
).reset_index(drop=True)
melted_for_errbars["condition_string"] = (
    melted_for_errbars["snr"].astype(str)
    + " dB "
    + melted_for_errbars["background_condition"]
    + " "
    + melted_for_errbars["metric"]
)
melted_for_errbars = melted_for_errbars.sort_values(["condition_string"])

model_list_err = [
    model for model in melted_for_errbars.group.unique() if "Human" not in model
]

unique_conds = melted_for_errbars.condition_string.unique()
n_conditions = melted_for_errbars.condition_string.nunique()

full_human_measure = melted_for_errbars[
    melted_for_errbars.group == "Human"
].sort_values(["condition_string"])


def bootstrap_stats(
    model,
    full_human_measure_df,
    melted_df,
    unique_conds,
    n_conditions,
    n_boots=1000,
):
    model_measure = melted_df[melted_df.group == model].sort_values(
        ["condition_string"]
    )
    full_human_measure = full_human_measure_df.sort_values(["condition_string"])

    r, _ = stats.pearsonr(full_human_measure.measure, model_measure.measure)
    r2_full = r**2
    rmse_full = np.sqrt(
        np.mean((full_human_measure.measure.values - model_measure.measure.values) ** 2)
    )
    rho_full = stats.spearmanr(
        full_human_measure.measure, model_measure.measure
    ).statistic

    r_boots = np.zeros(n_boots)
    rmse_boots = np.zeros(n_boots)
    rho_boots = np.zeros(n_boots)

    for ix in range(n_boots):
        conditions_to_sample = np.random.choice(unique_conds, size=n_conditions, replace=True)

        human_sample = np.array(
            [
                full_human_measure.loc[
                    full_human_measure.condition_string == cond, "measure"
                ].item()
                for cond in conditions_to_sample
            ]
        )
        model_sample = np.array(
            [
                model_measure.loc[
                    model_measure.condition_string == cond, "measure"
                ].item()
                for cond in conditions_to_sample
            ]
        )

        r_boot, _ = stats.pearsonr(human_sample, model_sample)
        r_boots[ix] = r_boot**2
        rmse_boots[ix] = np.sqrt(np.mean((human_sample - model_sample) ** 2))
        rho_boots[ix] = stats.spearmanr(human_sample, model_sample).statistic

    r_ci = np.percentile(r_boots, [2.5, 97.5])
    rmse_ci = np.percentile(rmse_boots, [2.5, 97.5])
    rho_ci = np.percentile(rho_boots, [2.5, 97.5])

    return {
        "group": model,
        "r2": r2_full,
        "r2_ci": r_ci,
        "rmse": rmse_full,
        "rmse_ci": rmse_ci,
        "rho": rho_full,
        "rho_ci": rho_ci,
    }


model_agg_sim_records = []
for model in model_list_err:
    model_agg_sim_records.append(
        bootstrap_stats(
            model,
            full_human_measure,
            melted_for_errbars,
            unique_conds,
            n_conditions,
            n_boots=1000,
        )
    )

model_agg_sim_df = pd.DataFrame.from_records(model_agg_sim_records)


def get_star(p_val: float) -> str:
    if p_val < 0.0001:
        return "****"
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    return ""


def draw_stats_bar(
    ax,
    x1,
    x2,
    y,
    h,
    text,
    th=0.025,
    lw=1.5,
    col="k",
    fontsize=10,
    text_gap=0.02,
):
    text_x = (x1 + x2) * 0.5
    gap_half_width = text_gap * len(text)
    ax.plot([x1, x1, text_x - gap_half_width], [y, y + h, y + h], lw=lw, c=col)
    ax.plot([text_x + gap_half_width, x2, x2], [y + h, y + h, y], lw=lw, c=col)
    ax.text(
        text_x,
        y - 0.002,
        text,
        ha="center",
        va="center",
        color=col,
        fontsize=fontsize,
    )


# Final bar plot of pooled RMSE with error bars (Extended Data 8/9)
to_plot = model_agg_sim_df.copy()

model_order = deepcopy(util_analysis.model_name_order)
model_colors = deepcopy(util_analysis.model_color_dict)

to_plot.sort_values(
    "group",
    key=lambda x: x.map({model: i for i, model in enumerate(model_order)}),
    inplace=True,
)
model_order[0] = "Feature-gain\nModel"
to_plot["group"].replace("Feature-gain Model", "Feature-gain\nModel", inplace=True)

xtick_labels = model_order.copy()
xtick_labels[0] = "Feature-gain\nmodel"
xtick_labels[1] = "Baseline\nCNN"
xtick_labels[-1] = "Norm pre\ngain model"
model_colors["Feature-gain\nModel"] = model_colors["Feature-gain Model"]

x_vals = {model: ix for ix, model in enumerate(model_order)}

aspect = 4.25
fontsize = 10

fig, ax = plt.subplots(figsize=(aspect * 1.25, aspect))

bar_width = 0.5
for _, row in to_plot.iterrows():
    model = row["group"]
    if model not in model_order:
        continue
    y_err = np.array(
        [row["rmse"] - row["rmse_ci"][0], row["rmse_ci"][1] - row["rmse"]]
    ).reshape(2, -1)
    ax.bar(
        x_vals[model],
        row["rmse"],
        yerr=y_err,
        alpha=0.7,
        color=model_colors[row["group"]],
        width=bar_width,
        edgecolor="k",
        capsize=3,
    )
    if model != "Feature-gain\nModel":
        p_val = sign_test_df[sign_test_df.model == model].rmse_sign_test_p.item()
        star = get_star(p_val)
        draw_stats_bar(
            ax,
            0,
            x_vals[model],
            0.2,
            0.00,
            star,
            col="k",
            lw=1,
            fontsize=fontsize,
            text_gap=0.1,
        )

ax.set_ylabel(
    "Human-model dissimilarity\n(Root-mean-square error)", fontsize=fontsize + 2
)
ax.set_xticks(np.arange(len(model_order)))
ax.set_xticklabels(xtick_labels, fontsize=fontsize)

if not DRY_RUN:
    fig.savefig(
        fig_out_dir / "figure_6_human-model_dissim_bar_w_norm_pre_gain.pdf",
        bbox_inches="tight",
        transparent=True,
    )



