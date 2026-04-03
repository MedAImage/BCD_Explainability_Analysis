import json
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import create_dictionary_from_results


def _darken(color, factor=0.65):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(np.clip(rgb * factor, 0, 1))


def read_jsonl(file):
    json_obj = []
    buffer = ""
    for l in file:
        if l=="}{\n":
            l = "}\n"
        buffer += l
        try:
            obj = json.loads(buffer)
            json_obj.append(obj)
            buffer = "{\n"
        except json.JSONDecodeError:
            pass
    return json_obj

def dict_to_df(d):
    df = (
        pd.DataFrame.from_dict(
            {(exp, model): metrics
            for exp, models in d.items()
            for model, metrics in models.items()},
            orient="index"
        )
    )

    df.index = pd.MultiIndex.from_tuples(
        df.index, names=["Experiment", "Model"]
    )
    return df

def dict_to_df_fold_seed_metric(d):
    rows = []

    for exp, models in d.items():
        for model, seeds in models.items():
            for seed, folds in seeds.items():
                for fold, metrics in folds.items():

                    row = {
                        "Experiment": exp,
                        "Model": model,
                        "Seed": seed,
                        "Fold": fold,                        
                    }

                    row.update(metrics)

                    rows.append(row)

    df = pd.DataFrame(rows)

    df = df.set_index(
        ["Experiment", "Model", "Seed", "Fold"]
    ).sort_index()

    df = df.reindex(['Baseline', 'P1', 'P2', 'A1', 'A2'], level=0)

    return df

def dict_to_df_fold_seed_XAI_metric(d):
    rows = []

    for exp, models in d.items():
        for model, seeds in models.items():
            for seed, folds in seeds.items():
                for fold, methods in folds.items():
                        for method, metrics in methods.items():
                            row = {
                                "Experiment": exp,
                                "Model": model,
                                "Seed": seed,
                                "Fold": fold,
                                "Method": method
                            }

                            row.update(metrics)

                            rows.append(row)

    df = pd.DataFrame(rows)

    df = df.set_index(
        ["Experiment", "Model", "Seed", "Fold", "Method"]
    ).sort_index()

    df = df.reindex(['Baseline', 'P1', 'P2', 'A1', 'A2'], level=0)

    return df


metric_names = ["precision", "recall", "f1", "acc", "auc-roc"]
def get_results(experiments):
    kfold_results = dict()
    final_results = dict()
    final_results_std = dict()
    for exp in experiments:        
        kfold_results[exp] = dict()
        final_results[exp] = dict()
        final_results_std[exp] = dict()
        for K in experiments[exp]:
            for model in experiments[exp][K]:
                if model not in kfold_results[exp]:
                    kfold_results[exp][model] = dict()
                    final_results[exp][model] = {m: [] for m in metric_names}
                kfold_results[exp][model][K] = dict()
                for m in metric_names:
                    kfold_results[exp][model][K][m] = np.mean(np.array(experiments[exp][K][model][m]))
                    # final_results[exp][model][m].append(kfold_results[exp][model][K][m])
                    final_results[exp][model][m]+=experiments[exp][K][model][m]
        for model in final_results[exp]:
            final_results_std[exp][model] = dict()
            for m in metric_names:
                # print("exp", exp, "model", model, "metric", m, len(final_results[exp][model][m]))
                final_results_std[exp][model][m] = np.std(np.array(final_results[exp][model][m]))
                final_results[exp][model][m] = np.mean(np.array(final_results[exp][model][m]))

    return final_results, final_results_std

def performance_with_without_expl_plot_bars(
    ax,
    df_runs_orig: pd.DataFrame,
    df_runs_expl: pd.DataFrame,
    metric: str = "auc-roc",
    title: str | None = None,
    figsize=(10, 5),
    inner_factor=0.65,
    outer_width=0.16,
    inner_width=0.09,
):

    dfo = df_runs_orig.copy().reset_index()
    dfe = df_runs_expl.copy().reset_index()

    # Compute mean on folds-seeds
    mean_orig = (
        dfo.groupby(["Experiment", "Model"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "value_orig"})
    )
    mean_expl = (
        dfe.groupby(["Experiment", "Model"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "value_expl"})
    )

    # Align standard and explained results
    d = mean_orig.merge(mean_expl, on=["Experiment", "Model"], how="inner")

    models = sorted(d["Model"].unique())
    exps = ['Baseline', 'P1', 'P2', 'A1', 'A2']# d["Experiment"].unique()
    print(exps)

    x = np.arange(len(models))

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    exp_color = {exp: color_cycle[i % len(color_cycle)] for i, exp in enumerate(exps)}

    offsets = (np.arange(len(exps)) - (len(exps) - 1) / 2) * outer_width

    if ax is None:
        print('ax is None')
        fig, ax = plt.subplots(figsize=figsize)

    for j, exp in enumerate(exps):
        c = exp_color[exp]
        c_inner = _darken(c, factor=inner_factor)

        de = d[d["Experiment"] == exp].set_index("Model")

        y_orig = [de.loc[m, "value_orig"] if m in de.index else np.nan for m in models]
        y_expl = [de.loc[m, "value_expl"] if m in de.index else np.nan for m in models]

        xpos = x + offsets[j]

        ax.bar(xpos, y_orig, width=outer_width, label=exp, color=c, edgecolor="none")
        ax.bar(xpos, y_expl, width=inner_width, color=c_inner, edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric)
    # ax.set_title(title or f"{metric}: mean over folds�seeds (outer) vs explainability-weighted (inner)")
    ax.legend(title="Experiment", loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.grid(axis="y", alpha=0.25)
    return ax


def create_multiple_scatterplot_std_vs_exp(df_dict, metric, size, hue):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    fig.delaxes(axes[1][2])
    axes_flatten = axes.flatten()

    global_min_std = 100
    global_max_std = -100

    global_min_exp = 100
    global_max_exp = -100

    margin = 0.02

    models = ['DenseNet', 'EfficientNet', 'MobileNet', 'ResNet18', 'ResNet50']
    
    for i, (cat, df) in enumerate(df_dict.items()):
    # for i, model in enumerate(models):
        # df_model = df.loc(axis=0)[:,model]
        global_min_std = min(global_min_std, df.loc(axis=1)[metric].min())
        global_max_std = max(global_max_std, df.loc(axis=1)[metric].max())
        global_min_exp = min(global_min_exp, df.loc(axis=1)[metric+'_exp'].min())
        global_max_exp = max(global_max_exp, df.loc(axis=1)[metric+'_exp'].max())

        
        if i==len(models)-1:
            # axes_flatten[i].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
            legend = 'brief'
        else:
            legend = False

        axes_flatten[i] = sns.scatterplot(data=df, x=metric, y=metric+'_exp', size=size, hue=hue, legend=legend,ax=axes_flatten[i], s=60, alpha=0.7, sizes=(60,200))
        axes_flatten[i].set_title(cat)

    global_min_std -= margin
    global_max_std += margin
    global_min_exp -= margin
    global_max_exp += margin

    for i in range(len(models)):
        axes_flatten[i].set_xlim(global_min_std, global_max_std)
        axes_flatten[i].set_ylim (global_min_exp, global_max_exp)


    symbols, labels = axes_flatten[4].get_legend_handles_labels()
    axes_flatten[4].get_legend().set_visible(False)
    fig.legend(symbols, labels, loc="upper left", bbox_to_anchor=(0.72, 0.48))
    return fig, axes


if __name__ == "__main__":

    path_to_metrics = sys.argv[1]
    lesion=sys.argv[2]

    experiment_results = create_dictionary_from_results(path_to_metrics, lesion)
    df_std = dict_to_df_fold_seed_metric(experiment_results["standard"])
    df_exp = dict_to_df_fold_seed_metric(experiment_results["explain"])
    df_xai = dict_to_df_fold_seed_XAI_metric(experiment_results["XAI"])

    df_xai_baseline = df_xai.loc(axis=0)['Baseline']


    # mean_energy = (
    #     df_xai_baseline.groupby(["Model", "Method"])["energy_0.25"].agg(["mean", "std"])
    #     # .rename(columns={metric: "value_orig"})
    # )


    # pd.set_option('display.max_rows', None)
    # print(mean_energy)



    # mean_baseline = mean_orig.loc(axis=0)['Baseline']





    df_exp_temp = df_exp.rename(columns={"precision":"precision_exp", "recall":"recall_exp", 
                                         "f1":"f1_exp", "acc":"acc_exp", "auc-roc":"auc-roc_exp"})

    df_all = pd.concat([df_std, df_exp_temp], axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    axes = axes.reshape(1, -1)

    
    # Flatten the axes array for easy iteration
    axes_flat = axes.flatten()

    axes_flat[0] = performance_with_without_expl_plot_bars(axes_flat[0], df_std, df_exp, metric = 'f1') #metric=sys.argv[2])
    axes_flat[1] = performance_with_without_expl_plot_bars(axes_flat[1], df_std, df_exp, metric = 'auc-roc') #metric=sys.argv[2])

    plt.tight_layout()
    plt.show()










##########################



    # # METRIC vs METRIC WITH EXPLAINABILITY for models
    # models = ['DenseNet', 'EfficientNet', 'MobileNet', 'ResNet18', 'ResNet50']

    # df_models = dict()
    # for model in models:
    #     df_models[model] = df_all.loc(axis=0)[:,model]

    # fig, axes = create_multiple_scatterplot_std_vs_exp(df_models, metric, size='Fold', hue='Experiment')

    # plt.tight_layout()
    # plt.show()



    # # HEATMAP COMPARING WITH THE BASELINE

    # mean_orig = (
    #     df_all.groupby(["Experiment", "Model", "Seed"])#[metric]#, as_index=False)[metric]
    #     .mean()
    #     # .rename(columns={metric: "value_orig"})
    # )


    # mean_baseline = mean_orig.loc(axis=0)['Baseline']

    # diff_baseline = mean_orig - mean_baseline

    # df_dict_diff_with_baseline = dict()
    # global_min = 100
    # global_max = -100

    # for m in models:

    #     diff_model = diff_baseline.loc(axis=0)[m]

    #     metric_exp = metric+'_exp'

    #     diff_model_metric = diff_model.loc(axis=1)[[metric_exp]]
    #     global_min = min(global_min, diff_model.loc(axis=1)[metric_exp].min())
    #     global_max = max(global_min, diff_model.loc(axis=1)[metric_exp].max())
    #     df_dict_diff_with_baseline[m] = diff_model_metric[metric_exp].unstack("Experiment")#.pivot(index = 'Experiment', columns = 'Seed', values = metric)


    # fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    # fig.delaxes(axes[1][2])

    
    # # Flatten the axes array for easy iteration
    # axes_flat = axes.flatten()

    # global_min = -global_max
    # global_min -= 0.05
    # global_max += 0.05

    # for i, m in enumerate(models):
    #     axes_flat[i] = sns.heatmap(df_dict_diff_with_baseline[m], vmin=global_min, 
    #                                vmax=global_max, ax = axes_flat[i], cmap='coolwarm')
    #     axes_flat[i].set_title(m)
    # plt.tight_layout()
    # plt.show()

    # pd.set_option('display.max_rows', None)
    # print(mean_orig)
    # print(mean_baseline)
    # print(diff_model_metric)

