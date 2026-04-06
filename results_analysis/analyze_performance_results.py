import json
import sys
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import create_dictionary_from_results, dict_to_df_fold_seed_metric


def _darken(color, factor=0.65):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(np.clip(rgb * factor, 0, 1))


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
    exps = ['Bl', 'P1', 'P2', 'A1', 'A2']# d["Experiment"].unique()
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

def plot_model_threshold_heatmap(
    df,
    metric_prefix,
    ax,
    exp_order=None,
    model_order=None,
    thresholds=('0.25', '0.5', '0.75'),
    cmap='viridis',
    vmin=None,
    vmax=None,
    annot=False,
    cbar=False,
    cbar_ax=None
):
    dfr = df.reset_index()

    blocks = []
    for model in model_order:
        sub = dfr[dfr['Model'] == model].copy().set_index('Experiment')
        cols = [f'{metric_prefix}_{t}' for t in thresholds]
        sub = sub[cols]

        if exp_order is not None:
            sub = sub.loc[exp_order]

        sub.columns = pd.MultiIndex.from_product([[model], thresholds])
        blocks.append(sub)

    full = pd.concat(blocks, axis=1)

    sns.heatmap(
        full,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=".2f",
        cbar=cbar,
        cbar_ax=cbar_ax
    )

    # Etiquetas inferiores: thresholds
    lower_labels = [str(t) for _m in model_order for t in thresholds]
    ax.set_xticklabels(lower_labels, rotation=0)
    ax.set_ylabel("Experiment", loc="center")
    ax.set_xlabel("Energy threshold")

    # Separadores entre bloques de modelos
    n_thr = len(thresholds)
    for k in range(1, len(model_order)):
        ax.axvline(k * n_thr, color='white', lw=2)

    # Etiquetas superiores: modelos
    y_top = -0.2
    for i, model in enumerate(model_order):
        center = i * n_thr + n_thr / 2
        ax.text(center, y_top, model, ha='center', va='bottom', fontsize=10)

    # Ajustar l�mites para que se vea el texto superior
    ax.set_ylim(len(full.index), -1.0)

if __name__ == "__main__":

    path_to_metrics = sys.argv[1]
    lesion=sys.argv[2]

    experiment_results = create_dictionary_from_results(path_to_metrics, lesion, 0.75)
    df_std = dict_to_df_fold_seed_metric(experiment_results["standard"])
    df_exp = dict_to_df_fold_seed_metric(experiment_results["explain"])

    df_exp_75 = df_exp
    experiment_results_25 = create_dictionary_from_results(path_to_metrics, lesion, 0.25)
    df_exp_25 = dict_to_df_fold_seed_metric(experiment_results_25["explain"])
    experiment_results_50 = create_dictionary_from_results(path_to_metrics, lesion, 0.5)
    df_exp_50 = dict_to_df_fold_seed_metric(experiment_results_50["explain"])


    df_exp_temp = df_exp_75.rename(columns={"precision":"P_exp", "recall":"recall_exp", 
                                         "f1":"f1_exp", "acc":"acc_exp", "auc-roc":"auc-roc_exp", "auprc":"auprc_exp"})

    df_all = pd.concat([df_std, df_exp_temp], axis=1)

    df_mean_metrics = df_all.groupby(["Experiment","Model"]).agg("mean")

    df_mean_metrics_effNet = df_mean_metrics.loc(axis=0)[:,"EfficientNet"].reset_index(level=1, drop=True)

    idx_min = df_mean_metrics_effNet.idxmin()
    idx_max = df_mean_metrics_effNet.idxmax()    
    df_mean_metrics_effNet = df_mean_metrics_effNet.round(2)
    for col, row_idx in idx_min.items():
        val = df_mean_metrics_effNet.loc[row_idx, col]
        new_val = f"{{\color{{red}} {val:.2f} }}"
        df_mean_metrics_effNet.loc[row_idx, col] = new_val

    for col, row_idx in idx_max.items():
        val = df_mean_metrics_effNet.loc[row_idx, col]
        new_val = f"{{\color{{mygreen}} {val:.2f} }}"
        df_mean_metrics_effNet.loc[row_idx, col] = new_val

    latex_table = df_mean_metrics_effNet.to_latex(escape=False)
    latex_table = re.sub(' +', ' ', latex_table)
    print(latex_table) #float_format="%.2f"))
    

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    axes = axes.reshape(1, -1)

    
    # Flatten the axes array for easy iteration
    axes_flat = axes.flatten()

    figure_metrics = ["recall", "f1", "auc-roc", "auprc"]
    for idx, metric in enumerate(figure_metrics):
        axes_flat[idx] = performance_with_without_expl_plot_bars(axes_flat[idx], df_std, df_exp, metric = metric)

        axes_flat[idx].legend_.remove()

    # Crear una sola leyenda
    handles, labels = axes_flat[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc='upper center',
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.02)
    )

    plt.tight_layout()
    plt.savefig("performance_metrics_"+sys.argv[2]+".png")
    plt.show()



    #Plots for explainability penalization

    vmin = 0
    vmax = 1

    exp_order = ['Bl', 'P1', 'P2', 'A1', 'A2']
    model_order = ['DenseNet', 'EfficientNet', 'MobileNet', 'ResNet18', 'ResNet50']
    thresholds = ['0.25', '0.5', '0.75']

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.subplots_adjust(right=0.88)
    axes_flat = axes.flatten()

    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])


    metrics = ['f1', 'auc-roc']
    titles = ['F1 penalty', 'AUC-ROC penalty']
    for idx_m, metric in enumerate(metrics):
        df_exp_th = [df_exp_25, df_exp_50, df_exp_75]
        thresholds = [0.25, 0.50, 0.75]

        df_metric = df_all[metric]
        df_penalization_all = []
        for idx in range(len(df_exp_th)):
            df_metric_explained = df_exp_th[idx][metric]
            df_penalization = 1-df_metric_explained/df_metric
            df_penalization = df_penalization.to_frame(name=metric+'_'+str(thresholds[idx]))

            mean_penalization = (
                    df_penalization.groupby(["Experiment", "Model"])#[metric]#, as_index=False)[metric]
                    .mean().rename(columns={metric: 'f1_'+str(thresholds[idx])})
                )
            df_penalization_all.append(mean_penalization)
        
        df_penalization_all = pd.concat(df_penalization_all, axis=1)

        # print(df_penalization_all)
        # axes = axes.reshape(1, -1)

        plot_model_threshold_heatmap(
            df_penalization_all,
            metric_prefix=metric,
            ax=axes_flat[idx_m],
            exp_order=exp_order,
            model_order=model_order,
            thresholds=thresholds,
            vmin=vmin,
            vmax=vmax,
            annot=True,
            cbar=idx_m==0,
            cbar_ax=cbar_ax if idx_m==0 else None
        )

        axes_flat[idx_m].set_title(titles[idx_m])


    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig("metrics_penalization_"+sys.argv[2]+".png")    
    plt.show()







# ##########################



#     # METRIC vs METRIC WITH EXPLAINABILITY for models
#     models = ['DenseNet', 'EfficientNet', 'MobileNet', 'ResNet18', 'ResNet50']

#     df_models = dict()
#     for model in models:
#         df_models[model] = df_all.loc(axis=0)[:,model]

#     fig, axes = create_multiple_scatterplot_std_vs_exp(df_models, metric, size='Fold', hue='Experiment')

#     plt.tight_layout()
#     plt.show()



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

