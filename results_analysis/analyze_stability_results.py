import json
import sys
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import create_dictionary_from_results, dict_to_df_fold_seed_metric

def plot_model_std_heatmap(
    df,
    ax,
    metric,
    exp_order=None,
    model_order=None,
    versions=('std', "exp"),
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
        sub = dfr[dfr['Model'] == model].copy().set_index(['version','Experiment'])
        # rows = [v for v in versions]
        sub_v1 = sub.loc(axis=0)[versions[0]][[metric]].rename(columns={metric: versions[0]})
        sub_v2 = sub.loc(axis=0)[versions[1]][[metric]].rename(columns={metric: versions[1]})
        sub = pd.concat([sub_v1, sub_v2], axis=1)
        if exp_order is not None:
            sub = sub.loc[exp_order]
        # sub = sub.reset_index(drop=True)
        # sub.columns = pd.MultiIndex.from_product([[model], versions])
        print(sub)
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
    if cbar:
        cbar_ax.tick_params(labelsize=12)

    # Etiquetas inferiores: thresholds
    ax.set_ylabel("Data configuration", y=0.4, fontsize=12)
    ax.set_xlabel("Metric version", fontsize=12)
    ax.tick_params(axis='both', labelsize = 12)

    # Separadores entre bloques de modelos
    n_thr = len(versions)
    for k in range(1, len(model_order)):
        ax.axvline(k * n_thr, color='white', lw=2)

    # Etiquetas superiores: modelos
    y_top = -0.2
    for i, model in enumerate(model_order):
        center = i * n_thr + n_thr / 2
        ax.text(center, y_top, model, ha='center', va='bottom', fontsize=12)

    # Ajustar l�mites para que se vea el texto superior
    ax.set_ylim(len(full.index), -1.0)


if __name__ == "__main__":

    path_to_metrics = sys.argv[1]
    lesion=sys.argv[2]

    experiment_results = create_dictionary_from_results(path_to_metrics, lesion, 0.75)
    df_std = dict_to_df_fold_seed_metric(experiment_results["standard"])
    df_exp = dict_to_df_fold_seed_metric(experiment_results["explain"])

    # df_exp_temp = df_exp.rename(columns={"precision":"P_exp", "recall":"recall_exp", 
    #                                      "f1":"f1_exp", "acc":"acc_exp", "auc-roc":"auc-roc_exp", "auprc":"auprc_exp"})

    df_all = pd.concat([df_std, df_exp], keys=['std','exp'], names=["version"])

    print(df_all)

    df_std_metrics = df_all.groupby(["version","Experiment","Model"]).agg("std")

    print(df_std_metrics)

    exp_order = ['Bl', 'P1', 'P2', 'A1', 'A2']
    model_order = ['DenseNet', 'EfficientNet', 'MobileNet', 'ResNet18', 'ResNet50']

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.subplots_adjust(right=0.88)
    axes_flat = axes.flatten()

    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])

    titles = ['Standard deviation in F1-score', 'Standard deviation in AUC']

    for idx, metric in enumerate(['F1-score', 'AUC']):

        plot_model_std_heatmap(
            df_std_metrics,
            ax=axes_flat[idx],
            metric=metric,
            exp_order=exp_order,
            model_order=model_order,
            versions=['std','exp'],
            vmin=0,
            vmax=0.2,
            annot=True,
            cbar=idx==0,
            cbar_ax=cbar_ax if idx==0 else None
        )

        axes_flat[idx].set_title(titles[idx], fontsize=14)


    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig("std_heatmaps_"+sys.argv[2]+".png")        
    plt.show()


    ##### FOLD-SEED DISTRIBUTION #####

    # metric = 'f1'

    # # models = ["DenseNet", "EfficientNet", "MobileNet", "ResNet18", "ResNet50"]
    # models = ["ResNet50"]

    # for model in models:

    #     df_model = df_all.loc(axis=0)[:,:,model].reset_index()


    #     # axes = sns.catplot(df_std_resnet50, x='f1', y='Fold', hue='Seed', row=version', col='Experiment', palette=sns.color_palette())
    #     axes = sns.FacetGrid(df_model, row='version', col='Experiment', height=2, aspect=1)
    #     axes.map_dataframe(sns.stripplot,x=metric, y='Fold', hue='Seed', palette=sns.color_palette())
    #     axes.set_titles(col_template='{col_name}', row_template='{row_name}')
    #     axes.figure.suptitle(model, fontsize=12)
    #     axes.figure.subplots_adjust(top=0.85)


    #     plt.tight_layout()
    #     # plt.savefig("fold_seed_distribution_"+model+"_"+metric+"_"+sys.argv[2]+".png")        
    #     plt.show()

    titles = ['Distribution of F1-score across folds and seeds', 'Distribution of AUC across folds and seeds']

    for idx, metric in enumerate(['F1-score', 'AUC']):
        df_models = df_all.reset_index()
        df_models = df_models[(df_models['Model']=='DenseNet') | (df_models['Model']=='ResNet50')]
        axes = sns.FacetGrid(df_models, row='version', col='Model', height=3, aspect=1)
        axes.map_dataframe(sns.swarmplot,x=metric, y='Experiment', hue='Fold', palette=sns.color_palette())

        axes.set_titles(col_template='{col_name}', row_template='{row_name}', size=12)
        axes.figure.suptitle(titles[idx], fontsize=14)
        axes.figure.subplots_adjust(top=0.85)
        axes.set_axis_labels(metric, "Data configuration", fontsize=12)

        for ax in axes.axes.flat:
            ax.tick_params(axis='both', labelsize=12)

        plt.tight_layout()
        plt.savefig("fold_seed_distribution_all"+"_"+metric+"_"+sys.argv[2]+".png")        
        # plt.show()
    



