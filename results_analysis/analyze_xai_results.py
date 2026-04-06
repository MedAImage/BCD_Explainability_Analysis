import json
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import create_dictionary_from_results, dict_to_df_fold_seed_XAI_metric


if __name__ == "__main__":

    path_to_metrics = sys.argv[1]
    lesion=sys.argv[2]

    experiment_results = create_dictionary_from_results(path_to_metrics, lesion)
    df_xai = dict_to_df_fold_seed_XAI_metric(experiment_results["XAI"])

    df_xai_baseline = df_xai.loc(axis=0)['Baseline']

    xai_metric = 'PG_1-top'

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes_flat = axes.flatten()

    backbone = ["DenseNet", "EfficientNet", "MobileNet", "ResNet50"]
    df_xai_spearman = df_xai_baseline.loc[:,:,:,['C+', 'EC-cnn', 'EC-prj', 'EC-att', 'GC-cnn', 'GC-prj', 'GC-att']]

    mean_spearman = (
        df_xai_baseline.groupby(["Model", "Method"])["spearman"].agg(["mean", "std"]).reset_index()
        # .rename(columns={metric: "value_orig"})
    )
    print(mean_spearman)

    mean_spearman = mean_spearman.melt(id_vars=['Model', 'Method'], value_vars=['mean', 'std'],
            var_name='stat', value_name='value')
    mean_spearman = mean_spearman.pivot(index=['Model', 'stat'], columns='Method', values='value')
    mean_spearman = mean_spearman.reindex(['mean', 'std'], level='stat')


    # mean_spearman = pd.pivot_table(mean_spearman, values=['mean', 'std'], index=["Model"], columns=["Method"])
    
    # mean_spearman = pd.melt(mean_spearman, id_vars=["Model","mean", "std"])#, value_vars=["mean", "std"])
    # mean_spearman.columns = backbone 

    spearman_table = mean_spearman.to_latex(float_format="%.2f")
    print(spearman_table)

    for i, bb in enumerate(backbone):
        axes_flat[i] = sns.boxplot(df_xai_baseline.loc(axis=0)[bb], x="Method", y=xai_metric, hue="Method", 
                                   fill = False, linewidth=1.5, width=0.5, ax = axes_flat[i])
        
        axes_flat[i].set_title(bb)
        axes_flat[i].set_ylim([0, 70])
        axes_flat[i].set_ylabel('Pointing Game Accuracy (%)')
        # axes_flat[i].set_ylabel('Spearman correlation')
    
    plt.tight_layout()
    plt.savefig("pointing_game.png")
    plt.show()

    df_xai_energy = df_xai_baseline[["energy_0.25", "energy_0.5", "energy_0.75"]]
    df_xai_energy = df_xai_energy.melt(ignore_index=False)#.reset_index()

    df_xai_energy = df_xai_energy.replace({'energy_0.25': '0.25', 'energy_0.5': '0.50', 'energy_0.75': '0.75'})
    print(df_xai_energy)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes_flat = axes.flatten()

    backbone = ["DenseNet", "EfficientNet", "MobileNet", "ResNet50"]

    for i, bb in enumerate(backbone):
        axes_flat[i] = sns.swarmplot(df_xai_energy.loc[bb], y="variable", x = "value", hue = "Method", size=4, ax = axes_flat[i])
        axes_flat[i].set_title(bb)
        axes_flat[i].set_ylabel("Threshold")
        axes_flat[i].set_xlabel("ROI Energy Fraction")

        axes_flat[i].legend_.remove()

    # Crear una sola leyenda
    handles, labels = axes_flat[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc='upper center',
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.02)
    )


    # ax = sns.stripplot(df_xai_energy, y="variable", x = "value", hue = "Method")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("roi_energy_fraction.png")
    plt.show()
