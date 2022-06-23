import os
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def compare_results(dict_path,out_dir,improveCheckModel=None):
    out_dir = os.path.join(out_dir , time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(os.path.join(out_dir,"diff_figs"), exist_ok=True)
    os.chmod(os.path.join(out_dir,"diff_figs"), mode=0o777)
    os.makedirs(os.path.join(out_dir,"figs"), exist_ok=True)
    os.chmod(os.path.join(out_dir,"figs"), mode=0o777)
    os.makedirs(os.path.join(out_dir,"csv"), exist_ok=True)
    os.chmod(os.path.join(out_dir,"csv"), mode=0o777)
    if improveCheckModel:
        df_check = pd.DataFrame()
    metric_list = ['sensitivity', 'specificity', 'precision', 'accuracy', 'auroc', 'auprc', 'f_measure', 'f_beta_measure',
     'g_beta_measure']

    for metric in metric_list:
        df = pd.DataFrame()
        for key, value in dict_path.items():
            df_tmp = pd.read_csv(os.path.join(key,"analyze_all_classes.csv"))
            if 'Unnamed: 0' in df_tmp:
                df_tmp = df_tmp.drop(columns=['Unnamed: 0'])
            df[value] =df_tmp[metric]
            print("hello")

        bar_plot(df=df, metric_str=metric)
        plt.savefig(os.path.join(out_dir,"figs", metric + ".png"))

        if improveCheckModel:
            df = calc_diff(df=df, improveCheckModel=improveCheckModel)
            bar_plot(df = df.filter(like='_diff', axis=1), metric_str=metric+"_diff")
            plt.savefig(os.path.join(out_dir,"diff_figs", metric + "_diff.png"))
        df.to_csv(os.path.join(out_dir,"csv", metric + ".csv"), index=False)


        # df.loc[df['original model + focal loss'] > df['original model'], "indicator"] = 1
        # bool_check = df.idxmax(axis=1) == "original model + focal loss"
        # idx = np.where(bool_check.to_numpy())[0]
    pass

def calc_diff(df,improveCheckModel):
    col_diff = df.columns.to_list()
    col_diff.remove(improveCheckModel)

    for col in col_diff:
        df[col + "_diff"] = df[improveCheckModel] - df[col]
        df.loc[df[col + "_diff"]>0 , col +"_indicator"] = 1

    return df

def bar_plot(df, metric_str):
    df = df.copy()
    df.plot.bar()
    plt.grid(axis="y")
    plt.ylabel(metric_str)
    plt.xlabel("abnormalities")
    plt.title(metric_str)


if __name__ == '__main__':
    dict_path = {
        "/tcmldrive/project_dl/results/restore": "original model",
        "/tcmldrive/project_dl/results/20220622-204757/": "original model + focal loss"
    }
    out_dir = "/tcmldrive/project_dl/results/compare_results"
    improveCheckModel = "original model + focal loss"
    compare_results(dict_path,out_dir,improveCheckModel)
    pass
