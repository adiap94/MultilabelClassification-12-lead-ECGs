import os
import time
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def compare_results(dict_path,out_dir,improveCheckModel=None):
    out_dir = os.path.join(out_dir , time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(os.path.join(out_dir,"diff_figs"), exist_ok=True)
    os.chmod(os.path.join(out_dir,"diff_figs"), mode=0o777)
    os.makedirs(os.path.join(out_dir,"figs"), exist_ok=True)
    os.chmod(os.path.join(out_dir,"figs"), mode=0o777)
    os.makedirs(os.path.join(out_dir,"csv"), exist_ok=True)
    os.chmod(os.path.join(out_dir,"csv"), mode=0o777)
    os.makedirs(os.path.join(out_dir,"heatmap"), exist_ok=True)
    os.chmod(os.path.join(out_dir,"heatmap"), mode=0o777)
    os.makedirs(os.path.join(out_dir,"csvPerModel"), exist_ok=True)
    os.chmod(os.path.join(out_dir,"csvPerModel"), mode=0o777)
    if improveCheckModel:
        df_check = pd.DataFrame()
    metric_list = ['sensitivity', 'specificity', 'precision', 'accuracy', 'auroc', 'auprc', 'f_measure', 'f_beta_measure',
     'g_beta_measure']

    # compare metrics of all runs
    df = compare_all_metrics(dict_path)
    df.to_csv(os.path.join(out_dir, "csv", "all_metrics.csv"), index=False)

    for metric in metric_list:
        df = pd.DataFrame()
        for key, value in dict_path.items():
            df_tmp = pd.read_csv(os.path.join(key,"analyze_all_classes_2.csv"))
            df_tmp = df_tmp.set_index(['dx'])
            if 'Unnamed: 0' in df_tmp:
                df_tmp = df_tmp.drop(columns=['Unnamed: 0'])
            df[value] =df_tmp[metric]

        bar_plot(df=df, metric_str=metric)
        plt.savefig(os.path.join(out_dir,"figs", metric + ".png"))

        if improveCheckModel:
            df = calc_diff(df=df, improveCheckModel=improveCheckModel)
            df_diff = df = df.filter(like='_diff', axis=1)
            bar_plot(df_diff, metric_str=metric+"_diff")
            plt.savefig(os.path.join(out_dir,"diff_figs", metric + "_diff.png"))

            df_diff_metric = df_diff.add_prefix(metric+'_')
            df_check = pd.concat([df_check, df_diff_metric],axis=1)

        df.to_csv(os.path.join(out_dir,"csv", metric + ".csv"), index=False)

    if improveCheckModel:
        for model in improveCheckModel:
            model_cols = [col for col in df_check.columns if model in col]
            run_heatmap_per_model(df_check = df_check[model_cols], out_dir = out_dir, model_str=model)

    pass

def run_heatmap_per_model(df_check,out_dir,model_str):
    df_check= df_check.copy()
    mapping={}
    # fix column names to have shorter name with metric only
    for col in df_check.columns:
        mapping[col] = col.split("_" + model_str + "_diff")[0]
    df_check.rename(columns=mapping, inplace=True)

    df_check.to_csv(os.path.join(out_dir,"csvPerModel", model_str+"_diff.csv"), index=False)
    plt.figure()
    sns.heatmap(df_check)
    plt.title(model_str+" diff")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"heatmap", "DiffHeatmap_"+model_str+".png"))

    df_check1 = df_check.copy()
    df_check1[df_check1 < 0] = 0
    df_check1[df_check > 0] = 1
    plt.figure()
    sns.heatmap(df_check1)
    plt.title(model_str + " diff")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"heatmap", "BinaryDiffHeatmap_"+model_str+".png"))

def compare_all_metrics(dict_path):
    df = pd.DataFrame()
    for key, value in dict_path.items():
        df_tmp = pd.read_csv(os.path.join(key, "metrics.csv"))
        df_tmp = df_tmp.set_index([pd.Index([value])])
        df = pd.concat([df, df_tmp], axis=0)
    df = df.reset_index(col_fill='Model')
    df = df.reset_index()
    df.rename(columns={"index": "model"}, inplace=True)

    return df

def calc_diff(df,improveCheckModel):
    cols = df.columns.to_list()
    col_diff = [fruit for fruit in cols if fruit not in improveCheckModel][0]
    # col_diff.remove(improveCheckModel)

    for model in improveCheckModel:
        df[model + "_diff"] =  df[model] - df[col_diff]
        df.loc[df[model + "_diff"]>0 , model +"_indicator"] = 1

    return df

def bar_plot(df, metric_str):
    df = df.copy()
    df.plot.bar()
    plt.grid(axis="y")
    plt.ylabel(metric_str)
    plt.xlabel("abnormalities")
    plt.title(metric_str)
    plt.tight_layout()

if __name__ == '__main__':
    dict_path = {
        "/tcmldrive/project_dl/results/restore": "BCE loss",
        "/tcmldrive/project_dl/results/20220623-201217/": "Focal loss",
        "/tcmldrive/project_dl/results/20220624-120955":"ASL loss",
        "/tcmldrive/project_dl/results/20220624-181009": "Weighted BCE loss"
    }
    out_dir = "/tcmldrive/project_dl/results/compare_results"
    improveCheckModel = ["Focal loss","ASL loss","Weighted BCE loss"]
    compare_results(dict_path,out_dir,improveCheckModel)
    pass
