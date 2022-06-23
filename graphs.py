import os
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def compare_results(dict_path,out_dir,improveCheckModel=None):
    out_dir = os.path.join(out_dir , time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    os.chmod(out_dir, mode=0o777)

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

        # if improveCheckModel:

        df.to_csv(os.path.join(out_dir, metric + ".csv"), index=False)
        plt.savefig(os.path.join(out_dir,metric+".png"))

        # df.loc[df['original model + focal loss'] > df['original model'], "indicator"] = 1
        # bool_check = df.idxmax(axis=1) == "original model + focal loss"
        # idx = np.where(bool_check.to_numpy())[0]
    pass

# def calc_diff(df,improveCheckModel):
#     b = df.columns.to_list()
#     b.remove(improvecheck)
def bar_plot(df, metric_str):
    labels = df.index

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()

    width_delta = [- width / 2, width / 2]
    count = 0
    for model_str in df.columns:
        ax.bar(x + width_delta[count], df[model_str], width, label=model_str)
        count = count+1

    ax.set_ylabel(metric_str)
    ax.set_ylabel("abnormalities")
    ax.set_title(metric_str)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    # plt.show()


if __name__ == '__main__':
    dict_path = {
        "/tcmldrive/project_dl/results/restore": "original model",
        "/tcmldrive/project_dl/results/20220622-204757/": "original model + focal loss"
    }
    out_dir = "/tcmldrive/project_dl/results/compare_results"
    compare_results(dict_path,out_dir)
    pass
