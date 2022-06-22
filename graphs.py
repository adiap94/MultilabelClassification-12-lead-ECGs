import os
import pandas as pd

def compare_results(dict_path):
    cols = ['sensitivity', 'specificity']

    for col in cols:
        df = pd.DataFrame()
        for key, value in dict_path.items():
            df_tmp = pd.read_csv(os.path.join(key,"sen&spe_all.csv"))
            # if 'Unnamed: 0' in df_tmp:
            #     df_tmp = df_tmp.drop(columns=['Unnamed: 0'])
            df[value] =df_tmp[col]
            print("hello")
    pass


if __name__ == '__main__':
    dict_path = {
        "/tcmldrive/project_dl/results/restore": "original model",
        "/tcmldrive/project_dl/results/20220622-023413/": "second model"
    }
    compare_results(dict_path)
    pass
