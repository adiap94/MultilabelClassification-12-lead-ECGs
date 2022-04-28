import os
import re
import pandas as pd

def get_Dx(header_path):
    with open(header_path,'r') as f:
        header_data=f.readlines()
        Dx = header_data[15]
        Dx = re.findall('\d+', Dx)
    return Dx

def get_test_code(dx_mapping_scored_path):
    df_mapping = pd.read_csv(dx_mapping_scored_path)
    test_code_list = df_mapping["SNOMED CT Code"].unique()
    test_code_list = [str(x) for x in test_code_list]

    return test_code_list
def main(source_dir):
    mat_files = [os.path.join(root, f) for root, subdirs, files in os.walk(source_dir) for f in files if
               f.lower().endswith(".mat")]
    #create dataframe
    df = pd.DataFrame({"mat_path": mat_files})
    df["hea_path"] = df.mat_path.apply(lambda x : x.replace('.mat','.hea'))
    df["Dx"] = df.hea_path.apply(lambda x : get_Dx(x))

    test_code_list = get_test_code(dx_mapping_scored_path)
    for index, row in df.iterrows():
        Dx_code_list = row.Dx
        for dx_code in Dx_code_list:
            if dx_code in test_code_list:
                df.loc[index,"potential_code"]=1
    # Extract type
    db["type"] = db.seg.apply(lambda x: x.split(os.sep)[-3])

    # train
    idx = db.groupby(["type"]).sample(frac=0.8).index
    db.loc[idx,"set"] = "train"

    # val
    idx = db[db.set.isna()].groupby(["type"]).sample(frac=0.5).index
    db.loc[idx,"set"] = "val"

    # test
    db.set.fillna(value="test" , inplace=True)

    print("finish creating db")
if __name__ == '__main__':
    dx_mapping_scored_path ="/tcmldrive/project_dl/code/adi/utils/dx_mapping_scored.csv"

    source_dir="/tcmldrive/project_dl/data/"
    main(source_dir=source_dir)