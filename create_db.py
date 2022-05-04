import os
import re
import pandas as pd

dx_list = ['10370003', '111975006', '164889003',
       '164890007', '164909002', '164917005', '164934002', '164947007',
       '251146004', '270492004', '284470004', '39732003', '426177001',
       '426627000', '426783006', '427084000', '427172004', '427393009',
       '445118002', '47665007', '59931005', '698252002', '713426002',
       '713427006']

equivalent_classes = {'59118001': '713427006','63593006': '284470004','17338001': '427172004'}
equivalent_classes_list = ['59118001', '63593006', '17338001']

def get_info(header_path):
    with open(header_path,'r') as f:
        header_data=f.readlines()
        Dx = header_data[15]
        Dx = re.findall('\d+', Dx)

        age =  header_data[13]
        # age = float(re.findall('\d+', age)[0])
        age = float(age.split("#Age: ")[1].split("\n")[0])
        gender = header_data[14]
        gender = gender.split("#Sex: ")[1].split("\n")[0]

        fs = float(header_data[0].split(" ")[2])
    return age,gender,fs,Dx

def get_test_code(dx_mapping_scored_path):
    df_mapping = pd.read_csv(dx_mapping_scored_path)
    test_code_list = df_mapping["SNOMED CT Code"].unique()
    test_code_list = [str(x) for x in test_code_list]

    return test_code_list
def main(source_dir,out_dir):
    mat_files = [os.path.join(root, f) for root, subdirs, files in os.walk(source_dir) for f in files if
               f.lower().endswith(".mat")]
    #create dataframe
    df = pd.DataFrame({"mat_path": mat_files})
    # df = df.head()
    df["hea_path"] = df.mat_path.apply(lambda x : x.replace('.mat','.hea'))
    # df[["age","gender","fs","Dx"]] = df.hea_path.apply(lambda x : get_info(x))
    for index, row in df.iterrows():
        # age,gender,fs,Dx = get_info(header_path=row.hea_path)
        df.loc[index,["age", "gender", "fs", "Dx"]] = get_info(header_path=row.hea_path)

    df[dx_list] = 0

    for index, row in df.iterrows():
        for dx in row.Dx:
            if dx in dx_list:
                if dx in equivalent_classes_list:
                    dx = equivalent_classes[dx]
                df.loc[index, dx] = 1

    train_split = "./data_split/train_split4.csv"
    val_split = "./data_split/test_split4.csv"

    train_db = pd.read_csv(train_split)
    val_db = pd.read_csv(val_split)

    train_db["basename"] = train_db.filename.apply(lambda x: os.path.basename(x))
    val_db["basename"] = val_db.filename.apply(lambda x: os.path.basename(x))
    df["basename"] = df.mat_path.apply(lambda x: os.path.basename(x))

    train_files = train_db.basename.to_list()
    df.loc[df.basename.isin(train_files), "indicator"] = 1
    val_files = val_db.basename.to_list()
    df.loc[df.basename.isin(val_files), "indicator"] = 2

    # df.set.fillna(value="train" , inplace=True)
    df = df[~df.indicator.isna()]
    df.loc[df.indicator == 1, "set"] = "train"

    idx = df[df.set.isna()].sample(frac=0.5).index
    df.loc[idx, "set"] = "val"

    df.set.fillna(value="test", inplace=True)
    df.to_csv(os.path.join(out_dir,"db.csv"),index=False)

    print("finish creating db")
    print("sava db to: " + os.path.join(out_dir,"db.csv"))
if __name__ == '__main__':
    dx_mapping_scored_path ="/tcmldrive/project_dl/code/adi/utils/dx_mapping_scored.csv"
    out_dir = "/tcmldrive/project_dl/db/"
    source_dir="/tcmldrive/project_dl/data/"
    main(source_dir=source_dir,out_dir=out_dir)