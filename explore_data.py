from scipy.io import loadmat
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
# case = r"/tcmldrive/project_dl/data/Training_2/Q0001.mat"
# x = loadmat(case)
# data = np.asarray(x['val'], dtype=np.float64)
# # data = Resample(data, src_fs, tar_fs)

train_split = "./data_split/train_split4.csv"
val_split = "./data_split/test_split4.csv"
our_db = "/tcmldrive/project_dl/db/db.csv"

train_db = pd.read_csv(train_split)
val_db = pd.read_csv(val_split)
our_db = pd.read_csv(our_db)

train_db["basename"]=train_db.filename.apply(lambda x : os.path.basename(x))
val_db["basename"]=val_db.filename.apply(lambda x : os.path.basename(x))
our_db["basename"]=our_db.mat_path.apply(lambda x : os.path.basename(x))

train_files = train_db.basename.to_list()
our_db.loc[our_db.basename.isin(train_files),"indicator"]= 1
val_files = val_db.basename.to_list()
our_db.loc[our_db.basename.isin(val_files),"indicator"]= 2


our_db.indicator.fillna(3 , inplace=True)

our_db.loc[our_db.indicator==1,"set"] = "val"
our_db.loc[our_db.indicator==2,"set"] = "train"
our_db.loc[our_db.indicator==3,"set"] = "test"

our_db.drop(columns=["potential_code"],inplace=True)

dx_list = ['10370003', '111975006', '164889003',
       '164890007', '164909002', '164917005', '164934002', '164947007',
       '251146004', '270492004', '284470004', '39732003', '426177001',
       '426627000', '426783006', '427084000', '427172004', '427393009',
       '445118002', '47665007', '59931005', '698252002', '713426002',
       '713427006']

equivalent_classes = {'713427006': '59118001','284470004': '63593006','427172004': '17338001'}
equivalent_classes_list = list(equivalent_classes.keys())

our_db[dx_list] = 0

for index, row in our_db.iterrows():
    for dx in row.Dx:
        our_db.loc[index, dx] = 1

print("debug")

