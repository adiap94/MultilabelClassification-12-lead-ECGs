import pandas as pd
df = pd.read_csv("/tcmldrive/project_dl/db/db_03052022.csv")
df_sum = df.groupby(["set"])['10370003','111975006', '164889003', '164890007', '164909002', '164917005',
         '164934002', '164947007', '251146004', '270492004', '284470004',
         '39732003', '426177001', '426627000', '426783006', '427084000',
         '427172004', '427393009', '445118002', '47665007', '59931005',
                             '698252002', '713426002', '713427006'].sum()
a = df_sum.sum()
a_ratio = a/a.min()
df_ratio = a_ratio.to_frame("weights")
df_ratio["dx"]=df_ratio.index
df_ratio = df_ratio.reset_index(drop=True)
df_ratio.to_csv("weightsByRatio.csv",index=False)

