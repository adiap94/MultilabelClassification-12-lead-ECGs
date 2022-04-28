from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
case = r"/tcmldrive/project_dl/data/Training_2/Q0001.mat"
x = loadmat(case)
data = np.asarray(x['val'], dtype=np.float64)
data = Resample(data, src_fs, tar_fs)
