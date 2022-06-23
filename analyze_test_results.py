import sys
sys.path.insert(0, '..')
from utils.metrics import *
import pandas as pd
from collections import namedtuple
import os
from utiles import json_load
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score
import torch
import torch.nn.functional as F
import models


if __name__ == '__main__':

    args = json_load("./config/config.json")
    args = namedtuple('Struct', args.keys())(*args.values())

    output_directory = '/tcmldrive/project_dl/results/restore/test/'
    auroc, auprc, auroc_all, auprc_all, accuracy, accuracy_all, f_measure, f_measure_all, f_beta_measure, f_beta_all, g_beta_measure, g_all, challenge_metric, confusion_matrices = evaluate_12ECG_score(args, output_directory)

