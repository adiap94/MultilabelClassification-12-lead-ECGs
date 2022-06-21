#!/usr/bin/env python
import torch
import numpy as np, os, sys
from scipy.io import loadmat
from run_12ECG_classifier import load_12ECG_model, run_12ECG_classifier
from collections import namedtuple
import utiles
import pandas as pd
from utils.metrics import *

def load_challenge_data(filename):

    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file,'r') as f:
        header_data=f.readlines()


    return data, header_data

def save_challenge_predictions(output_directory,filename,scores,labels,classes):
    basename = os.path.basename(filename)
    recording = os.path.splitext(basename)[0]
    new_file = basename.replace('.mat','.csv')
    output_file = os.path.join(output_directory,new_file)

    # Include the filename as the recording number
    recording_string = '#{}'.format(recording)
    class_string = ','.join(classes)
    label_string = ','.join(str(i) for i in labels)
    score_string = ','.join(str(i) for i in scores)

    with open(output_file, 'w') as f:
        f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')

def main(run_dir,gpu_num = "0"):

    config_path = os.path.join(run_dir,"config.json")
    args = utiles.json_load(config_path)
    args = namedtuple('Struct', args.keys())(*args.values())

    model_list = [os.path.join(run_dir,"Models",p) for p in os.listdir(os.path.join(run_dir,"Models"))]
    model_path = model_list[-1]

    output_directory = os.path.join(run_dir,"test")

    #define gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    device = torch.device("cuda:0")

    df = pd.read_csv(args.test_data_csv)
    input_files = df.mat_path.to_list()

    # debug
    # input_files = input_files[:2]

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    # Load model.
    print('Loading 12ECG model...')
    model = load_12ECG_model(model_path=model_path,device=device)

    # Iterate over files.
    print('Extracting 12ECG features...')
    num_files = len(input_files)

    for i, f in enumerate(input_files):
        print('    {}/{}...'.format(i+1, num_files))
        tmp_input_file = f
        data,header_data = load_challenge_data(tmp_input_file)
        current_label, current_score, classes = run_12ECG_classifier(data, header_data, model,args)
        # Save results.
        save_challenge_predictions(output_directory,f,current_score,current_label,classes)

    # evaluation metrics
    auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric, confusion_matrices = evaluate_12ECG_score(args, output_directory)

    cm = confusion_matrices.reshape(24,4)
    cm = pd.DataFrame(cm, columns = ['TP', 'FP', 'FN', 'TN'])
    cm.to_csv(os.path.join(run_dir,"confusion_matrices.csv"))

    precision, sensitivity, specificity = np.zeros(cm.shape[0]), np.zeros(cm.shape[0]), np.zeros(cm.shape[0])

    for i in range (cm.shape[0]):
        precision[i] = cm['TP'][i]/(cm['TP'][i]+cm['FP'][i])
        sensitivity[i] = cm['TP'][i]/(cm['TP'][i]+cm['FN'][i])  #recall
        specificity[i] = cm['TP'][i]/(cm['TP'][i]+cm['FN'][i])

    sen_spe = np.array([[sensitivity, specificity]])
    sen_spe = pd.DataFrame(sen_spe, columns = ['sensitivity', 'specificity'])
    sen_spe.to_csv(os.path.join(run_dir,"sen&spe_all.csv"))

    eval_res = np.array([[auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric, np.namean(precision), np.namean(sensitivity), np.namean(specificity)]])
    eval_res = pd.DataFrame(eval_res, columns = ['auroc', 'auprc', 'accuracy', 'f_measure', 'f_beta_measure', 'g_beta_measure', 'challenge_metric', 'precision', 'sensitivity', 'specificity'])
    eval_res.to_csv(os.path.join(run_dir,"metrics.csv"))

    print('Done.')

if __name__ == '__main__':

    main(run_dir = "/tcmldrive/project_dl/results/debug_mode/20220620-215139/" )
    pass
