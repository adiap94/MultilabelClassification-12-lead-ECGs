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

def main(run_dir, only_eval_bool, gpu_num = "0"):

    output_directory = os.path.join(run_dir,"test")
    config_path = os.path.join(run_dir,"config.json")
    args = utiles.json_load(config_path)
    args = namedtuple('Struct', args.keys())(*args.values())

    if (not only_eval_bool):

        model_list = [os.path.join(run_dir,"Models",p) for p in os.listdir(os.path.join(run_dir,"Models"))]
        model_path = model_list[-1]


        #define gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
        device = torch.device("cuda:0")

        df = pd.read_csv(args.test_data_csv)
        input_files = df.mat_path.to_list()

        # debug
        # input_files = input_files[:2]

        os.makedirs(output_directory, exist_ok=True)
        os.chmod(output_directory, mode=0o777)

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

    print('saving evaluation methods...')

    classes, auroc, auprc, auroc_all, auprc_all, accuracy, accuracy_all, f_measure, f_measure_all, f_beta_measure, f_beta_all, g_beta_measure, g_all, challenge_metric, confusion_matrices = evaluate_12ECG_score(args, output_directory)

    cm = confusion_matrices.reshape(24,4)
    cm = pd.DataFrame(cm, columns = ['TP', 'FP', 'FN', 'TN'])
    cm.to_csv(os.path.join(run_dir,"confusion_matrices.csv"))

    precision, sensitivity, specificity = np.zeros(cm.shape[0]), np.zeros(cm.shape[0]), np.zeros(cm.shape[0])

    for i in range (cm.shape[0]):
        precision[i] = cm['TP'][i]/(cm['TP'][i]+cm['FP'][i])
        sensitivity[i] = cm['TP'][i]/(cm['TP'][i]+cm['FN'][i])  #recall
        specificity[i] = cm['TN'][i]/(cm['TN'][i]+cm['FP'][i])

    all_class_analayze = np.array([sensitivity, specificity, precision, accuracy_all, auroc_all, auprc_all, f_measure_all, f_beta_all, g_all]).T
    all_class_analayze = pd.DataFrame(all_class_analayze, columns = ['sensitivity', 'specificity','precision', 'accuracy', 'auroc', 'auprc', 'f_measure', 'f_beta_measure', 'g_beta_measure'], index = classes)
    all_class_analayze.to_csv(os.path.join(run_dir,"analyze_all_classes_2.csv"))

    eval_res = np.array([[auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric, np.nanmean(precision), np.nanmean(sensitivity), np.nanmean(specificity)]])
    eval_res = pd.DataFrame(eval_res, columns = ['auroc', 'auprc', 'accuracy', 'f_measure', 'f_beta_measure', 'g_beta_measure', 'challenge_metric', 'precision', 'sensitivity', 'specificity'], index = classes)
    eval_res.to_csv(os.path.join(run_dir,"metrics.csv"))

    print('Done.')

if __name__ == '__main__':

    main(run_dir = "/tcmldrive/project_dl/results/20220623-201217", only_eval_bool = True)
    pass
