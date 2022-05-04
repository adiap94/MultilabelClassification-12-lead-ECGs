#!/usr/bin/env python
import torch
import numpy as np, os, sys
from scipy.io import loadmat
from run_12ECG_classifier import load_12ECG_model, run_12ECG_classifier
from collections import namedtuple
import utiles
import pandas as pd

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
    model = load_12ECG_model(model_path,device)

    # Iterate over files.
    print('Extracting 12ECG features...')
    num_files = len(input_files)

    for i, f in enumerate(input_files):
        print('    {}/{}...'.format(i+1, num_files))
        tmp_input_file = f
        data,header_data = load_challenge_data(tmp_input_file)
        current_label, current_score, classes = run_12ECG_classifier(data, header_data, model)
        # Save results.
        save_challenge_predictions(output_directory,f,current_score,current_label,classes)


    print('Done.')
if __name__ == '__main__':

    main(run_dir = "/tcmldrive/project_dl/results/restore/" )
    pass
