#!/usr/bin/env python
import argparse
import os
from collections import namedtuple
from utils.logger import setlogger
import logging
from utils.train_utils_clip_ag import train_utils
import utiles
import torch

def train_12ECG_classifier():
    # Load Configuration File
    work_dir = utiles.create_working_folder()
    json_config_file_path = os.path.join(work_dir, 'config.json')
    args = utiles.json_load(json_config_file_path)
    args = namedtuple('Struct', args.keys())(*args.values())
    #define gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0")

    setlogger(os.path.join(args.workdir, 'train.log'))
    trainer = train_utils(args)
    trainer.setup()
    trainer.train()




def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    # basic parameters
    parser.add_argument('--model_name', type=str, default='seresnet18_1d_ag', help='the name of the model')
    parser.add_argument('--output_directory', type=str, default='/tcmldrive/project_dl/results')
    parser.add_argument('--train_data_csv', type=str, default='/tcmldrive/project_dl/db/train_split5.csv')
    parser.add_argument('--val_data_csv', type=str, default='/tcmldrive/project_dl/db/val_split5.csv')
    parser.add_argument('--data_name', type=str, default='ECGag', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default='./', help='the directory of the data')
    parser.add_argument('--split', type=str, default='0', help='The number of split')
    parser.add_argument('--monitor_acc', type=str, default='ecgAcc', help='the directory of the data')
#    parser.add_argument('--label_dir', type=str, default='/Users/michael', help='the directory of the labels')
    #parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint_seresnet18_clip_1d_split0',
                        help='the directory to save the model')
#    parser.add_argument("--pretrained", type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    # optimization information
    parser.add_argument('--layer_num_last', type=int, default=0, help='the number of last layers which unfreeze')
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.0003, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix', 'cos'], default='step',
                        help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='2', help='the learning rate decay for step and stepLR')

    # save, load and display information
    # parser.add_argument('--load_model', type=str, default='./load_model/48-0.6294-best_model.pth', help='the directory of the resume training model')
    parser.add_argument('--load_model', type=str, default=None, help='the directory of the resume training model')
    parser.add_argument('--load_model_bool', type=bool, default=False)
    parser.add_argument('--max_epoch', type=int, default=4, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=200, help='the interval of log training information')
    parser.add_argument('--debug_mode', type=bool, default=False)

    args, _ = parser.parse_known_args()
    return args

