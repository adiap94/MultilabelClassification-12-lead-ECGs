#!/usr/bin/python
# -*- coding:utf-8 -*-
from dataset.ECGDatasetsag import dataset
import pandas as pd
from dataset.sequence_aug import *

normlizetype = 'none'
start = 0
seq_length = 4096
sample_ratio = 0.5
data_transforms = {
    'train': Compose([
        #Reshape(),
        #DownSample(sample_ratio),
        #ZerosPadding(len=seq_length),
        # ConstantStart(start=start, num=seq_length),
        RandomClip(len=seq_length),
        Normalize(normlizetype),
        # RandomAddGaussian(),
        # RandomScale(0.1),
        # RandomAmplify(),
        # Randomverflip(),
        # Randomshift(),
        # RandomStretch(0.02),
        # RandomCrop(),
        Retype()
    ]),
    'val': Compose([
        #Reshape(),
        #DownSample(sample_ratio),
        #RandomClip(len=seq_length),
        ValClip(len=seq_length),
        #ZerosPadding(len=seq_length),
        # ConstantStart(start=start, num=seq_length),
        Normalize(normlizetype),
        Retype()
    ]),
    'test': Compose([
        #Reshape(),
        #DownSample(sig_resample_len),
        ValClip(len=seq_length),
        Normalize(normlizetype),
        Retype()
    ])
}


def load_and_clean_sub(path):
    label_test = pd.read_csv(path, sep="\t", names=list(range(0, 3)))
    label_test.columns = ['id', 'age', 'gender']  # 设置列名
    return label_test




class ECG(object):
    num_classes = 24
    inputchannel = 12


    def __init__(self,args):
        # self.data_dir = data_dir
        # self.split = splits
        self.args = args



    def data_preprare(self, test=False):
        if test:
            train_path = './data_split/train_split' + self.split + '.csv'
            val_path = './data_split/test_split' + self.split + '.csv'
            train_pd = pd.read_csv(train_path)
            val_pd = pd.read_csv(val_path)

            train_dataset = dataset(anno_pd=train_pd, transform=data_transforms['train'], data_dir=self.data_dir,args=self.args)
            val_dataset = dataset(anno_pd=val_pd, transform=data_transforms['val'], data_dir=self.data_dir,args=self.args)
            return train_dataset, val_dataset
        else:
            # train_path = './data_split/train_split' + self.split + '.csv'
            # val_path = './data_split/test_split' + self.split + '.csv'
            train_path = self.args.train_data_csv
            val_path = self.args.val_data_csv
            train_pd = pd.read_csv(train_path)
            val_pd = pd.read_csv(val_path)
            if self.args.debug_mode:
                train_pd = train_pd.head()
                val_pd = val_pd.head()
            train_dataset = dataset(anno_pd=train_pd, transform=data_transforms['train'],args=self.args)
            val_dataset = dataset(anno_pd=val_pd, transform=data_transforms['val'],args=self.args)
            return train_dataset, val_dataset




