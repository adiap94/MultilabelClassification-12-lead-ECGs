#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings
import driver
import torch
from torch import nn
from torch import optim
import numpy as np
import math

import models
import dataset
from loss.ASL_loss import AsymmetricLoss , AsymmetricLossOptimized
from kornia.losses.focal import BinaryFocalLossWithLogits
from utils.save import Save_Tool
from utils.freeze import set_freeze_by_id
from utils.metrics import *
from loss.focal_loss import binary_focal_loss
from loss.weight_BCE_loss import WeightedMultilabel
from loss.LDAMLoss import LDAMLoss
from loss.DiceLoss import DiceLoss
from loss.DiceLoss import TverskyLoss
# from monai.networks.nets import UNETR
from models.transformer import UNETR
from models.TransformerModel import TransformerModel
from transformers import AutoModelForImageClassification
# from transformers_main.src.transformers import AutoModelForImageClassification
def nn_forward(model, sigmoid, criterion, inputs, ag, labels):
    logits = model(inputs, ag)
    logits_prob = sigmoid(logits)
    loss = criterion(logits, labels)
    loss_temp = loss.item() * inputs.size(0)
    return logits_prob, loss_temp


class train_utils(object):
    def __init__(self, args):
        self.args = args
        # self.save_dir = save_dir
        self.out_dir = args.workdir
        self.csvLoggerFile_path = os.path.join(self.out_dir, "history.csv")
        self.epoch_log = {}
        self.best_metric = np.inf
    def setup(self):
        """
        Initialize the dataset, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        self.device = torch.device("cuda:0")



        # Load the dataset
        Dataset = getattr(dataset, args.data_name)
        self.dataset = {}
        self.dataset['train'], self.dataset['val'] = Dataset(args).data_preprare()
        self.dataloaders = {
            x: torch.utils.data.DataLoader(self.dataset[x], batch_size=(args.batch_size if x == 'train' else 1),
                                           shuffle=(True if x == 'train' else False),
                                           num_workers=args.num_workers,
                                           pin_memory=(True if self.device == 'cuda' else False),
                                           drop_last=True)
            for x in ['train', 'val']}

        # Define the model
        self.num_classes = Dataset.num_classes
        if args.model_name=="seresnet18_1d_ag":
            self.model = getattr(models, args.model_name)(in_channel=len(args.leads), out_channel=Dataset.num_classes)
        elif args.model_name=="UNETR":
            self.model = UNETR(in_channels=2, out_channels=1,
                          img_size=(12, 4096),  norm_name='batch', spatial_dims=2,feature_size=4,num_heads=8,hidden_size=4096)
        elif args.model_name =="TransformerModel":
            # self.model = TransformerModel(64, 4, 256, 4, 2, 24, 0.25, 0.1)
            self.model = TransformerModel(2, 2, 8, 4, 2, 24, 0.25, 0.1)
            print("hello")
        elif args.model_name == "TransformerHF":
            # self.model = TransformerModel(2, 2, 8, 4, 2, 24, 0.25, 0.1)
            self.model = AutoModelForImageClassification.from_pretrained(
                "google/vit-base-patch16-224-in21k",
                num_labels=24
                # id2label=id2label,
                # label2id=label2id,
            )


        # self.model = getattr(models, args.model_name)()
        # self.model.fc = torch.nn.Linear(self.model.fc.in_features, Dataset.num_classes)
        # parameter_list = self.model.parameter_list(args.lr)

        # if args.layer_num_last != 0:
        #     set_freeze_by_id(self.model, args.layer_num_last)
        # if self.device_count > 1:
        #     self.model = torch.nn.DataParallel(self.model)

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'cos':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, steps, 0)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        # Define the monitoring accuracy
        if args.monitor_acc == 'acc':
            self.cal_acc = None
        elif args.monitor_acc == 'AUC':
            self.cal_acc = RocAucEvaluation
        elif args.monitor_acc == 'ecgAcc':
            self.cal_acc = cal_Acc
        else:
            raise Exception("monitor_acc is not implement")

        # Load the checkpoint
        self.start_epoch = 0
        if args.load_model_bool:
            print('load the model:' + args.load_model)
            self.model.load_state_dict(torch.load(args.load_model))

        #self.criterion = DiceLoss()
        if args.loss=="BCE":
            self.criterion = nn.BCEWithLogitsLoss()
        elif args.loss=="BCE-focalLoss":
            kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
            self.criterion  = BinaryFocalLossWithLogits(**kwargs)
        elif args.loss =="KL":
            self.criterion = nn.KLDivLoss(reduction="batchmean")
        elif args.loss == "AsymmetricLoss":
            self.criterion = AsymmetricLoss()
        elif args.loss == "AsymmetricLossOptimized":
            self.criterion = AsymmetricLossOptimized()
        #self.criterion = TverskyLoss()
        self.sigmoid = nn.Sigmoid()
        self.sigmoid.to(self.device)
        self.model.to(self.device)

    def writeCSVLoggerFile(self):
        df = pd.DataFrame([self.epoch_log])
        with open(self.csvLoggerFile_path, 'a') as f:
            df.to_csv(f, mode='a', header=f.tell() == 0, index=False)

    def run_test(self, epoch):
        if self.args.run_test:
            if epoch == self.args.max_epoch - 1 :
                print("run test")
                driver.main(self.args.workdir, gpu_num=self.args.gpu, only_eval_bool=False)

    def save_model_checkpoint(self):

        # if self.epoch_log["challenge_metric_val"] > self.best_metric:
        #     self.best_metric = self.epoch_log["challenge_metric_val"]
        if self.epoch_log["epoch_loss_val"] < self.best_metric:
            self.best_metric = self.epoch_log["epoch_loss_val"]

            PATH = os.path.join(self.args.workdir, "Models", "best_metric_model.pt")
            torch.save(self.model, PATH)
            print("saved new best metric model")

    def train(self):
        """
        Training process
        :return:
        """
        args = self.args

        step = 0
        best_acc = 0.0
        batch_count = 0

        batch_loss = 0.0
        step_start = time.time()

        for epoch in range(self.start_epoch, args.max_epoch):
            self.epoch_log = {}
            self.epoch_log["epoch"] = epoch
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_length = 0
                batch_length = 0
                batch_acc = 0

                # Set model to train mode or test mode
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                for batch_idx, (inputs, ag, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    ag = ag.to(self.device)
                    labels = labels.to(self.device)
                    # Do the learning process, in val, we do not care about the gradient for relaxing
                    with torch.set_grad_enabled(phase == 'train'):
                        # forward
                        if phase == 'train':
                            if args.model_name=="seresnet18_1d_ag":
                                logits = self.model(inputs, ag)
                            elif args.model_name == "TransformerHF":
                                inputs=inputs.unsqueeze(1)
                                logits = self.model(inputs)
                            else:
                                # inputs = inputs.unsqueeze(0)
                                self.model(inputs)
                            #logits_prob = logits
                            logits_prob = self.sigmoid(logits)
                            if batch_idx == 0:
                                labels_all = labels
                                logits_prob_all = logits_prob
                            else:
                                labels_all = torch.cat((labels_all, labels), 0)
                                logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)

                            loss = self.criterion(logits, labels)
                            loss_temp = loss.item() * inputs.size(0)
                            epoch_loss += loss_temp
                        else:
                            val_length = inputs.shape[2]
                            win_length = 4096
                            overlap = 256
                            patch_number = math.ceil(abs(val_length - win_length) / (win_length - overlap)) + 1
                            if patch_number > 1:
                                start = int((val_length - win_length) / (patch_number - 1))
                            for i in range(patch_number):
                                if i == 0:
                                    logits_prob, loss_temp = nn_forward(self.model, self.sigmoid, self.criterion,
                                                                        inputs[:, :, 0: val_length], ag, labels)
                                elif i == patch_number-1:
                                    logits_prob_tmp, loss_temp_tmp = nn_forward(self.model, self.sigmoid, self.criterion,
                                                                        inputs[:, :, val_length - win_length : val_length], ag, labels)
                                    logits_prob = (logits_prob + logits_prob_tmp) / patch_number
                                    loss_temp = (loss_temp + loss_temp_tmp) / patch_number
                                else:
                                    logits_prob_tmp, loss_temp_tmp = nn_forward(self.model, self.sigmoid, self.criterion,
                                                                        inputs[:, :, i*start:i*start+win_length], ag, labels)
                                    logits_prob = logits_prob + logits_prob_tmp
                                    loss_temp = loss_temp + loss_temp_tmp

                            if batch_idx == 0:
                                labels_all = labels
                                logits_prob_all = logits_prob
                            else:
                                labels_all = torch.cat((labels_all, labels), 0)
                                logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)
                            epoch_loss += loss_temp

                        # Calculate the training information
                        if phase == 'train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            # batch_acc += accuracy
                            batch_count += inputs.size(0)
                            batch_length += 1
                            # Print the training information
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                # batch_acc = batch_acc / batch_length
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0 * batch_count / train_time
                                logging.info('Epoch: {} [{}/{}], Train Loss: {:.4f},'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_idx * len(inputs), len(self.dataloaders[phase].dataset),
                                    batch_loss, sample_per_sec, batch_time
                                ))
                                # batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1

                challenge_metric = self.cal_acc(labels_all, logits_prob_all, threshold=0.5, num_classes=self.num_classes)

                # Print the train and val information via each epoch
                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                logging.info('Epoch: {} {}-Loss: {:.4f} {}-challenge_metric: {:.4f}, Cost {:.1f} sec'.
                             format(epoch, phase, epoch_loss, phase, challenge_metric, time.time() - epoch_start))

                self.epoch_log["epoch_loss_"+phase] = epoch_loss
                self.epoch_log["challenge_metric_"+phase] = challenge_metric
                # This depends on the weights
                epoch_acc = challenge_metric

                # save the model
                # if phase == 'val':
                #     # save the checkpoint for other learning
                #     model_state_dic =  self.model.state_dict()
                #     # save the best model according to the val accuracy
                #     if epoch_acc > best_acc:
                #         best_acc = epoch_acc
                #         logging.info("save best model epoch {}, CM: {:.4f}".
                #                      format(epoch, challenge_metric))
                #         torch.save(model_state_dic,
                #                    os.path.join(self.args.workdir,"Models", '"best_metric_model.pth'))

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # update Logger File
            self.writeCSVLoggerFile()

            # save model checkpoint
            self.save_model_checkpoint()

            self.run_test(epoch)










