"""
The class wrapper for the networks
"""
# Built-in
import os
import time
import sys
from collections import defaultdict
from math import inf

# Torch
import torch
from torch import nn
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

# Libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import jaccard_similarity_score as jsc

# Own module
from utils.time_recorder import time_keeper
class Network(object):
    def __init__(self, model_fn, flags, train_loader, test_loader,
                 ckpt_dir=os.path.join(os.path.abspath(''), 'models'),
                 inference_mode=False, saved_model=None):
        self.model_fn = model_fn                                # The model maker function
        self.flags = flags                                      # The Flags containing the specs
        if inference_mode:                                      # If inference mode, use saved model
            if saved_model.startswith('models/'):
                saved_model = saved_model.replace('models/','')
            self.ckpt_dir = os.path.join(ckpt_dir, saved_model)
            self.saved_model = saved_model
            print("This is inference mode, the ckpt is", self.ckpt_dir)
        else:                                                   # training mode, create a new ckpt folder
            if flags.model_name is None:                    # leave custume name if possible
                self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
            else:
                self.ckpt_dir = os.path.join(ckpt_dir, flags.model_name)
        self.model = self.create_model()                        # The model itself
        self.optm = None                                        # The optimizer: Initialized at train() due to GPU
        self.optm_eval = None                                   # The eval_optimizer: Initialized at eva() due to GPU
        self.lr_scheduler = None                                # The lr scheduler: Initialized at train() due to GPU
        self.train_loader = train_loader                        # The train data loader
        self.test_loader = test_loader                          # The test data loader
        self.log = SummaryWriter(self.ckpt_dir)     # Create a summary writer for keeping the summary to the tensor board
        self.best_validation_loss = float('inf')    # Set the BVL to large number

    def create_model(self):
        """
        Function to create the network module from provided model fn and flags
        :return: the created nn module
        """
        model = self.model_fn(self.flags)
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            model.cuda()
        summary(model, input_size=(3, 224, 224))
        print(model)
        return model

    def dice_loss(self, pred, target, smooth=1.):
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=2).sum(dim=2)

        loss = (1 - ((2. * intersection + smooth) / (
                    pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

        return loss.mean()

    def make_loss(self, pred, target, metrics, bce_weight=0.5):
        """
        Create a tensor that represents the loss. This is consistant both at training time \
        and inference time for Backward model
        :param logit: The output of the network
        :param labels: The ground truth labels
        :return: the total loss
        """
        import torch.nn.functional as F
        print("pred shape", np.shape(pred), "type", type(pred))
        print("target shape", np.shape(target), "type", type(target))
        print(pred)
        print(target)
        bce = F.binary_cross_entropy_with_logits(pred, target)
        pred = torch.sigmoid(pred)
        dice = self.dice_loss(pred, target)
        loss = bce * bce_weight + dice * (1 - bce_weight)

        metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
        metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

        return loss

    def compute_iou(self, pred, target):
        """
        Compute the IOU
        """
        print("shape of the original prediction", np.shape(pred.cpu().data.numpy()))
        lbl = pred.cpu().data.numpy()[:,0,:,:].reshape(-1) > 0
        tgt = target.cpu().data.numpy()[:,0,:,:].reshape(-1)
        print("shape of lbl in compute iou", np.shape(lbl))
        print("shape of tgt in compute iou", np.shape(tgt))
        return jsc(tgt, lbl)

    def print_metrics(self, metrics, epoch_samples, phase):
        outputs = []
        for k in metrics.keys():
            outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

        print("{}: {}".format(phase, ", ".join(outputs)))

    def make_optimizer(self):
        """
        Make the corresponding optimizer from the flags. Only below optimizers are allowed. Welcome to add more
        :return:
        """
        if self.flags.optim == 'Adam':
            op = torch.optim.Adam(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'RMSprop':
            op = torch.optim.RMSprop(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SGD':
            op = torch.optim.SGD(self.model.parameters(), lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        else:
            raise Exception("Your Optimizer is neither Adam, RMSprop or SGD, please change in param or contact Ben")
        return op

    def make_lr_scheduler(self, optm):
        """
        Make the learning rate scheduler as instructed. More modes can be added to this, current supported ones:
        1. ReduceLROnPlateau (decrease lr when validation error stops improving
        :return:
        """
        return lr_scheduler.ReduceLROnPlateau(optimizer=optm, mode='min',
                                              factor=self.flags.lr_decay_rate,
                                              patience=10, verbose=True, threshold=1e-4)

    def save(self):
        """
        Saving the model to the current check point folder with name best_model_forward.pt
        :return: None
        """
        # torch.save(self.model.state_dict, os.path.join(self.ckpt_dir, 'best_model_state_dict.pt'))
        torch.save(self.model, os.path.join(self.ckpt_dir, 'best_model_forward.pt'))

    def load(self):
        """
        Loading the model from the check point folder with name best_model_forward.pt
        :return:
        """
        # self.model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, 'best_model_state_dict.pt')))
        if torch.cuda.is_available():
            self.model = torch.load(os.path.join(self.ckpt_dir, 'best_model_forward.pt'))
        else:
            self.model = torch.load(os.path.join(self.ckpt_dir, 'best_model_forward.pt'), map_location=torch.device('cpu'))

    def train(self):
        """
        The major training function. This would start the training using information given in the flags
        :return: None
        """
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()

        # Construct optimizer after the model moved to GPU
        self.optm = self.make_optimizer()
        self.lr_scheduler = self.make_lr_scheduler(self.optm)

        # Time keeping
        tk = time_keeper(time_keeping_file=os.path.join(self.ckpt_dir, 'training time.txt'))

        for epoch in range(self.flags.train_step):
            # Set to Training Mode
            epoch_samples = 0
            metrics = defaultdict(float)
            # boundary_loss = 0                 # Unnecessary during training since we provide geometries
            self.model.train()
            for j, sample in enumerate(self.train_loader):
                inputs = sample['image']                                # Get the input
                labels = sample['labels']                               # Get the labels
                if cuda:
                    inputs = inputs.cuda()                              # Put data onto GPU
                    labels = labels.cuda()                              # Put data onto GPU
                self.optm.zero_grad()                                   # Zero the gradient first
                logit = self.model(inputs.float())                        # Get the output
                loss = self.make_loss(logit, labels, metrics)               # Get the loss tensor
                loss.backward()                                     # Calculate the backward gradients
                self.optm.step()                                    # Move one step the optimizer
                epoch_samples += inputs.size(0)

            if epoch % self.flags.eval_step:
                IoU = self.compute_iou(logit, labels)
                self.print_metrics(metrics, epoch_samples, 'training')
                print('training IoU in current epoch is', IoU)
                self.log.add_scalar('training/bce', metrics['bce'], epoch)
                self.log.add_scalar('training/dice', metrics['dice'], epoch)
                self.log.add_scalar('training/loss', metrics['loss'], epoch)
                self.log.add_scalar('training/IoU', IoU, epoch)
                # Set eval mode
                self.model.eval()
                # Set to Training Mode
                epoch_samples = 0
                test_metrics = defaultdict(float)
                for j, sample in enumerate(self.test_loader):
                    inputs = sample['image']                                # Get the input
                    labels = sample['labels']                               # Get the labels
                    if cuda:
                        inputs = inputs.cuda()                              # Put data onto GPU
                        labels = labels.cuda()                              # Put data onto GPU
                    self.optm.zero_grad()                                   # Zero the gradient first
                    logit = self.model(inputs.float())                        # Get the output
                    loss = self.make_loss(logit, labels, test_metrics)               # Get the loss tensor
                IoU = self.compute_iou(logit, labels)
                self.print_metrics(metrics, epoch_samples, 'training')
                print('IoU in current epoch is', IoU)
                self.log.add_scalar('test/bce', test_metrics['bce'], epoch)
                self.log.add_scalar('test/dice', test_metrics['dice'], epoch)
                self.log.add_scalar('test/loss', test_metrics['loss'], epoch)
                self.log.add_scalar('test/IoU', IoU, epoch)

            if loss.cpu().data.numpy() < self.best_validation_loss:
                self.best_validation_loss = loss.cpu().data.numpy()
            # Learning rate decay upon plateau
            self.lr_scheduler.step(loss)

        self.log.close()
        tk.record(1)                    # Record at the end of the training

        # Save the module at the end
        self.save()


