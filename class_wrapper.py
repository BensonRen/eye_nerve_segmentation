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
from sklearn.metrics import jaccard_score as jsc

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
        summary(model, input_size=(3, 512, 512))
        print(model)
        return model

    def dice_loss(self, pred, target, smooth=1.):
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=2).sum(dim=2)
        #print("start of pred", pred.detach().cpu().numpy()[0,:,0,0])
        #print("mean of pred", np.mean(pred.detach().cpu().numpy()))
        #print("mean of target", np.mean(target.detach().cpu().numpy()))
        #print("shape of target", np.shape(target.detach().cpu().numpy()))
        #print("shape of pred", np.shape(pred.detach().cpu().numpy()))
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
        #print("pred shape", np.shape(pred), "type", type(pred))
        #print("target shape", np.shape(target), "type", type(target))
        #print(pred)
        #print(target)
        bce = F.binary_cross_entropy_with_logits(pred, target)
        #print("start of pred", pred.detach().cpu().numpy()[0, :, 0, 0])
        pred_sigmoid = torch.sigmoid(pred)
        #print("start of pred_sigmoid", pred_sigmoid.detach().cpu().numpy()[0, :, 0, 0])
        dice = self.dice_loss(pred_sigmoid, target)
        loss = bce
        #loss = bce * bce_weight + dice * (1 - bce_weight)

        metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
        metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

        return loss

    def compute_iou(self, pred, target):
        """
        Compute the IOU
        """
        #print("shape of the original prediction", np.shape(pred.cpu().data.numpy()))
        lbl = pred.cpu().data.numpy().reshape(-1) > 0
        tgt = target.cpu().data.numpy().reshape(-1)
        #print("shape of lbl in compute iou", np.shape(lbl))
        #print("shape of tgt in compute iou", np.shape(tgt))
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
                                              patience=100, verbose=True, threshold=1e-4)

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

        # Set up the total number of training samples allowed to see
        total_training_samples = 0
        train_end_flag = False
        for epoch in range(self.flags.train_step):
            if train_end_flag:          # Training is ended due to max sample reached
                break;
            # Set to Training Mode
            epoch_samples = 0
            metrics = defaultdict(float)
            self.model.train()
            iou_sum_train = 0
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
                total_training_samples += inputs.size(0)

                # change from epoch base to mini-batch base
                if j % self.flags.eval_step == 0:
                    IoU = self.compute_iou(logit, labels)
                    iou_sum_train += IoU
                    IoU_aggregate = iou_sum_train/total_training_samples
                    self.print_metrics(metrics, epoch_samples, 'training')
                    print('training IoU in current batch {} is'.format(j), IoU)
                    print('training IoU uptillnow {} is'.format(j), IoU_aggregate)
                    self.log.add_scalar('training/bce', metrics['bce']/epoch_samples, j)
                    self.log.add_scalar('training/dice', metrics['dice']/epoch_samples, j)
                    self.log.add_scalar('training/loss', metrics['loss']/epoch_samples, j)
                    self.log.add_scalar('training/IoU', IoU_aggregate, j)
                    # Set eval mode
                    self.model.eval()
                    # Set to Training Mode
                    test_epoch_samples = 0
                    test_metrics = defaultdict(float)
                    iou_sum = 0
                    for jj, sample in enumerate(self.test_loader):
                        inputs = sample['image']                                # Get the input
                        labels = sample['labels']                               # Get the labels
                        if cuda:
                            inputs = inputs.cuda()                              # Put data onto GPU
                            labels = labels.cuda()                              # Put data onto GPU
                        self.optm.zero_grad()                                   # Zero the gradient first
                        logit = self.model(inputs.float())                        # Get the output
                        loss = self.make_loss(logit, labels, test_metrics)               # Get the loss tensor
                        test_epoch_samples += inputs.size(0)
                        IoU = self.compute_iou(logit, labels)
                        iou_sum += IoU
                        if test_epoch_samples > self.flags.max_test_sample:
                            break
                    IoU = iou_sum / test_epoch_samples
                    self.print_metrics(metrics, test_epoch_samples, 'testing')
                    print('IoU in current test batch is', IoU)
                    self.log.add_scalar('test/bce', test_metrics['bce']/test_epoch_samples, j)
                    self.log.add_scalar('test/dice', test_metrics['dice']/test_epoch_samples, j)
                    self.log.add_scalar('test/loss', test_metrics['loss']/test_epoch_samples, j)
                    self.log.add_scalar('test/IoU', IoU, j)
                    self.plot_eval_graph(inputs.cpu().numpy(), logit.detach().cpu().numpy(),
                                         labels.detach().cpu().numpy(), j)
                        #raise Exception("Testing stop point for getting shapes")
                        

                if loss.cpu().data.numpy() < self.best_validation_loss:
                    self.best_validation_loss = loss.cpu().data.numpy()
                # Learning rate decay upon plateau
                self.lr_scheduler.step(loss)
            
                if total_training_samples > self.flags.max_train_sample:
                    print("Maximum training samples requirement meet, I have been training for more than ", total_training_samples, " samples.")
                    train_end_flag = True
                    break;

        self.log.close()
        tk.record(999)                    # Record at the end of the training

        # Save the module at the end
        self.save()

    def plot_eval_graph(self, image_numpy, segment_output, gt_segment, batch_label):
        """
        The function to plot the evaluation figure to see the result
        :param image_numpy: The image tensor input [numpy array]
        :param segment_output: The segmented output from the network [numpy array]
        :param gt_segment:  The grount truth segmentation result [numpy array]
        :return: None, the graph plotted would be added to the tensorboard instead of returned
        """
        #print("shape of image numpy is", np.shape(image_numpy))
        #print("shape of segment output is", np.shape(segment_output))
        #print("shape of gt_segment is", np.shape(gt_segment))

        ##########################################
        # Plot the original image as a reference #
        ##########################################
        f = plt.figure()
        plt.imshow(image_numpy[0, 0, :, :])
        plt.title('original image')

        ##########################################
        # Plot the binary mask im as a reference #
        ##########################################
        h = plt.figure()
        plt.imshow(gt_segment[0, 0, :, :])
        plt.title('binary mask image')

        ############################
        # Plot the confusion image #
        ############################
        # Stage-1: new confusion map
        g = plt.figure()
        # Create the confusion map
        confusion_map = np.zeros([512, 512, 3])
        # Stage-2: add ground truth to the first channel
        confusion_map[:, :, 0] = gt_segment[0, 0, :, :]
        # Stage-3: add prediction map to the second channel and add legend
        prediction = segment_output[0, 0, :, :] > 0.5
        confusion_map[:, :, 1] = prediction
        plt.imshow(confusion_map)
        # Add the legend for different colors
        plt.plot(0, 0, "s", c='y', label='True positive')
        plt.plot(0, 0, "s", c='g', label='False positive')
        plt.plot(0, 0, "s", c='k', label='True negative')
        plt.plot(0, 0, "s", c='r', label='False negative')
        plt.title('confusion map')
        plt.legend()

        # Stage-4: Add that to tensorboard
        self.log.add_figure('Confusion Sample', g, global_step=batch_label)
        self.log.add_figure('Original Image', f, global_step=batch_label)
        self.log.add_figure('Gt label', h, global_step=batch_label)
        """
        # The debugging phase
        np.save('image.npy', image_numpy)
        np.save('segment_out.npy', segment_output)
        np.save('gt_segment.npy', gt_segment)
        """


