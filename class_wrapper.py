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
from sklearn.metrics import roc_curve,auc
import cv2

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
        print(model)
        summary(model, input_size=(3, 512, 512))
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

    def make_loss(self, pred, target, metrics, bce_weight=0.5, boundary_weight=1, boundary_width=3):
        """
        Create a tensor that represents the loss. This is consistant both at training time \
        and inference time for Backward model
        :param logit: The output of the network
        :param labels: The ground truth labels
        :param metrics: The metrics to save
        :param bce_weight: The weight of binary cross entropy loss (1 is without dice loss)
        :param boundary_width: The size of the morphological kernel (Defalut: 3)
        :param boundary_weight: The weight on the loss boundary
        :return: the total loss
        """
        import torch.nn.functional as F
        if boundary_weight != 0:
            # Add the boundary weight to the training
            target_numpy = target.cpu().numpy()
            kernel = np.ones((boundary_width, boundary_width), np.uint8)
            # Using morphological gradient to do the boundary weighting
            for i in range(len(target_numpy)):
                target_numpy[i,0,:,:] = cv2.morphologyEx(target_numpy[i,0,:,:], cv2.MORPH_GRADIENT, kernel)
            weight = np.ones_like(target_numpy) + boundary_weight * target_numpy
            bce = F.binary_cross_entropy_with_logits(weight=torch.tensor(weight, requires_grad=False),
                                                     input=pred, target=target)
        else:
            bce = F.binary_cross_entropy_with_logits(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        dice = self.dice_loss(pred_sigmoid, target)
        loss = bce * bce_weight + dice * (1 - bce_weight)

        metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
        metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

        return loss

    def compute_iou(self, pred, target):
        """
        Compute the IOU
        """
        if isinstance(pred, torch.Tensor):
            lbl = pred.cpu().data.numpy().reshape(-1) > 0
        else:   # this is a numpy array
            lbl = pred.reshape(-1) > 0

        if isinstance(target, torch.Tensor):
            tgt = target.cpu().data.numpy().reshape(-1)
        else:   # This is a numpy array
            tgt = target.reshape(-1)
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
        print("In model loading, the self.ckpt_dir is ", self.ckpt_dir)
        path = os.path.join(self.ckpt_dir, 'best_model_forward.pt')
        #path = self.ckpt_dir + 'best_model_forward.pt'
        if torch.cuda.is_available():
            self.model = torch.load(path)
        else:
            self.model = torch.load(path, map_location=torch.device('cpu'))

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
                loss = self.make_loss(logit, labels, metrics, 
                                      bce_weight=self.flags.bce_weight)               # Get the loss tensor
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
                    self.log.add_scalar('training/IoU', IoU, j)
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
                        loss = self.make_loss(logit, labels, test_metrics,
                                              bce_weight=self.flags.bce_weight)   # Get the loss tensor
                        test_epoch_samples += inputs.size(0)
                        IoU = self.compute_iou(logit, labels)
                        iou_sum += IoU
                        if test_epoch_samples > self.flags.max_test_sample:
                            break
                    IoU = iou_sum / (jj+1)
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
                    break

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
        prediction = segment_output[0, 0, :, :] > 0
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

    def evaluate(self, eval_number_max=10, save_img=False, post_processing=False, ROC=False, save_label=None):
        """
        Evaluate the trained model, output the IoU of the test case
        :param eval_number_max: The maximum number of images to evaluate
        :param save_img: Flag to save the image and binary mask
        :param post_processing: Do the post-processing as illustrated in the function post-processing
        :param ROC: Plot the ROC function and give the AUROC in the plot
        :param save_label: If save_img, save the label as the name of the image or not
        :return: IoU of the prediction in test case
        """
        self.load()
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        # Use evaluation mode for evaluate
        self.model.eval()

        auroc = -1 # dummy variable

        # Eval loop
        iou_sum = 0
        total_eval_num = 0
        label_list = []
        pred_list = []
        for j, sample in enumerate(self.test_loader):
            inputs = sample['image']    # Get the input
            labels = sample['labels']   # Get the labels
            name = sample['name']       # Get the name
            if cuda:
                inputs = inputs.cuda()  # Put data onto GPU
                labels = labels.cuda()  # Put data onto GPU
            logit = self.model(inputs.float())  # Get the output
            # Get those numpy version for later use
            input_numpy = inputs.cpu().numpy()
            labels_numpy = labels.cpu().numpy()
            logit_numpy = logit.detach().cpu().numpy()
            if post_processing:
                logit_numpy = self.post_processing(logit_numpy)
            if ROC:         # If calculating ROC, put those into the list
                label_list.append(labels_numpy)
                pred_list.append(logit_numpy)
            if save_img:                        # If choose to save the evaluation images
                self.save_eval_image(input_numpy, labels_numpy, logit_numpy,
                                     batch_num=j, save_label=name)
            batch_IoU = self.compute_iou(logit_numpy, labels_numpy)  # Get the batch IoU
            iou_sum += batch_IoU  # Aggregate the batch IoU
            total_eval_num += inputs.size(0)
            if total_eval_num > eval_number_max:    # Reached the limit of inference
                break
        average_iou = iou_sum/(j+1)
        print("The average IoU of your evaluation is: ", average_iou)
        if ROC:
            auroc =  self.plot_ROC(label_list, pred_list)
            print("The AUROC of the prediction is :", auroc)
        return average_iou, auroc

    def plot_ROC(self, label_list, pred_list):
        """
        Plot the ROC curve from the given label and prediction list
        :param label_list: The list of labels in format of numpy arrays of shape [batch_size, 0, w, l]
        :param pred_list: The list of predictions in format of numpy arrays of shape [batch_size, 0, w, l]
        :return: AUROC : Area under the ROC curve
        """
        label = np.reshape(np.array(label_list), [-1, 1])
        pred = np.reshape(np.array(pred_list), [-1, 1])
        fpr, tpr, threshold = roc_curve(label, pred, pos_label=1)
        auroc = auc(fpr, tpr)
        f = plt.figure()#figsize=[15,15])
        plt.plot(fpr, tpr, lw=3, c='b',label='ROC curve')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                label='Chance', alpha=.8)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve with AUROC = {}'.format(auroc))
        #plt.xlim()
        plt.legend()
        plt.savefig(os.path.join('data', self.ckpt_dir.replace('models','') + 'ROC.jpg'))
        return auroc

    def save_eval_image(self, inputs, labels, logit, batch_num, save_dir='data/', save_label=None):
        """
        Plot and save the evaluation image for the evaluation
        :param inputs: The input raw images
        :param labels: The gt labels read from
        :param logit: The output predictions from the model
        :param batch_num: The batch number to record
        :param save_dir: The direction to save
        :param save_label: If save_img, save the label as the name of the image or not
        :return: None
        """
        for i in range(np.shape(inputs)[0]):
            f = plt.figure(figsize=[15,15])
            ##########################################
            # Plot the original image as a reference #
            ##########################################
            ax = plt.subplot(221)
            plt.imshow(inputs[i, 0, :, :])
            plt.title('original')
            #####################################
            # Plot the ground truth binary mask #
            #####################################
            ax = plt.subplot(222)
            plt.imshow(labels[i, 0, :, :])
            plt.title('gt')
            ###################################
            # Plot the generated segmentation #
            ###################################
            ax = plt.subplot(223)
            plt.imshow(logit[i, 0, :, :] > 0)
            plt.title('prediction')
            #####################################
            # Plot the confusion matrix / image #
            #####################################
            ax = plt.subplot(224)
            confusion_map = np.zeros([512, 512, 3])
            # Stage-2: add ground truth to the first channel
            confusion_map[:, :, 0] = labels[i, 0, :, :]
            # Stage-3: add prediction map to the second channel and add legend
            prediction = logit[i, 0, :, :] > 0
            confusion_map[:, :, 1] = prediction
            plt.imshow(confusion_map)
            # Add the legend for different colors
            plt.plot(0, 0, "s", c='y', label='True positive')
            plt.plot(0, 0, "s", c='g', label='False positive')
            plt.plot(0, 0, "s", c='k', label='True negative')
            plt.plot(0, 0, "s", c='r', label='False negative')
            plt.title('confusion map')
            plt.legend()

            ##################
            # Save the image #
            ##################
            if save_label is None:
                f.savefig(os.path.join(save_dir, 'eval_graph_{}_{}.jpg'.format(batch_num, i)))
            else:
                f.savefig(os.path.join(save_dir, save_label[i].replace('.jpg', '') + '_segmentation_result.jpg'))
            # Debuggin purpose to save the array
            #np.save('image.npy', inputs[0,0,:,:])
            #np.save('segment_out.npy', logit[0,0,:,:])
            #np.save('gt_segment.npy', labels[0,0,:,:])
        return None

    def post_processing(self, logit, operation_style='open+close',
                        erosion_dialation_kernel_size=3, erosion_dialation_iteration=3):
        """
        The post-processing of the predicted segmentation map, current process techniques:
        1. CV2.erode + CV2.dialate (Remove small noise)
        :param logit: The predicted segmentation map which is in NUMPY format
        :param operation_style: The morphological operation for post-processing, now we only support
                open, close and open+close / close+open
        :param erosion_dialation_iteration: Number of iterations for erosion and dialation
        :param erosion_dialation_kernel_size: The kernel size for erosion and dialation
        :return: The post_processed map
        """
        kernel = np.ones((erosion_dialation_kernel_size, erosion_dialation_kernel_size), np.uint8)
        # Doing Morphological Close operation
        if operation_style == 'close':
            for i in range(len(logit)):
                img_erosion = cv2.erode(logit[i,0,:,:], kernel, iterations=erosion_dialation_iteration)
                logit[i,0,:,:] = cv2.dilate(img_erosion, kernel, iterations=erosion_dialation_iteration)
            return logit
        # Doing Morphological Open operation
        elif operation_style == 'open':
            for i in range(len(logit)):
                img_dilation  = cv2.dilate(logit[i,0,:,:], kernel, iterations=erosion_dialation_iteration)
                logit[i,0,:,:] = cv2.erode(img_dilation, kernel, iterations=erosion_dialation_iteration)
            return logit
        # Doing Morphological Open + close operation, recursive structure of calling myself
        elif operation_style == 'open+close':
            logit = self.post_processing(logit, operation_style='open', erosion_dialation_kernel_size, erosion_dialation_ieration) 
            logit = self.post_processing(logit, operation_style='close', erosion_dialation_kernel_size, erosion_dialation_ieration) 
            return logit
        # Doing Morphological close + open operation, recursive structure of calling myself
        elif operation_style == 'close+open':
            logit = self.post_processing(logit, operation_style='close', erosion_dialation_kernel_size, erosion_dialation_ieration) 
            logit = self.post_processing(logit, operation_style='open', erosion_dialation_kernel_size, erosion_dialation_ieration) 
            return logit
        else:
            raise Exception("Your post-processing operation_style has to be one of: Open, close, open+close, close+open, please contact Ben")
                
            



