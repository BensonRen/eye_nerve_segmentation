"""
This file serves to hold helper functions that is related to the "Flag" object which contains
all the parameters during training and inference
"""
# Built-in
import argparse
# Libs

# Own module
from parameters import *

# Torch

def read_flag():
    """
    This function is to write the read the flags from a parameter file and put them in formats
    :return: flags: a struct where all the input params are stored
    """
    parser = argparse.ArgumentParser()
    # Data_Set parameter
    parser.add_argument('--test-ratio', default=TEST_RATIO, type=float, help='the ratio of the test set')

    # Optimizer Params
    parser.add_argument('--model-name', default=MODEL_NAME, type=str, help='model name to save')


    # Optimizer Params
    parser.add_argument('--optim', default=OPTIM, type=str, help='the type of optimizer that you want to use')
    parser.add_argument('--reg-scale', type=float, default=REG_SCALE, help='#scale for regularization of dense layers')
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (100)')
    parser.add_argument('--eval-step', default=EVAL_STEP, type=int, help='# steps between evaluations')
    parser.add_argument('--train-step', default=TRAIN_STEP, type=int, help='# steps to train on the dataSet')
    parser.add_argument('--lr', default=LEARN_RATE, type=float, help='learning rate')
    parser.add_argument('--lr-decay-rate', default=LR_DECAY_RATE, type=float,
                        help='decay learn rate by multiplying this factor')
    parser.add_argument('--stop_threshold', default=STOP_THRESHOLD, type=float,
                        help='The threshold below which training should stop')
    #    parser.add_argument('--decay-step', default=DECAY_STEP, type=int,
    #                        help='decay learning rate at this number of steps')

    # Data Specific Params
    parser.add_argument('--img-l', type=int, default=IMG_L, help='Length of the image')
    parser.add_argument('--img-w', type=int, default=IMG_W, help='Width of the image')
    parser.add_argument('--root-dir', type=str, default=ROOT_DIR, help='The Root directory to get the images')
    parser.add_argument('--label-file', type=str, default=LABEL_FILE, help='The label file to get the list of names')
    parser.add_argument('--cut-square', type=bool, default=CUT_SQUARE, help='the flag to cut the image into 496 x 496')



    flags = parser.parse_args()  # This is for command line version of the code
    # flags = parser.parse_args(args = [])#This is for jupyter notebook version of the code
    # flagsVar = vars(flags)
    return flags
