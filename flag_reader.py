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
    parser.add_argument('--network-backbone', default=NETWORK_BACKBONE, type=str, help='the backbone of the Unet encoder')

    # Optimizer Params
    parser.add_argument('--model-name', default=MODEL_NAME, type=str, help='model name to save')
    parser.add_argument('--eval-model', default=EVAL_MODEL, type=str, help='The model to evaluate (load)')

    # Optimizer Params
    parser.add_argument('--optim', default=OPTIM, type=str, help='the type of optimizer that you want to use')
    parser.add_argument('--pretrain', default=PRETRAIN, type=str, help='the option to use pretrained model or not')
    parser.add_argument('--reg-scale', type=float, default=REG_SCALE, help='#scale for regularization of dense layers')
    parser.add_argument('--batch-size', default=BATCH_SIZE, type=int, help='batch size (100)')
    parser.add_argument('--eval-step', default=EVAL_STEP, type=int, help='# steps between evaluations')
    parser.add_argument('--train-step', default=TRAIN_STEP, type=int, help='# steps to train on the dataSet')
    parser.add_argument('--max-train-sample', default=MAX_TRAIN_SAMPLE, type=int, help='The maximum training samples to see')
    parser.add_argument('--max-test-sample', default=MAX_TEST_SAMPLE, type=int, help='The maximum testing samples to see')
    parser.add_argument('--lr', default=LEARN_RATE, type=float, help='learning rate')
    parser.add_argument('--lr-decay-rate', default=LR_DECAY_RATE, type=float,
                        help='decay learn rate by multiplying this factor')
    parser.add_argument('--stop_threshold', default=STOP_THRESHOLD, type=float,
                        help='The threshold below which training should stop')
    #    parser.add_argument('--decay-step', default=DECAY_STEP, type=int,
    #                        help='decay learning rate at this number of steps')

    # Data Specific Params
    parser.add_argument('--random-seed', type=int, default=RANDOM_SEED, help='The random SEED for data')
    parser.add_argument('--img-l', type=int, default=IMG_L, help='Length of the image')
    parser.add_argument('--img-w', type=int, default=IMG_W, help='Width of the image')
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS, help='number of workers to  load the pictures')
    parser.add_argument('--train-root-dir', type=str, default=TRAIN_ROOT_DIR, help='The Root directory to get the images')
    parser.add_argument('--train-label-file', type=str, default=TRAIN_LABEL_FILE, help='The label file to get the list of names')
    parser.add_argument('--test-root-dir', type=str, default=TEST_ROOT_DIR, help='The Root directory to get the images')
    parser.add_argument('--test-label-file', type=str, default=TEST_LABEL_FILE, help='The label file to get the list of names')
    parser.add_argument('--cut-square', type=bool, default=CUT_SQUARE, help='the flag to cut the image into 496 x 496')
    parser.add_argument('--bce-weight', default=BCE_WEIGHT, type=float, help='The weight of BCE loss in DICE loss [0,1], 1 means no dice loss')
    parser.add_argument('--boundary-weight', default=BOUNDARY_WEIGHT, type=float, help='The weight of boundary loss close to the boundary of ground truth of target class')



    flags = parser.parse_args()  # This is for command line version of the code
    # flags = parser.parse_args(args = [])#This is for jupyter notebook version of the code
    # flagsVar = vars(flags)
    return flags
