"""
This file serves as a training interface for training the network
"""
# Built in
import glob
import os
import shutil
import sys
sys.path.append('../utils/')

# Torch

# Own
import flag_reader
from utils.helper_functions import load_flags
from utils import data_reader
from class_wrapper import Network
from model_maker import ResNetUNet
from utils.helper_functions import put_param_into_folder, write_flags_and_BVE

def evaluate_from_flag(model_dir):
    """
    Training interface. 1. Read data 2. initialize network 3. train network 4. record flags
    :param flag: The training flags read from command line or parameter.py
    :return: None
    """
    flags = load_flags(os.path.join("models", model_dir))
    flags.model_name = os.path.join(model_dir)
    print(flags.model_name)

    # Get the data
    train_loader, test_loader = data_reader.read_data(flags)
    print("Making network now")

    # Make Network
    ntwk = Network(ResNetUNet, flags, train_loader, test_loader)

    # Training process
    print("Start training now...")
    ntwk.evaluate(eval_number_max=12, save_img=True, post_processing=False, ROC=False, save_label=True)


if __name__ == '__main__':
    # Read the parameters to be set
    flags = flag_reader.read_flag()

    # Call the train from flag function
    evaluate_from_flag(flags.eval_model)
    # Do the retraining for all the data set to get the training
    #for i in range(10):
    #retrain_different_dataset()
