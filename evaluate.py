"""
This file serves as a training interface for training the network
"""
# Built in
import glob
import os
import shutil
import sys
import time
sys.path.append('../utils/')

# Torch

# Own
import flag_reader
from utils.helper_functions import load_flags
from utils import data_reader
from class_wrapper import Network
from model_maker import ResNetUNet
from utils.helper_functions import put_param_into_folder, write_flags_and_BVE

def evaluate_from_model(model_dir, write_summary_to_csv_flag=True, num=10000, 
                       post_processing=False, ROC=True, save_img=False, save_label=False):
    """
    Training interface. 1. Read data 2. initialize network 3. train network 4. record flags
    :param flag: The training flags read from command line or parameter.py
    :param write_summary_to_csv: Flag to write the summary to a eval_summary.csv with time, model name
    #data inferred, iou and auroc
    :return: None
    """
    if not model_dir.startswith("models"):
        model_dir = os.path.join("models", model_dir)
    flags = load_flags(model_dir)
    flags.model_name = model_dir.replace('models/','')
    print(flags.model_name)

    # Get the data
    train_loader, test_loader = data_reader.read_data(flags)
    print("Making network now")

    # Make Network
    ntwk = Network(ResNetUNet, flags, train_loader, test_loader)

    # Training process
    print("Start training now...")
    iou, auroc = ntwk.evaluate(eval_number_max=num, save_img=save_img, post_processing=post_processing,
                                ROC=ROC, save_label=save_label)
    if write_summary_to_csv_flag:
        write_summary_to_csv(model_dir, num, iou, auroc, post_processing)


def evaluate_all(models_dir="models"):
    """
    This function evaluate all the models in the models/. directory
    :return: None
    """
    for file in os.listdir(models_dir):
        if os.path.isfile(os.path.join(models_dir, file, 'flags.obj')):
            evaluate_from_model(os.path.join(models_dir, file))
    return None


def write_summary_to_csv(model_dir, num_infered, iou, auroc, post_processing, summary_file=os.path.join('data','eval_summary.csv')):
    """
    The function to write the summary information into the data/eval_summary file
    :param model_dir: The model name of the thing to save
    :param num_infered: The number of images this round of evaluation has done
    :param iou: the average iou
    :param auroc: the Aera under the ROC curve
    :param post_processing: Flag of this round did the post-processing or not
    :param summary_file: the file to write, default to be eval_summary.csv under data folder
    """
    # Initialize it if the file is not there
    if not os.path.isfile(summary_file):
        with open(summary_file, 'a') as f:
            f.write("Time  model_name  #point_eval  Post_processing  IOU  AUROC\n")
    with open(summary_file, 'a') as f:
        f.write(time.asctime(time.localtime(time.time())).replace(' ','_')+'  '+\
                model_dir+ '  '+ \
                str(num_infered)+ '  ' +\
                str(post_processing)+ '  '+\
                str(iou)+ '  '+\
                str(auroc)+ '\n')
    

if __name__ == '__main__':
    # Read the parameters to be set
    #flags = flag_reader.read_flag()

    # Call the train from flag function
    #evaluate_from_flag(flags.eval_model)

    evaluate_all()

    # Do the retraining for all the data set to get the training
    #for i in range(10):
    #retrain_different_dataset()
